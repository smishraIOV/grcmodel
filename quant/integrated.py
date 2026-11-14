"""The integrated firm, one period: both vaults joined through one loan book
(design.md sections 2.1-2.6 and 3; docs/plan-integrated.md section 2).

**One loan book, two sources of funding.** Borrowers take stablecoins against
BTC collateral, at one rate and one LTV the firm sets. The stablecoins come
from the dollar vault (size A_L) and from the BTC vault's **lending sleeve**:
a share a_4 of the BTC vault (size A_B, in dollars) swapped into stablecoins.
The loan book is the smaller of demand and that supply. Each source lends in
proportion to what it makes available, so both have the same share lent, and
interest, penalties, bad debt, operating losses and the firm's interest share
fall on each dollar alike. What is not lent sits idle and earns nothing.

**The sleeve is hedged.** The BTC vault holds a long BTC perp of the sleeve's
dollar size and pays funding on it, r + perp_premium, lent or not. A sleeve
dollar therefore ends the period worth (BTC's price) + (its lending result) -
funding, so in BTC terms it earns (lending result - funding) / price: the
BTC move itself cancels.

**Incidents.** At most one a period (design.md section 4.3), in either vault:
none, one in the lending system, or one in the BTC vault's other strategies.
Each has the chance `jump_probability` of the single-vault models, so with
one vault empty the model is exactly the other vault alone. An incident in
the lending system takes a share of every stablecoin there, the sleeve's
included. Investors leave the vault hit, and some leave the other one too:
both carry the firm's name. That second outflow is scaled by the hit vault's
share of the two, so an incident in an empty vault is no event.

**The firm** earns its share of the interest on the whole loan book and the
BTC vault's fees, pays one set of operating costs and GRC, repays a share of
incident losses and bad debt to outside investors in both vaults, keeps a
stake in each, and holds its own equity under `quant.firm`'s rules.

Scenario order: (xi_L, xi_1, xi_2, eps_BTC, crash) -- the lending system's
operating-loss shock, the two active BTC strategies' yield shocks, BTC's
smooth move, and the share of BTC's price a crash takes.
"""

from dataclasses import dataclass

import numpy as np
import torch

from quant.core import jump_probability, mitigation
from quant.firm import btc_prices, capital_floor, scenario_rule, treasury
from quant.lending import liquidation_terms, loan_demand, loan_payoff
from quant.numerics import NumericsProfile
from quant.params import Params
from quant.quadrature import JumpRule, ShockRule

_TINY = 1e-300   # keeps 0 / 0 out of an empty vault; never changes a non-empty one


def integrated_correlation(p: Params) -> np.ndarray:
    """The two active BTC strategies' yields correlate; the lending
    system's shock is separate from them but shares their Student-t tail;
    BTC's smooth move (the last direction) is separate and normal."""
    corr = np.eye(4)
    corr[1, 2] = corr[2, 1] = p.strategy_rho
    return corr


def integrated_rule(
    p: Params, profile: NumericsProfile, n_ops: int = 2, n_strategy: int = 2, n_btc: int = 7,
    n_mixing: int = 6, n_crash: int = 5, even_btc: bool = False,
) -> ShockRule:
    """Scenarios over (xi_L, xi_1, xi_2, eps_BTC, crash) (`quant.firm.scenario_rule`)."""
    return scenario_rule(
        p, integrated_correlation(p), (n_ops, n_strategy, n_strategy, n_btc), profile,
        n_mixing=n_mixing, n_crash=n_crash, even_btc=even_btc,
    )


@dataclass(frozen=True)
class IntegratedControls:
    """One period's decision, taken before the shocks.

    K: the firm's equity after dividends or issuance. g: GRC spend.
    q: the firm's own BTC, bought spot. H: BTC hedged with short perps, at most
    the firm's BTC including its stake in the BTC vault.
    rate, ltv: the loan book's terms. alloc: the BTC vault's shares in lending
    BTC, basis trading, cold custody and the lending sleeve, summing to one,
    shape (..., 4).
    """

    K: torch.Tensor
    g: torch.Tensor
    q: torch.Tensor
    H: torch.Tensor
    rate: torch.Tensor
    ltv: torch.Tensor
    alloc: torch.Tensor


def integrated_capital_floor(
    p: Params, A_L: torch.Tensor, A_B: torch.Tensor, g: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """The firm's capital floor (`quant.firm.capital_floor`): own funds against
    the vaults' combined size (design.md section 2.6), and its stakes in both."""
    return capital_floor(p, A_L + A_B, g, q, p.lend_co_invest * A_L + p.btc_co_invest * A_B)


def integrated_btc_held(p: Params, A_B: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """BTC the firm owns, and may hedge: its own plus its stake in the BTC vault."""
    return q + p.btc_co_invest * A_B


def loan_supply(A_L: torch.Tensor, A_B: torch.Tensor, alloc: torch.Tensor) -> torch.Tensor:
    """Stablecoins the two sources make available: A_L + a_4 A_B."""
    return A_L + alloc[..., 3] * A_B


def integrated_loan_book(
    p: Params, A_L: torch.Tensor, A_B: torch.Tensor, c: IntegratedControls,
    demand_scale: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The loan book, and the share of the supply that is lent (the same for
    both sources: pro-rata sharing, design.md section 2.1). `demand_scale`
    multiplies loan demand."""
    supply = loan_supply(A_L, A_B, c.alloc)
    loans = torch.minimum(loan_demand(p, c.rate, c.ltv) * demand_scale, supply)
    return loans, loans / torch.clamp(supply, min=_TINY)


@dataclass(frozen=True)
class IntegratedStep:
    """One period's results at given shocks. Every tensor has the shape the
    arguments of `integrated_step` broadcast to. Fields as in
    `IntegratedOutcomes`; `price` is BTC's at the close."""

    wealth: torch.Tensor
    aum_l: torch.Tensor
    aum_b: torch.Tensor
    ret_l: torch.Tensor
    ret_b: torch.Tensor
    yield_b: torch.Tensor
    received_l: torch.Tensor
    received_b: torch.Tensor
    interest: torch.Tensor
    fees_b: torch.Tensor
    cover: torch.Tensor
    bad_debt: torch.Tensor
    loans: torch.Tensor
    use: torch.Tensor
    price: torch.Tensor


def integrated_step(
    p: Params,
    A_L: torch.Tensor,
    A_B: torch.Tensor,
    c: IntegratedControls,
    S_next: torch.Tensor,
    xi_l: torch.Tensor,
    xi_1: torch.Tensor,
    xi_2: torch.Tensor,
    eps: torch.Tensor,
    crash: torch.Tensor,
    sev_l: torch.Tensor,
    sev_b: torch.Tensor,
    bias: float | None = None,
    demand_scale: torch.Tensor | float = 1.0,
) -> IntegratedStep:
    """Closing firm equity and both vaults' next sizes, at given shocks.

    With u the share of the supply lent, P and B the payoff and bad debt per
    dollar lent (`quant.lending.loan_payoff`), ops = phi_A m + phi_B s xi_L
    the lending system's operating loss per dollar, f the firm's interest
    share, J_L and J_B the incident severities (at most one is non-zero), and
    X the BTC price at the close (it starts at 1):

    Per stablecoin dollar in the lending system, from either source:

        earned = clamp(P - 1, 0, rate)                       interest collected
        gain   = u (P - 1) - ops - f u earned - J_L          after the firm's share

    The dollar vault (co_L the firm's stake), as in `quant.lending`:

        R_L     = gain                                       providers' return
        cover_L = psi (1 - co_L) (u B + J_L) A_L
        A_L'    = A_L (1 + R_L) + cover_L + theta (cap_L - A_L)
                  + eta_L (R_L + cover_L / ((1 - co_L) A_L) - y_out) A_L - [J_L > 0] omega A_L
                  - [J_B > 0] omega_c A_B / (A_L + A_B) A_L       investors leave the other vault too

    The BTC vault, in BTC per unit of its BTC, a the allocation:

        y_4     = (gain - (r + premium)) / X                 the sleeve: lending less funding
        y       = sum_{i<=3} a_i (mu_i - phi_A m_i - c_i C) + phi_B (a_1 s_1 xi_1 + a_2 s_2 xi_2)
                  + a_4 y_4 - J_B sum_{i<=3} a_i j_i
        fee     = f_m + f_p max(y, 0)
        R_B     = y - fee
        cover_B = psi (1 - co_B) (J_B sum_{i<=3} a_i j_i + a_4 (u B + J_L) / X)
        A_B'    = A_B (1 + R_B + cover_B) X + theta (cap_B - A_B)
                  + eta_B (R_B + cover_B / (1 - co_B) - y_out_B) A_B - [J_B > 0] omega A_B
                  - [J_L > 0] omega_c A_L / (A_L + A_B) A_B

    The firm, with one set of operating costs inside `treasury_close`:

        w' = treasury_close(stake = co_L A_L + co_B A_B)
             + f u earned (A_L + a_4 A_B)                    interest share, whole loan book
             + co_L A_L (1 + R_L) - cover_L
             + (fee + co_B (1 + R_B) - cover_B) A_B X

    Both next sizes are floored at zero. Where the sleeve's dollar results are
    put in BTC terms, X is floored at `price_floor`.

    `bias` is the firm's expected BTC return above the market's; the firm
    decides with `p.btc_bias`, and its value is measured with 0.

    **Element-wise.** The state, the decisions, the stock S_next, the shocks
    (xi_L, xi_1, xi_2, eps, crash) and the incident severities (J_L, J_B, at
    most one non-zero) broadcast together; `c.alloc` carries its four shares
    on a last axis. `integrated_outcomes` lays them out over every scenario
    and incident; a simulated path passes one shock per path. `demand_scale`
    multiplies loan demand (1 unless a caller moves demand with BTC's price).
    """
    if bias is None:
        bias = p.btc_bias
    a1, a2, a3, a4 = (c.alloc[..., k] for k in range(4))
    phi_A = mitigation(S_next, p.a_A, p.rho_A)
    phi_B = mitigation(S_next, p.a_B, p.rho_B)
    discount, slippage = liquidation_terms(p, S_next)

    price, smooth = btc_prices(p, eps, crash, bias)
    R_btc = price - 1.0
    in_btc = 1.0 / torch.clamp(price, min=p.price_floor)                     # dollars to BTC, at the close

    # -- the loan book, per stablecoin dollar from either source ------------
    supply = loan_supply(A_L, A_B, c.alloc)
    loans, use = integrated_loan_book(p, A_L, A_B, c, demand_scale)
    payoff, bad = loan_payoff(p, c.rate, c.ltv, price, smooth, discount, slippage)
    earned = torch.minimum(torch.relu(payoff - 1.0), c.rate)
    ops = phi_A * p.lend_loss + phi_B * p.lend_vol * xi_l
    interest_per = p.interest_share * use * earned
    gain = use * (payoff - 1.0) - ops - interest_per                         # before incidents
    lost = use * bad

    hit_l = (sev_l > 0).to(price.dtype)
    hit_b = (sev_b > 0).to(price.dtype)
    co_l, co_b = p.lend_co_invest, p.btc_co_invest
    # Each vault's share of the two: how much an incident in it weighs on the firm's name.
    both = torch.clamp(A_L + A_B, min=_TINY)
    share_l, share_b = A_L / both, A_B / both

    # -- the dollar vault -----------------------------------------------------
    R_l = gain - sev_l
    cover_l_per = p.compensation * (1.0 - co_l) * (lost + sev_l)             # per vault dollar
    received_l = R_l + cover_l_per / (1.0 - co_l)                            # what an outside provider gets
    aum_l = torch.relu(
        A_L * (1.0 + R_l) + cover_l_per * A_L
        + p.inflow_speed * (p.lend_capacity - A_L)
        + p.flow_yield * (received_l - p.outside_rate) * A_L
        - hit_l * p.outflow_incident * A_L
        - hit_b * p.outflow_contagion * share_b * A_L
    )

    # -- the BTC vault --------------------------------------------------------
    mu, loss, vol = p.strategy_yield, p.strategy_loss, p.strategy_vol
    in_crash, exposure = p.strategy_crash, p.strategy_incident
    own = (a1, a2, a3)
    drift = sum(a * (mu[k] - phi_A * loss[k]) for k, a in enumerate(own))
    risky = phi_B * (a1 * vol[0] * xi_1 + a2 * vol[1] * xi_2)
    crashed = sum(a * in_crash[k] for k, a in enumerate(own)) * crash
    funding = p.r + p.perp_premium
    sleeve = a4 * (gain - funding) * in_btc
    exposed = sum(a * exposure[k] for k, a in enumerate(own)) * sev_b
    y = drift + risky - crashed + sleeve - exposed - a4 * sev_l * in_btc
    fee_b = p.fee_mgmt + p.fee_perf * torch.relu(y)
    R_b = y - fee_b
    cover_b = p.compensation * (1.0 - co_b) * (exposed + a4 * (lost + sev_l) * in_btc)
    received_b = R_b + cover_b / (1.0 - co_b)                                # an outside investor, in BTC
    aum_b = torch.relu(
        A_B * (1.0 + R_b + cover_b) * price
        + p.inflow_speed * (p.btc_capacity - A_B)
        + p.btc_flow_yield * (received_b - p.btc_outside_yield) * A_B
        - hit_b * p.outflow_incident * A_B
        - hit_l * p.outflow_contagion * share_l * A_B
    )

    # -- the firm -------------------------------------------------------------
    books = treasury(p, c.K, c.g, c.q, c.H, co_l * A_L + co_b * A_B, R_btc)
    interest = interest_per * supply
    fees_b = fee_b * A_B * price
    cover = cover_l_per * A_L + cover_b * A_B * price
    wealth = (
        books + interest + co_l * A_L * (1.0 + R_l)
        + fees_b + co_b * (1.0 + R_b) * A_B * price - cover
    )
    return IntegratedStep(
        wealth=wealth, aum_l=aum_l, aum_b=aum_b, ret_l=R_l, ret_b=R_b, yield_b=y,
        received_l=received_l, received_b=received_b, interest=interest, fees_b=fees_b,
        cover=cover, bad_debt=lost * supply, loans=loans, use=use, price=price,
    )


@dataclass(frozen=True)
class IntegratedOutcomes:
    """Every next-period outcome of a decision, with its probability.

    Tensors are (..., M) with M = N (1 + 2J): each market scenario with no
    incident, then with each severity of a lending-system incident, then with
    each severity of a BTC-vault incident. `shocks` is (M, 5).

    wealth: the firm's closing equity. aum_l, aum_b: the vaults' next sizes,
    in dollars. ret_l: the dollar vault's return to providers; ret_b: the BTC
    vault's BTC return to investors; both after fees, before compensation.
    yield_b: the BTC vault's BTC yield before its fees.
    received_l, received_b: what an outside investor in each vault receives,
    compensation included (the BTC vault's in BTC).
    interest: the firm's share of the interest, dollars. fees_b: the BTC
    vault's fees at the closing price, dollars. cover: what the firm repays
    investors, dollars. bad_debt: on the whole loan book, dollars.
    loans, use: the loan book and the share of supply lent, (...,).
    """

    wealth: torch.Tensor
    aum_l: torch.Tensor
    aum_b: torch.Tensor
    weights: torch.Tensor
    shocks: torch.Tensor
    ret_l: torch.Tensor
    ret_b: torch.Tensor
    yield_b: torch.Tensor
    received_l: torch.Tensor
    received_b: torch.Tensor
    interest: torch.Tensor
    fees_b: torch.Tensor
    cover: torch.Tensor
    bad_debt: torch.Tensor
    loans: torch.Tensor
    use: torch.Tensor

    def expect(self, values: torch.Tensor) -> torch.Tensor:
        return (values * self.weights).sum(-1)


def integrated_outcomes(
    p: Params,
    A_L: torch.Tensor,
    A_B: torch.Tensor,
    c: IntegratedControls,
    S_next: torch.Tensor,
    rule: ShockRule,
    jumps: JumpRule | None,
    bias: float | None = None,
) -> IntegratedOutcomes:
    """`integrated_step` over every outcome: each market scenario of `rule`,
    with no incident and then each incident size in either vault, and the
    probability of each. The state and decisions are (...,); the results are
    (..., M) (see `IntegratedOutcomes`)."""
    def wide(x):
        return x[..., None, None]                                            # (..., 1, 1)

    def node(k):
        return rule.nodes[:, k].unsqueeze(-1)                                # (N, 1)

    # -- incidents: none, lending system, BTC vault -------------------------
    if jumps is not None and p.jump_rate > 0.0:
        zero = jumps.severity.new_zeros(jumps.severity.shape[0])
        sev_l = torch.cat([zero[:1], jumps.severity, zero])                  # (1 + 2J,)
        sev_b = torch.cat([zero[:1], zero, jumps.severity])
        prob = jump_probability(p, S_next).unsqueeze(-1)                     # (..., 1)
        branch = torch.cat([1.0 - 2.0 * prob, prob * jumps.weights, prob * jumps.weights], dim=-1)
    else:
        sev_l = sev_b = A_L.new_zeros(1)
        branch = torch.ones_like((A_L + A_B).unsqueeze(-1))

    spread = IntegratedControls(
        K=wide(c.K), g=wide(c.g), q=wide(c.q), H=wide(c.H), rate=wide(c.rate), ltv=wide(c.ltv),
        alloc=c.alloc[..., None, None, :],
    )
    step = integrated_step(
        p, wide(A_L), wide(A_B), spread, wide(S_next), node(0), node(1), node(2), node(3), node(4),
        sev_l, sev_b, bias=bias,
    )

    weights = rule.weights[:, None] * branch.unsqueeze(-2)                   # (..., N, 1 + 2J)
    m = rule.nodes.shape[0] * branch.shape[-1]
    shocks = rule.nodes.repeat_interleave(branch.shape[-1], dim=0)
    shape = torch.broadcast_shapes(step.wealth.shape, step.aum_l.shape, step.aum_b.shape, weights.shape)

    def flat(t):
        return t.expand(shape).reshape(*shape[:-2], m)

    return IntegratedOutcomes(
        wealth=flat(step.wealth), aum_l=flat(step.aum_l), aum_b=flat(step.aum_b), weights=flat(weights),
        shocks=shocks, ret_l=flat(step.ret_l), ret_b=flat(step.ret_b), yield_b=flat(step.yield_b),
        received_l=flat(step.received_l), received_b=flat(step.received_b),
        interest=flat(step.interest), fees_b=flat(step.fees_b), cover=flat(step.cover),
        bad_debt=flat(step.bad_debt), loans=step.loans[..., 0, 0], use=step.use[..., 0, 0],
    )
