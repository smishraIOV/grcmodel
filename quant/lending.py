"""The stablecoin lending vault, one period: loans, liquidation, bad debt, and
the firm's books (design.md sections 2.2-2.6 and 3).

**The vault** belongs to its providers. Its size A is in dollars. The firm
lends its stablecoins to borrowers who post BTC worth more than the loan; what
is not lent sits idle and earns nothing. Providers share the vault's gains and
losses.

**The firm** belongs to shareholders. It sets the borrower rate and the
loan-to-value (LTV), takes a share of the interest collected, pays operating
costs and GRC, repays a share of bad debt and incident losses, keeps a stake of
its own money in the vault, and holds its equity as stablecoins (earning
nothing) and BTC, part of the BTC hedged with short perps.

Scenario order: (xi, eps_BTC, crash) -- the vault's operating-loss shock,
the shock to BTC's smooth move (both unit variance and separate: the first
Student-t, the second normal), and the share of BTC's price a crash
takes (zero without one). `lending_rule` builds them. Everything is element-wise, so it vectorizes over
grid points, quadrature nodes and jump outcomes at once.
"""

from dataclasses import dataclass

import numpy as np
import torch

from quant.core import jump_probability, mitigation
from quant.firm import btc_prices, capital_floor, scenario_rule, treasury_close
from quant.numerics import NumericsProfile
from quant.params import Params
from quant.quadrature import JumpRule, ShockRule


def lending_shock_correlation(p: Params) -> np.ndarray:
    return np.eye(2)


def lending_rule(
    p: Params, profile: NumericsProfile, n_ops: int = 3, n_btc: int = 7,
    n_mixing: int = 6, n_crash: int = 5,
) -> ShockRule:
    """Scenarios over (xi, eps_BTC, crash) (`quant.firm.scenario_rule`)."""
    return scenario_rule(
        p, lending_shock_correlation(p), (n_ops, n_btc), profile, n_mixing=n_mixing, n_crash=n_crash
    )


@dataclass(frozen=True)
class LendingControls:
    """One period's decision, taken before the shocks.

    K: the firm's equity after dividends or issuance. g: GRC spend.
    q: the firm's own BTC, bought spot. H: BTC hedged with short perps, at most q.
    rate: the borrower rate for the period. ltv: the loan-to-value at origination.
    """

    K: torch.Tensor
    g: torch.Tensor
    q: torch.Tensor
    H: torch.Tensor
    rate: torch.Tensor
    ltv: torch.Tensor


def loan_demand(p: Params, rate: torch.Tensor, ltv: torch.Tensor) -> torch.Tensor:
    """A straight line in the rate, highest at zero and gone at `rate_max`,
    shifted by how generous the LTV is (design.md section 2.2)."""
    line = p.loan_demand * torch.relu(1.0 - rate / p.rate_max)
    return line * torch.relu(1.0 + p.ltv_demand * (ltv - p.ltv_ref))


def loan_book(p: Params, A: torch.Tensor, c: LendingControls) -> torch.Tensor:
    """The smaller of demand and the stablecoins in the vault."""
    return torch.minimum(loan_demand(p, c.rate, c.ltv), A)


def lending_capital_floor(
    p: Params, A: torch.Tensor, g: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """The firm's capital floor (`quant.firm.capital_floor`), with its stake in
    the lending vault."""
    return capital_floor(p, A, g, q, p.lend_co_invest * A)


def loan_payoff(
    p: Params, rate: torch.Tensor, ltv: torch.Tensor,
    price: torch.Tensor, smooth: torch.Tensor, discount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """What the vault gets back per dollar lent, and the bad debt per dollar.

    The borrower owes O = 1 + rate against collateral of 1 / ltv BTC. The
    **threshold price** is T = O ltv / theta. Prices are watched
    continuously, so a loan whose price ends below T was liquidated **at T**
    -- unless a crash jumped the price past T before anything could be sold,
    in which case the sale is at the price after the crash. Either way the
    collateral sells at a discount d below that price, the firm takes only
    O (1 + penalty), and the rest goes back to the borrower. Bad debt is what
    even the whole collateral cannot cover:

        sale     = T if the smooth part alone crossed T, else min(T, price)
        proceeds = sale (1 - d) / ltv
        payoff   = min(O (1 + penalty), proceeds),    bad debt = (O - proceeds)^+

    Selling at T covers the loan with room to spare whenever d < 1 - theta,
    so bad debt comes only from crashes (design.md section 2.2).

    **Both switches are smoothed** over `liq_band` of the price -- whether the
    loan is liquidated at all, and whether the smooth part crossed T before a
    crash -- so the penalty and the change of sale price do not make the
    payoff jump, which would give the solver a wrong slope in the LTV and the
    rate.
    """
    owed = 1.0 + rate
    trigger = owed * ltv / p.liq_threshold
    liquidated = torch.sigmoid((trigger - price) / p.liq_band)
    crossed_first = torch.sigmoid((trigger - smooth) / p.liq_band)
    sale = crossed_first * trigger + (1.0 - crossed_first) * torch.minimum(trigger, price)
    proceeds = sale * (1.0 - discount) / ltv
    recovered = torch.minimum(owed * (1.0 + p.liq_penalty), proceeds)
    payoff = liquidated * recovered + (1.0 - liquidated) * owed
    bad_debt = torch.relu(owed - proceeds) * liquidated
    return payoff, bad_debt


@dataclass(frozen=True)
class LendingOutcomes:
    """Every next-period outcome of a decision, with its probability.

    Tensors are (..., M) with M = N (1 + J): each continuous shock node,
    without a jump and then with each jump severity. `shocks` is (M, 2).

    wealth: the firm's closing equity. aum: the vault's next size.
    ret: the vault's return to providers, before compensation.
    bad_debt: in dollars. loans: the loan book, (...,).
    """

    wealth: torch.Tensor
    aum: torch.Tensor
    weights: torch.Tensor
    shocks: torch.Tensor
    ret: torch.Tensor
    bad_debt: torch.Tensor
    loans: torch.Tensor

    def expect(self, values: torch.Tensor) -> torch.Tensor:
        return (values * self.weights).sum(-1)


def lending_outcomes(
    p: Params,
    A: torch.Tensor,
    c: LendingControls,
    S_next: torch.Tensor,
    rule: ShockRule,
    jumps: JumpRule | None,
    bias: float | None = None,
) -> LendingOutcomes:
    """Closing firm equity and the vault's next size, over every outcome.

    With L the loan book, P and B the payoff and bad debt per dollar lent
    (`loan_payoff`), J the jump severity and co the firm's stake:

        vault   = (A - L) + L P - A (phi_A m + phi_B s xi) - J A    idle earns nothing
        fee     = f L clamp(P - 1, 0, rate)                         share of interest collected
        R       = (vault - fee) / A - 1                             providers' return, before cover
        cover   = psi (1 - co) (L B + J A)                          the firm repays outsiders
        A'      = A (1 + R) + cover + theta (cap - A)
                  + eta (R + cover / ((1 - co) A) - y_out) A - [jump] omega A,    floored at 0
        w'      = (K - g - q - co A)                                stablecoins, earning nothing
                  + q (1 + R_btc - h_q) - H (R_btc - r) - c_H/2 H^2  BTC, its security cost, the perps
                  - tau K - opex + fee + co A (1 + R) - cover

    `bias` is the firm's expected BTC return above the market's; the firm
    decides with `p.btc_bias`, and its value is measured with 0 (design.md
    section 3.4).
    """
    if bias is None:
        bias = p.btc_bias
    xi, eps, crash = rule.nodes[:, 0], rule.nodes[:, 1], rule.nodes[:, 2]

    def col(x):
        return x.unsqueeze(-1)

    S_col = col(S_next)
    phi_A = mitigation(S_col, p.a_A, p.rho_A)
    phi_B = mitigation(S_col, p.a_B, p.rho_B)
    discount = p.sale_discount * mitigation(S_col, p.a_L, p.rho_L)

    loans = loan_book(p, A, c)
    price, smooth = btc_prices(p, eps, crash, bias)                         # (N,)
    R_btc = price - 1.0
    payoff, bad = loan_payoff(p, col(c.rate), col(c.ltv), price, smooth, discount)  # (..., N)
    L, A1 = col(loans), col(A)
    vault = (A1 - L) + L * payoff - A1 * (phi_A * p.lend_loss + phi_B * p.lend_vol * xi)
    fee = p.interest_share * L * torch.minimum(torch.relu(payoff - 1.0), col(c.rate))
    books = treasury_close(p, c.K, c.g, c.q, c.H, p.lend_co_invest * A, R_btc)    # (..., N)

    if jumps is not None and p.jump_rate > 0.0:
        severity = torch.cat([jumps.severity.new_zeros(1), jumps.severity])    # (1 + J,)
        prob = col(jump_probability(p, S_next))                                 # (..., 1)
        branch = torch.cat([1.0 - prob, prob * jumps.weights], dim=-1)          # (..., 1 + J)
    else:
        severity = A.new_zeros(1)
        branch = torch.ones_like(col(A))
    occurred = (severity > 0).to(A.dtype)

    def jumped(x):
        return x.unsqueeze(-1)                                                   # (..., N, 1)

    A3 = A[..., None, None]
    co = p.lend_co_invest
    hit = severity * A3                                                          # (..., 1, 1 + J)
    end = jumped(vault) - hit
    R = (end - jumped(fee)) / A3 - 1.0
    bad_debt = jumped(L * bad).expand_as(R)
    cover = p.compensation * (1.0 - co) * (bad_debt + hit)
    received = R + cover / ((1.0 - co) * A3)
    aum = torch.relu(
        A3 * (1.0 + R) + cover
        + p.inflow_speed * (p.lend_capacity - A3)
        + p.flow_yield * (received - p.outside_rate) * A3
        - occurred * p.outflow_incident * A3
    )
    wealth = jumped(books + fee) + co * A3 * (1.0 + R) - cover

    weights = rule.weights[:, None] * branch.unsqueeze(-2)                       # (..., N, 1 + J)
    m = rule.nodes.shape[0] * branch.shape[-1]
    shocks = rule.nodes.repeat_interleave(branch.shape[-1], dim=0)

    def flat(t):
        return t.expand(*R.shape).reshape(*R.shape[:-2], m)

    return LendingOutcomes(
        wealth=flat(wealth), aum=flat(aum), weights=flat(weights), shocks=shocks,
        ret=flat(R), bad_debt=flat(bad_debt), loans=loans,
    )
