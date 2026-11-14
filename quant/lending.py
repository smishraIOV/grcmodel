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

Shock vector order: (xi, eps_BTC) -- the vault's operating-loss shock and
BTC's return shock. Both unit variance; uncorrelated in the normal part, they
share the Student-t's tail. Everything is element-wise, so it vectorizes over
grid points, quadrature nodes and jump outcomes at once.
"""

from dataclasses import dataclass

import numpy as np
import torch

from quant.core import jump_probability, mitigation
from quant.params import Params
from quant.quadrature import JumpRule, ShockRule


def lending_shock_correlation(p: Params) -> np.ndarray:
    return np.eye(2)


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
    """The least equity the firm may hold.

    Own funds against the vault and the firm's BTC, hedged or not:
    K >= k_A A + k_BTC q. Non-negative stablecoins after GRC, BTC and the
    stake in the vault: K >= g + q + co A. And no negative equity.
    """
    cash = g + q + p.lend_co_invest * A
    return torch.maximum(torch.maximum(p.k_A * A + p.k_BTC * q, cash), torch.zeros_like(A))


def btc_return(p: Params, eps: torch.Tensor, bias: float) -> torch.Tensor:
    """BTC's dollar return over the period: the market's expectation r_E, plus
    `bias`, plus the shock. Floored at -1: a price cannot fall below zero."""
    return torch.clamp(p.r_E + bias + p.s_BTC * eps, min=-1.0)


def loan_payoff(
    p: Params, rate: torch.Tensor, ltv: torch.Tensor, R_btc: torch.Tensor, discount: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """What the vault gets back per dollar lent, and the bad debt per dollar.

    The borrower owes O = 1 + rate against collateral now worth
    V = (1 + R_btc) / ltv. Above the threshold price (O < theta V) the loan is
    repaid. Below it the collateral is liquidated: the firm sells only what it
    needs to recover O (1 + penalty), at a discount d below market, and the
    rest goes back to the borrower. Bad debt is what even the whole
    collateral, sold at that discount, cannot cover: (O - V (1 - d))^+.

    **The trigger is smoothed.** The penalty makes the payoff jump at the
    threshold, which gives the solver a wrong slope in the LTV and the rate.
    The share liquidated rises along a logistic curve over a narrow band of
    the BTC price (`liq_band`, 1% of the starting price) around the threshold
    price O ltv / theta (design.md section 2.2). Bad debt starts well past the
    threshold, where the share is one.
    """
    owed = 1.0 + rate
    price = 1.0 + R_btc
    collateral = price / ltv
    proceeds = collateral * (1.0 - discount)
    trigger = owed * ltv / p.liq_threshold
    liquidated = torch.sigmoid((trigger - price) / p.liq_band)
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
                  + q (1 + R_btc) - H (R_btc - r) - c_H/2 H^2       BTC and the short perps
                  - tau K - opex + fee + co A (1 + R) - cover

    `bias` is the firm's expected BTC return above the market's; the firm
    decides with `p.btc_bias`, and its value is measured with 0 (design.md
    section 3.4).
    """
    if bias is None:
        bias = p.btc_bias
    xi = rule.nodes[:, 0]
    eps = rule.nodes[:, 1]

    def col(x):
        return x.unsqueeze(-1)

    S_col = col(S_next)
    phi_A = mitigation(S_col, p.a_A, p.rho_A)
    phi_B = mitigation(S_col, p.a_B, p.rho_B)
    discount = p.sale_discount * mitigation(S_col, p.a_L, p.rho_L)

    loans = loan_book(p, A, c)
    R_btc = btc_return(p, eps, bias)                                       # (N,)
    payoff, bad = loan_payoff(p, col(c.rate), col(c.ltv), R_btc, discount)  # (..., N)
    L, A1 = col(loans), col(A)
    vault = (A1 - L) + L * payoff - A1 * (phi_A * p.lend_loss + phi_B * p.lend_vol * xi)
    fee = p.interest_share * L * torch.minimum(torch.relu(payoff - 1.0), col(c.rate))
    btc = (col(c.q) * (1.0 + R_btc) - col(c.H) * (R_btc - p.r)
           - 0.5 * p.c_H * col(c.H) * col(c.H))
    own = col((c.K - c.g - c.q - p.lend_co_invest * A) - p.tau * c.K - p.opex)

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
    wealth = jumped(own + btc + fee) + co * A3 * (1.0 + R) - cover

    weights = rule.weights[:, None] * branch.unsqueeze(-2)                       # (..., N, 1 + J)
    m = rule.nodes.shape[0] * branch.shape[-1]
    shocks = rule.nodes.repeat_interleave(branch.shape[-1], dim=0)

    def flat(t):
        return t.expand(*R.shape).reshape(*R.shape[:-2], m)

    return LendingOutcomes(
        wealth=flat(wealth), aum=flat(aum), weights=flat(weights), shocks=shocks,
        ret=flat(R), bad_debt=flat(bad_debt), loans=loans,
    )
