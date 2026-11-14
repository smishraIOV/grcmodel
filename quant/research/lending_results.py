"""What a solved lending firm says (design.md sections 2.7 and 3).

Everything reads a `LendingSolution`.

- **Cost ratios**: operating costs and payroll as shares of the firm's
  revenue at the terms it chooses -- the first thing a CFO checks against
  the industry. The agreed bounds: payroll at least 30% of revenue, all
  operating costs 50-60%.
"""

from dataclasses import dataclass

import torch

from quant.core import mitigation, next_grc
from quant.firm import btc_prices
from quant.lending import LendingControls, loan_book, loan_payoff
from quant.research.lending_solver import LendingSolution

PAYROLL_FLOOR = 0.30              # payroll as a share of revenue, at least
OPERATING_RANGE = (0.50, 0.60)    # all operating costs as a share of revenue


def state_index(sol: LendingSolution, at_w: float, at_A: float, at_S: float = 0.0) -> int:
    """The grid point nearest (w, S, A)."""
    w, s, A = sol.states()
    return int(torch.argmin((w - at_w).abs() + (s - at_S).abs() + (A - at_A).abs()))


def controls_at(sol: LendingSolution, i: int) -> LendingControls:
    c = sol.controls()
    return LendingControls(*(getattr(c, f)[i:i + 1] for f in ("K", "g", "q", "H", "rate", "ltv")))


@dataclass(frozen=True)
class CostRatios:
    """At one state, per year."""

    revenue: float      # the firm's expected share of the interest collected
    operating: float    # operating costs / revenue
    payroll: float      # payroll / revenue
    rate: float         # the borrower rate chosen
    loans: float        # the loan book

    @property
    def within_bounds(self) -> bool:
        lo, hi = OPERATING_RANGE
        return self.payroll >= PAYROLL_FLOOR and lo <= self.operating <= hi


def cost_ratios(sol: LendingSolution, at_w: float, at_A: float) -> CostRatios:
    """Revenue is the firm's expected share of the interest the vault actually
    collects in a period -- after liquidations and bad debt -- at the terms the
    firm chose in that state, annualized."""
    p = sol.params
    n = sol.inputs.periods_per_year if sol.inputs is not None else 1
    i = state_index(sol, at_w, at_A)
    _, s, A = sol.states()
    with torch.no_grad():
        c = controls_at(sol, i)
        discount = p.sale_discount * mitigation(next_grc(p, s[i:i + 1], c.g), p.a_L, p.rho_L)
        price, smooth = btc_prices(p, sol.rule.nodes[:, 1], sol.rule.nodes[:, 2], p.btc_bias)
        payoff, _ = loan_payoff(p, c.rate, c.ltv, price, smooth, discount)
        loans = loan_book(p, A[i:i + 1], c)
        fee = p.interest_share * loans * torch.minimum(torch.relu(payoff - 1.0), c.rate)
        revenue = float(sol.rule.expect(fee)) * n
    opex = p.opex * n
    if revenue <= 0.0:
        return CostRatios(revenue=0.0, operating=float("inf"), payroll=float("inf"),
                          rate=float(c.rate) * n, loans=float(loans))
    return CostRatios(
        revenue=revenue, operating=opex / revenue, payroll=p.payroll_share * opex / revenue,
        rate=float(c.rate) * n, loans=float(loans),
    )
