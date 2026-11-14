"""What a solved integrated firm says (design.md sections 2.7 and 3;
docs/plan-integrated.md steps 4a and 5).

Everything reads an `IntegratedSolution`, at one state: the grid point
nearest the equity, GRC stock and vault sizes asked for. Money is in the
model's unit (the dollar vault's capacity = 1); rates and flows are per year.

- **The read-out** (`readout`): what the firm decides there, what it expects
  to earn and spend over the next period, what its two groups of investors
  expect to earn against what they could earn elsewhere, and how likely the
  period is to end badly.
- **Cost ratios** (`ReadOut.operating_ratio`, `payroll_ratio`): operating
  costs and payroll against **combined revenue** -- the interest share plus
  the BTC vault's fees -- with the agreed bounds: payroll at least 30% of
  revenue, all operating costs 50-60%.
"""

from dataclasses import dataclass

import torch

from quant.core import issuance_cost, next_grc
from quant.integrated import integrated_capital_floor, integrated_outcomes
from quant.research.integrated_solver import IntegratedSolution

PAYROLL_FLOOR = 0.30              # payroll as a share of revenue, at least
OPERATING_RANGE = (0.50, 0.60)    # all operating costs as a share of revenue
STRATEGIES = ("lending BTC", "basis trading", "cold custody", "stablecoin lending sleeve")


def state_index(
    sol: IntegratedSolution, at_w: float, at_L: float, at_B: float, at_S: float = 0.0
) -> int:
    """The grid point nearest (w, S, A_L, A_B), each axis on its own."""
    index = 0
    for axis, target in zip(sol.grid.axes, (at_w, at_S, at_L, at_B), strict=True):
        index = index * axis.shape[0] + int(torch.argmin((axis - target).abs()))
    return index


@dataclass(frozen=True)
class ReadOut:
    """One state of a solved firm. Rates and flows per year; `periods` a year."""

    periods: int
    # -- the state ---------------------------------------------------------
    equity: float               # w, before this period's dividend or issue
    grc_stock: float
    dollar_vault: float
    btc_vault: float
    value: float                # V: what the firm is worth to shareholders, at its own BTC view
    # -- capital -----------------------------------------------------------
    equity_kept: float          # K
    capital_floor: float        # the least it may hold, given its other decisions
    dividend: float             # this period
    issue: float                # this period
    stakes: float               # its own money in the two vaults
    own_btc: float              # q
    hedged: float               # H, short perps
    stablecoins: float          # what is left of K
    # -- GRC ---------------------------------------------------------------
    grc_spend: float            # per year
    incident_chance: float      # per year, either vault, after the spend
    # -- the business ------------------------------------------------------
    borrower_rate: float        # per year
    ltv: float
    loans: float
    share_lent: float           # of the stablecoins the two sources supply
    allocation: tuple[float, ...]   # the BTC vault's four shares (STRATEGIES)
    # -- what it expects to earn and spend, per year --------------------------
    interest_income: float      # its share of the interest
    btc_vault_fees: float       # at the closing BTC price
    operating_costs: float
    compensation: float         # expected repayments to investors
    # -- its investors, per year ----------------------------------------------
    dollar_yield: float         # providers' expected return, with compensation
    dollar_outside: float
    btc_yield: float            # BTC investors' expected BTC return, with compensation
    btc_outside: float
    expected_bad_debt: float    # on the loan book, dollars
    # -- how the period may end -----------------------------------------------
    chance_below_floor: float   # closing equity below this period's capital floor
    chance_walk_away: float     # closing in a state where shareholders walk away
    expected_shortfall: float   # expected gap below that floor at the close, dollars

    payroll_share: float        # of operating costs

    @property
    def revenue(self) -> float:
        return self.interest_income + self.btc_vault_fees

    @property
    def operating_ratio(self) -> float:
        return self.operating_costs / self.revenue if self.revenue > 0 else float("inf")

    @property
    def payroll_ratio(self) -> float:
        return self.payroll_share * self.operating_ratio

    @property
    def within_bounds(self) -> bool:
        lo, hi = OPERATING_RANGE
        return self.payroll_ratio >= PAYROLL_FLOOR and lo <= self.operating_ratio <= hi


def readout(
    sol: IntegratedSolution, at_w: float, at_L: float, at_B: float, at_S: float = 0.0
) -> ReadOut:
    p, grid = sol.params, sol.grid
    n = sol.inputs.periods_per_year if sol.inputs is not None else 1
    i = state_index(sol, at_w, at_L, at_B, at_S)
    w, s, A_L, A_B = (x[i:i + 1] for x in sol.states())
    with torch.no_grad():
        full = sol.controls()
        c = type(full)(**{k: getattr(full, k)[i:i + 1] for k in ("K", "g", "q", "H", "rate", "ltv", "alloc")})
        s_next = next_grc(p, s, c.g)
        out = integrated_outcomes(p, A_L, A_B, c, s_next, sol.rule, sol.jumps)
        floor = integrated_capital_floor(p, A_L, A_B, c.g, c.q)
        V_next = grid.interpolate(
            sol.V, [out.wealth, s_next.unsqueeze(-1), out.aum_l, out.aum_b], lambda x: issuance_cost(p, x)
        )

        def mean(x):
            return float(out.expect(x))

        size = sol.rule.size
        no_incident = float(out.weights.reshape(size, -1)[:, 0].sum())
        short = torch.relu(floor.unsqueeze(-1) - out.wealth)
        stake = p.lend_co_invest * float(A_L) + p.btc_co_invest * float(A_B)
        return ReadOut(
            periods=n,
            equity=float(w), grc_stock=float(s), dollar_vault=float(A_L), btc_vault=float(A_B),
            value=float(sol.V.reshape(-1)[i]),
            equity_kept=float(c.K), capital_floor=float(floor),
            dividend=max(float(w - c.K), 0.0), issue=max(float(c.K - w), 0.0),
            stakes=stake, own_btc=float(c.q), hedged=float(c.H),
            stablecoins=float(c.K - c.g - c.q) - stake,
            grc_spend=float(c.g) * n,
            incident_chance=1.0 - no_incident ** n,
            borrower_rate=(1.0 + float(c.rate)) ** n - 1.0, ltv=float(c.ltv),
            loans=float(out.loans), share_lent=float(out.use),
            allocation=tuple(float(x) for x in c.alloc[0]),
            interest_income=mean(out.interest) * n, btc_vault_fees=mean(out.fees_b) * n,
            operating_costs=p.opex * n, compensation=mean(out.cover) * n,
            dollar_yield=(1.0 + mean(out.received_l)) ** n - 1.0,
            dollar_outside=(1.0 + p.outside_rate) ** n - 1.0,
            btc_yield=(1.0 + mean(out.received_b)) ** n - 1.0,
            btc_outside=(1.0 + p.btc_outside_yield) ** n - 1.0,
            expected_bad_debt=mean(out.bad_debt) * n,
            chance_below_floor=mean((short > 0).to(out.wealth.dtype)),
            chance_walk_away=mean((V_next == 0).to(out.wealth.dtype)),
            expected_shortfall=mean(short),
            payroll_share=p.payroll_share,
        )
