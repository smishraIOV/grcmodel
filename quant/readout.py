"""What one period holds for a firm under a given policy (design.md sections
2.7 and 3; docs/plan-tool.md step 3).

`read_out` takes a firm (`quant.profile.ModelFirm`) and a policy
(`quant.policy.Policy`) and runs the one-period accounting
(`quant.integrated`) over a fixed set of weighted scenarios. Nothing is
searched for and nothing is random: the same inputs give the same figures.

Everything is in **dollars**; flows and rates are **per year**, and the
chances under "how the period may end" are for the **next period** (a
quarter, at four decisions a year). Expectations are at the firm's own view
of BTC (`btc_view`).

The figures are only as good as the profile's assumptions, which are
illustrative until calibrated.
"""

from dataclasses import dataclass

import torch

from quant.core import next_grc
from quant.integrated import integrated_capital_floor, integrated_outcomes, integrated_rule
from quant.numerics import REFERENCE, NumericsProfile
from quant.params import Params
from quant.policy import ALLOCATION, Policy, position, to_controls
from quant.profile import ModelFirm
from quant.quadrature import JumpRule, ShockRule, pareto_jump_rule

PAYROLL_FLOOR = 0.30              # payroll as a share of revenue, at least (design.md section 2.7)
OPERATING_RANGE = (0.50, 0.60)    # all operating costs as a share of revenue


@dataclass(frozen=True)
class ScenarioSet:
    """The weighted market scenarios and incident sizes one period is
    averaged over (design.md section 5)."""

    rule: ShockRule
    jumps: JumpRule | None


def scenario_set(
    per_period: Params, profile: NumericsProfile = REFERENCE, n_ops: int = 2, n_strategy: int = 2,
    n_btc: int = 61, n_mixing: int = 24, n_crash: int = 24, n_jump: int = 12,
) -> ScenarioSet:
    """Scenarios for one firm at one state. A single state is cheap, so these
    are far finer than a solver's: BTC's direction, where liquidations and
    bad debt come from, gets the most points. The counts are where the
    figures stop moving (tests/test_readout.py): a set as coarse as a
    solver's misstates investors' yields by a point or more."""
    p = per_period
    rule = integrated_rule(p, profile, n_ops=n_ops, n_strategy=n_strategy, n_btc=n_btc,
                           n_mixing=n_mixing, n_crash=n_crash)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, profile, n=n_jump) if p.jump_rate > 0 else None
    return ScenarioSet(rule=rule, jumps=jumps)


@dataclass(frozen=True)
class ReadOut:
    """One firm, one policy, one period ahead. Dollars; rates and flows per year."""

    decisions_per_year: int
    # -- where the firm stands ---------------------------------------------------
    equity: float               # before this period's payout or raise
    dollar_vault: float
    btc_vault: float
    # -- capital -----------------------------------------------------------------
    equity_kept: float
    capital_needed: float       # the least it may keep, given its other decisions
    payout: float               # now
    raised: float               # now
    stakes: float               # its own money in the two vaults
    own_btc: float
    hedged: float
    stablecoins: float
    # -- GRC ---------------------------------------------------------------------
    grc_spend: float            # a year
    incident_chance: float      # a year, in either vault, after the spend
    # -- the loan book -------------------------------------------------------------
    borrower_rate: float
    ltv: float
    loans: float
    share_lent: float           # of the stablecoins the two sources supply
    expected_bad_debt: float    # a year
    # -- the BTC vault -------------------------------------------------------------
    allocation: tuple[float, float, float, float]   # in the order of quant.policy.ALLOCATION
    # -- expected over a year --------------------------------------------------------
    interest_income: float      # the firm's share of the interest
    btc_vault_fees: float       # at the closing BTC price
    operating_costs: float
    payroll: float
    compensation: float         # repaid to investors
    # -- investors, expected a year -----------------------------------------------------
    dollar_yield: float         # what an outside provider receives
    dollar_outside: float       # and could earn elsewhere
    btc_yield: float            # what an outside BTC investor receives, in BTC
    btc_outside: float
    dollar_vault_next: float    # the vaults' expected sizes at the period's end
    btc_vault_next: float
    # -- how the period may end ----------------------------------------------------------
    expected_equity: float      # at the period's end
    chance_of_loss: float       # ending with less equity than was kept
    chance_below_needed: float  # ending below this period's capital needed
    expected_shortfall: float   # the expected gap below it

    @property
    def revenue(self) -> float:
        return self.interest_income + self.btc_vault_fees

    @property
    def operating_ratio(self) -> float:
        return self.operating_costs / self.revenue if self.revenue > 0 else float("inf")

    @property
    def payroll_ratio(self) -> float:
        return self.payroll / self.revenue if self.revenue > 0 else float("inf")

    @property
    def costs_within_bounds(self) -> bool:
        lo, hi = OPERATING_RANGE
        return self.payroll_ratio >= PAYROLL_FLOOR and lo <= self.operating_ratio <= hi


def read_out(
    model: ModelFirm, policy: Policy, scenarios: ScenarioSet | None = None,
    profile: NumericsProfile = REFERENCE,
) -> ReadOut:
    """The read-out for `policy`, which is refused if unsound
    (`quant.policy.check`). `scenarios` defaults to `scenario_set`."""
    annual, scale = model.params, model.scale
    n = annual.periods_per_year
    p = annual.per_period()
    c = to_controls(policy, model, profile)
    if scenarios is None:
        scenarios = scenario_set(p, profile)
    pos = position(policy, model)

    def one(x: float) -> torch.Tensor:
        return profile.tensor([x])

    A_L, A_B = one(model.state.A_L), one(model.state.A_B)
    with torch.no_grad():
        s_next = next_grc(p, one(model.state.S), c.g)
        out = integrated_outcomes(p, A_L, A_B, c, s_next, scenarios.rule, scenarios.jumps)
        needed = integrated_capital_floor(p, A_L, A_B, c.g, c.q)

        def mean(x: torch.Tensor) -> float:
            return float(out.expect(x))

        def chance(event: torch.Tensor) -> float:
            return mean(event.to(out.wealth.dtype))

        no_incident = float(out.weights.reshape(scenarios.rule.size, -1)[:, 0].sum())
        short = torch.relu(needed.unsqueeze(-1) - out.wealth)
        return ReadOut(
            decisions_per_year=n,
            equity=model.state.w * scale, dollar_vault=model.state.A_L * scale,
            btc_vault=model.state.A_B * scale,
            equity_kept=policy.equity_kept, capital_needed=float(needed) * scale,
            payout=pos.payout, raised=pos.raised, stakes=pos.stakes,
            own_btc=policy.own_btc, hedged=policy.hedged, stablecoins=pos.stablecoins,
            grc_spend=policy.grc_spend, incident_chance=1.0 - no_incident ** n,
            borrower_rate=policy.borrower_rate, ltv=policy.ltv,
            loans=float(out.loans) * scale, share_lent=float(out.use),
            expected_bad_debt=mean(out.bad_debt) * n * scale,
            allocation=tuple(policy.allocation),
            interest_income=mean(out.interest) * n * scale,
            btc_vault_fees=mean(out.fees_b) * n * scale,
            operating_costs=annual.opex * scale, payroll=annual.payroll_share * annual.opex * scale,
            compensation=mean(out.cover) * n * scale,
            dollar_yield=(1.0 + mean(out.received_l)) ** n - 1.0, dollar_outside=annual.outside_rate,
            btc_yield=(1.0 + mean(out.received_b)) ** n - 1.0, btc_outside=annual.btc_outside_yield,
            dollar_vault_next=mean(out.aum_l) * scale, btc_vault_next=mean(out.aum_b) * scale,
            expected_equity=mean(out.wealth) * scale,
            chance_of_loss=chance(out.wealth < c.K.unsqueeze(-1)),
            chance_below_needed=chance(short > 0),
            expected_shortfall=mean(short) * scale,
        )


def report(r: ReadOut) -> str:
    """The read-out as plain text."""
    def usd(x: float) -> str:
        return f"{'-' if x < 0 else ''}${abs(x):>13,.0f}"

    def pct(x: float) -> str:
        return f"{100 * x:13.2f}%"

    lo, hi = OPERATING_RANGE
    period = {1: "year", 2: "half-year", 4: "quarter", 12: "month"}.get(
        r.decisions_per_year, f"period (1/{r.decisions_per_year} of a year)"
    )
    lines = [
        f"The firm: equity {usd(r.equity).strip()}, dollar vault {usd(r.dollar_vault).strip()}, "
        f"BTC vault {usd(r.btc_vault).strip()}",
        "",
        "Capital",
        f"  equity kept                      {usd(r.equity_kept)}   (needs at least {usd(r.capital_needed).strip()})",
        f"  paid out now                     {usd(r.payout)}",
        f"  raised now                       {usd(r.raised)}",
        f"  stakes in the two vaults         {usd(r.stakes)}",
        f"  own BTC                          {usd(r.own_btc)}   (hedged {usd(r.hedged).strip()})",
        f"  stablecoins                      {usd(r.stablecoins)}",
        "",
        "GRC",
        f"  spend a year                     {usd(r.grc_spend)}",
        f"  chance of an incident in a year  {pct(r.incident_chance)}",
        "",
        "The loan book",
        f"  borrower rate                    {pct(r.borrower_rate)}",
        f"  loan-to-value                    {pct(r.ltv)}",
        f"  loans                            {usd(r.loans)}   ({100 * r.share_lent:.0f}% of the stablecoins supplied)",
        f"  expected bad debt a year         {usd(r.expected_bad_debt)}",
        "",
        "The BTC vault's allocation",
        *(f"  {name:<32} {pct(share)}" for name, share in zip(ALLOCATION, r.allocation, strict=True)),
        "",
        "Expected over a year",
        f"  interest share                   {usd(r.interest_income)}",
        f"  BTC vault fees                   {usd(r.btc_vault_fees)}",
        f"  operating costs                  {usd(r.operating_costs)}   "
        f"({100 * r.operating_ratio:.0f}% of revenue; usual range {100 * lo:.0f}-{100 * hi:.0f}%)",
        f"  of which payroll                 {usd(r.payroll)}   "
        f"({100 * r.payroll_ratio:.0f}% of revenue; usually at least {100 * PAYROLL_FLOOR:.0f}%)",
        f"  repaid to investors              {usd(r.compensation)}",
        "",
        "Investors, expected a year",
        f"  dollar vault                     {pct(r.dollar_yield)}   (elsewhere {100 * r.dollar_outside:.2f}%)",
        f"  BTC vault, in BTC                {pct(r.btc_yield)}   (elsewhere {100 * r.btc_outside:.2f}%)",
        f"  dollar vault, expected size      {usd(r.dollar_vault_next)}   (after one {period})",
        f"  BTC vault, expected size         {usd(r.btc_vault_next)}",
        "",
        f"How the next {period} may end",
        f"  expected equity                  {usd(r.expected_equity)}",
        f"  chance of ending with less       {pct(r.chance_of_loss)}",
        f"  chance of ending below needs     {pct(r.chance_below_needed)}   "
        f"(expected gap {usd(r.expected_shortfall).strip()})",
    ]
    return "\n".join(lines)
