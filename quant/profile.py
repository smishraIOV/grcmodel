"""A firm's profile in real units, and its mapping onto the model
(design.md section 4.1; docs/plan-tool.md step 1).

A profile is what a CFO or risk manager enters: money in **dollars**, rates
**per year**. It has three parts:

- `FirmFacts` -- what the firm knows about itself: its vaults, its equity,
  its costs, its fees and terms;
- `MarketAssumptions` -- BTC, rates, loan demand, what investors do;
- `RiskAssumptions` -- incidents, operating losses, and how well GRC works.
  These cannot be observed, and are meant to be varied.

`to_model` maps a profile onto `Params`, the firm's state and the **scale**:
the dollar vault's capacity, in dollars, is one model unit. `to_profile` maps
back. `illustrative` fills the two sets of assumptions from the model's own
defaults (`quant.params.DEFAULTS`), so there is one source for them: they are
illustrative, not calibrated.

Three inputs are asked as statements a user can answer, and converted:

- **The cost of raising equity** (`RaiseCost`): the all-in cost of raising
  two amounts. The model's C(e) = lam e + kappa/2 e^2 costs lam + kappa/2 e
  per dollar raised, a straight line through the two answers.
- **What GRC does** (`GrcEffect`): "spending X a year, kept up, removes a
  share Y of this risk; no spend removes more than Z". The model's
  phi(S) = rho + (1 - rho) exp(-a S) then has rho = 1 - Z and
  a = -ln(1 - Y / Z) / S, at the stock S that spend settles at.
- **GRC today**: the yearly GRC budget, assumed steady, so the stock is where
  S = (1 - delta) S + g settles: the spend a period over the wear-off a period.
"""

import math
from dataclasses import dataclass, fields, replace

from quant.params import DEFAULTS, Params

STRATEGIES = ("lending BTC", "basis trading", "cold custody")

# Parameters a profile does not set. They keep the values of the `base`
# parameters passed to `to_model`: tail shapes, bounds and numerical guards.
LEFT_AT_DEFAULT = frozenset(
    {"nu", "jump_tail", "btc_crash_tail", "q_max", "c_H", "tau", "liq_band", "price_floor"}
)


@dataclass(frozen=True)
class RaiseCost:
    """Raising `amount` dollars of new equity costs `cost` of it, all in."""

    amount: float
    cost: float


@dataclass(frozen=True)
class LoanPoint:
    """`loans` dollars can be placed at the borrower `rate` (a year), at the
    reference loan-to-value."""

    loans: float
    rate: float


@dataclass(frozen=True)
class GrcEffect:
    """Spending `spend` dollars a year, kept up year after year, removes the
    share `reduction` of a risk; no amount of spend removes more than `limit`."""

    spend: float
    reduction: float
    limit: float


@dataclass(frozen=True, kw_only=True)
class FirmFacts:
    """What the firm knows about itself. Dollars, and rates per year."""

    dollar_vault: float              # its size now
    dollar_vault_capacity: float     # its full size; fixes the scale
    btc_vault: float                 # its size now, valued in dollars
    btc_vault_capacity: float
    equity: float                    # the firm's own equity now
    grc_budget: float                # GRC spend a year, taken as steady up to now
    operating_costs: float           # a year, fixed in dollars
    payroll_share: float             # of operating costs
    runway_months: float             # operating costs kept in stablecoins
    interest_share: float            # the firm's share of the interest borrowers pay
    management_fee: float            # BTC vault, share of its BTC a year
    performance_fee: float           # BTC vault, share of positive BTC yield
    dollar_vault_stake: float        # the firm's own money in each vault, share of its size
    btc_vault_stake: float
    compensation: float              # share of investors' incident losses and bad debt it repays
    own_funds: float                 # required per dollar of the vaults
    liquidation_threshold: float     # loan to collateral value at which liquidation starts
    liquidation_penalty: float
    ltv_min: float                   # the loan-to-value range it may offer
    ltv_max: float
    btc_security_cost: float         # securing its own BTC, share a year
    required_return: float           # shareholders', a year
    small_raise: RaiseCost           # the cost of raising equity, at two sizes
    large_raise: RaiseCost
    decisions_per_year: int = 4


@dataclass(frozen=True, kw_only=True)
class MarketAssumptions:
    """BTC, rates, loan demand and investors. Rates per year."""

    riskless_rate: float
    perp_premium: float              # perp funding above the riskless rate
    raise_cost_after_btc_halves: float   # how many times dearer raising equity is after BTC halves
    raise_cost_after_incident: float     # and after an incident
    btc_volatility: float            # of its smooth move
    btc_crashes_per_year: float
    btc_crash_size: float            # mean share of the price a crash takes
    btc_view: float                  # the firm's expected BTC return above the market's
    loan_demand: LoanPoint
    rate_no_demand: float            # borrower rate at which nobody borrows
    reference_ltv: float             # the loan-to-value `loan_demand` is stated at
    demand_per_10_ltv_points: float  # extra demand, as a share, per 10 points of loan-to-value
    dollar_outside_rate: float       # what investors could earn elsewhere
    btc_outside_yield: float
    dollar_flow_response: float      # net flows, share of the vault, per unit of yield gap
    btc_flow_response: float
    inflow_speed: float              # share of a vault's gap to capacity closed a year
    strategy_yield: tuple[float, float, float]         # BTC yield a year, in the order of STRATEGIES
    strategy_volatility: tuple[float, float, float]
    strategy_crash_loss: tuple[float, float, float]    # loss per unit of a BTC crash's size
    strategy_correlation: float      # of the two active strategies' yields


@dataclass(frozen=True, kw_only=True)
class RiskAssumptions:
    """Incidents, operating losses, and what GRC does about them. Not
    observable: vary them, and see what depends on them."""

    incidents_per_year: float        # per vault, with no GRC
    incident_loss: float             # mean share of the exposed assets lost
    lending_loss: float              # mean operating loss, share of the lending system a year
    lending_loss_volatility: float
    strategy_loss: tuple[float, float, float]              # the same for each BTC strategy
    strategy_incident_exposure: tuple[float, float, float]
    sale_discount: float             # how far below market liquidated collateral sells, with no GRC
    liquidation_slippage: float      # share of a smooth fall past the threshold lost before selling, with no GRC
    outflow_after_incident: float    # share of a vault that leaves
    outflow_from_other_vault: float  # share of the other vault that would leave if the vault hit were the whole business
    grc_wear_off: float              # share of GRC's effect lost a year
    grc_on_losses: GrcEffect         # on average operating losses (A)
    grc_on_spread: GrcEffect         # on how much they vary (B)
    grc_on_incidents: GrcEffect      # on how often incidents happen (J)
    grc_on_sale_discount: GrcEffect  # on the sale discount and the slippage (L)


@dataclass(frozen=True)
class FirmProfile:
    facts: FirmFacts
    market: MarketAssumptions
    risk: RiskAssumptions


@dataclass(frozen=True)
class FirmState:
    """Where the firm is now, in model units (design.md section 3.5)."""

    w: float      # equity
    S: float      # GRC stock
    A_L: float    # the dollar vault's size
    A_B: float    # the BTC vault's size


@dataclass(frozen=True)
class ModelFirm:
    """A profile as the model sees it. `params` has rates per year;
    `scale` is dollars per model unit."""

    params: Params
    state: FirmState
    scale: float

    def dollars(self, units: float) -> float:
        return units * self.scale

    def units(self, dollars: float) -> float:
        return dollars / self.scale


# -- conversions ----------------------------------------------------------------


def _per_period(rate: float, n: int) -> float:
    return (1.0 + rate) ** (1.0 / n) - 1.0


def steady_stock(spend_per_year: float, wear_off: float, n: int) -> float:
    """The GRC stock a steady yearly spend settles at, with n decisions a
    year: S = (1 - d) S + g at g = spend / n and d the wear-off a period."""
    return (spend_per_year / n) / (1.0 - (1.0 - wear_off) ** (1.0 / n))


def _demand_share(rate: float, rate_no_demand: float, n: int) -> float:
    """Demand at `rate` as a share of demand at a zero rate. The line is
    straight in the rate of one period (quant.lending.loan_demand)."""
    return 1.0 - _per_period(rate, n) / _per_period(rate_no_demand, n)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_share(value: float, name: str) -> None:
    _check(0.0 <= value <= 1.0, f"{name} must be between 0 and 1 (it is {value})")


def validate(profile: FirmProfile) -> None:
    """Refuse a profile the model cannot represent, saying why."""
    f, m, k = profile.facts, profile.market, profile.risk
    _check(f.dollar_vault_capacity > 0, "the dollar vault's capacity must be positive: it sets the scale")
    _check(f.btc_vault_capacity >= 0, "the BTC vault's capacity cannot be negative")
    _check(f.dollar_vault >= 0 and f.btc_vault >= 0, "a vault's size cannot be negative")
    _check(f.operating_costs >= 0 and f.grc_budget >= 0, "operating costs and the GRC budget cannot be negative")
    _check(f.runway_months >= 0, "the runway cannot be negative")
    _check(isinstance(f.decisions_per_year, int) and f.decisions_per_year >= 1,
           "decisions per year must be a whole number, at least 1")
    for name in ("payroll_share", "interest_share", "performance_fee", "dollar_vault_stake",
                 "btc_vault_stake", "compensation", "own_funds", "liquidation_penalty"):
        _check_share(getattr(f, name), name.replace("_", " "))
    _check(0 < f.small_raise.amount < f.large_raise.amount,
           "the two equity raises must be of different, positive sizes, the small one first")
    _check(f.large_raise.cost >= f.small_raise.cost,
           "the large equity raise cannot cost less per dollar than the small one")
    _check(m.raise_cost_after_btc_halves >= 1 and m.raise_cost_after_incident >= 1,
           "a bad period cannot make raising equity cheaper: the two multiples must be at least 1")
    _check(m.loan_demand.loans >= 0, "loan demand cannot be negative")
    _check(0 <= m.loan_demand.rate < m.rate_no_demand,
           "loan demand must be stated at a rate below the one at which nobody borrows")
    for effect, what in ((k.grc_on_losses, "losses"), (k.grc_on_spread, "spread"),
                         (k.grc_on_incidents, "incidents"), (k.grc_on_sale_discount, "sale discount")):
        _check(effect.spend > 0, f"GRC on {what}: the spend must be positive")
        _check(0 <= effect.reduction < effect.limit <= 1,
               f"GRC on {what}: the reduction must be below the limit, and the limit at most 1")
    _check(0 < k.grc_wear_off <= 1, "GRC's wear-off must be above 0 and at most 1 a year")


def _changes(profile: FirmProfile) -> dict:
    """Every parameter a profile sets, in model units, rates per year."""
    f, m, k = profile.facts, profile.market, profile.risk
    scale, n = f.dollar_vault_capacity, f.decisions_per_year

    slope = (f.large_raise.cost - f.small_raise.cost) / ((f.large_raise.amount - f.small_raise.amount) / scale)
    lam = f.small_raise.cost - slope * f.small_raise.amount / scale
    _check(lam >= 0, "the two equity raises imply that a very small raise costs less than nothing")

    def effect(e: GrcEffect) -> tuple[float, float]:
        stock = steady_stock(e.spend / scale, k.grc_wear_off, n)
        return -math.log(1.0 - e.reduction / e.limit) / stock, 1.0 - e.limit

    (a_A, rho_A), (a_B, rho_B) = effect(k.grc_on_losses), effect(k.grc_on_spread)
    (a_J, rho_J), (a_L, rho_L) = effect(k.grc_on_incidents), effect(k.grc_on_sale_discount)
    at_zero = (m.loan_demand.loans / scale) / _demand_share(m.loan_demand.rate, m.rate_no_demand, n)
    return dict(
        periods_per_year=n,
        # the firm's facts
        lend_capacity=1.0, btc_capacity=f.btc_vault_capacity / scale,
        opex=f.operating_costs / scale, payroll_share=f.payroll_share, runway=f.runway_months / 12.0,
        interest_share=f.interest_share, fee_mgmt=f.management_fee, fee_perf=f.performance_fee,
        lend_co_invest=f.dollar_vault_stake, btc_co_invest=f.btc_vault_stake,
        compensation=f.compensation, k_A=f.own_funds,
        liq_threshold=f.liquidation_threshold, liq_penalty=f.liquidation_penalty,
        ltv_min=f.ltv_min, ltv_max=f.ltv_max, btc_holding_cost=f.btc_security_cost,
        r_E=f.required_return, lam=lam, kappa=2.0 * slope,
        # the market
        r=m.riskless_rate, perp_premium=m.perp_premium, s_BTC=m.btc_volatility,
        issue_btc_fall=(m.raise_cost_after_btc_halves - 1.0) / 0.5,
        issue_incident=m.raise_cost_after_incident - 1.0,
        btc_crash_rate=m.btc_crashes_per_year, btc_crash_mean=m.btc_crash_size, btc_bias=m.btc_view,
        loan_demand=at_zero, rate_max=m.rate_no_demand, ltv_ref=m.reference_ltv,
        ltv_demand=m.demand_per_10_ltv_points / 0.10,
        outside_rate=m.dollar_outside_rate, btc_outside_yield=m.btc_outside_yield,
        flow_yield=m.dollar_flow_response, btc_flow_yield=m.btc_flow_response,
        inflow_speed=m.inflow_speed, strategy_yield=tuple(m.strategy_yield),
        strategy_vol=tuple(m.strategy_volatility), strategy_crash=tuple(m.strategy_crash_loss),
        strategy_rho=m.strategy_correlation,
        # risks and GRC
        jump_rate=k.incidents_per_year, jump_mean=k.incident_loss,
        lend_loss=k.lending_loss, lend_vol=k.lending_loss_volatility,
        strategy_loss=tuple(k.strategy_loss), strategy_incident=tuple(k.strategy_incident_exposure),
        sale_discount=k.sale_discount, liq_slippage=k.liquidation_slippage,
        outflow_incident=k.outflow_after_incident, outflow_contagion=k.outflow_from_other_vault,
        delta=k.grc_wear_off,
        a_A=a_A, rho_A=rho_A, a_B=a_B, rho_B=rho_B, a_J=a_J, rho_J=rho_J, a_L=a_L, rho_L=rho_L,
    )


def to_model(profile: FirmProfile, base: Params = DEFAULTS) -> ModelFirm:
    """The profile as parameters, a state and a scale. Parameters a profile
    does not set (`LEFT_AT_DEFAULT`) are taken from `base`."""
    validate(profile)
    f = profile.facts
    scale = f.dollar_vault_capacity
    params = replace(base, **_changes(profile))
    state = FirmState(
        w=f.equity / scale,
        S=steady_stock(f.grc_budget / scale, params.delta, f.decisions_per_year),
        A_L=f.dollar_vault / scale,
        A_B=f.btc_vault / scale,
    )
    return ModelFirm(params=params, state=state, scale=scale)


# -- and back ---------------------------------------------------------------------


def _facts_from(p: Params, state: FirmState, scale: float, raises: tuple[float, float]) -> FirmFacts:
    n = p.periods_per_year

    def raise_cost(amount: float) -> RaiseCost:
        return RaiseCost(amount=amount, cost=p.lam + 0.5 * p.kappa * amount / scale)

    return FirmFacts(
        dollar_vault=state.A_L * scale, dollar_vault_capacity=p.lend_capacity * scale,
        btc_vault=state.A_B * scale, btc_vault_capacity=p.btc_capacity * scale,
        equity=state.w * scale,
        grc_budget=state.S * (1.0 - (1.0 - p.delta) ** (1.0 / n)) * n * scale,
        operating_costs=p.opex * scale, payroll_share=p.payroll_share, runway_months=p.runway * 12.0,
        interest_share=p.interest_share, management_fee=p.fee_mgmt, performance_fee=p.fee_perf,
        dollar_vault_stake=p.lend_co_invest, btc_vault_stake=p.btc_co_invest,
        compensation=p.compensation, own_funds=p.k_A,
        liquidation_threshold=p.liq_threshold, liquidation_penalty=p.liq_penalty,
        ltv_min=p.ltv_min, ltv_max=p.ltv_max, btc_security_cost=p.btc_holding_cost,
        required_return=p.r_E, small_raise=raise_cost(raises[0]), large_raise=raise_cost(raises[1]),
        decisions_per_year=n,
    )


def _market_from(p: Params, scale: float, loan_rate: float) -> MarketAssumptions:
    n = p.periods_per_year
    return MarketAssumptions(
        riskless_rate=p.r, perp_premium=p.perp_premium, btc_volatility=p.s_BTC,
        raise_cost_after_btc_halves=1.0 + 0.5 * p.issue_btc_fall,
        raise_cost_after_incident=1.0 + p.issue_incident,
        btc_crashes_per_year=p.btc_crash_rate, btc_crash_size=p.btc_crash_mean, btc_view=p.btc_bias,
        loan_demand=LoanPoint(
            loans=p.loan_demand * _demand_share(loan_rate, p.rate_max, n) * scale, rate=loan_rate
        ),
        rate_no_demand=p.rate_max, reference_ltv=p.ltv_ref,
        demand_per_10_ltv_points=p.ltv_demand * 0.10,
        dollar_outside_rate=p.outside_rate, btc_outside_yield=p.btc_outside_yield,
        dollar_flow_response=p.flow_yield, btc_flow_response=p.btc_flow_yield,
        inflow_speed=p.inflow_speed, strategy_yield=tuple(p.strategy_yield),
        strategy_volatility=tuple(p.strategy_vol), strategy_crash_loss=tuple(p.strategy_crash),
        strategy_correlation=p.strategy_rho,
    )


def _risk_from(p: Params, scale: float, spends: tuple[float, float, float, float]) -> RiskAssumptions:
    n = p.periods_per_year

    def effect(spend: float, a: float, rho: float) -> GrcEffect:
        stock = steady_stock(spend / scale, p.delta, n)
        return GrcEffect(spend=spend, reduction=(1.0 - rho) * (1.0 - math.exp(-a * stock)), limit=1.0 - rho)

    return RiskAssumptions(
        incidents_per_year=p.jump_rate, incident_loss=p.jump_mean,
        lending_loss=p.lend_loss, lending_loss_volatility=p.lend_vol,
        strategy_loss=tuple(p.strategy_loss), strategy_incident_exposure=tuple(p.strategy_incident),
        sale_discount=p.sale_discount, liquidation_slippage=p.liq_slippage,
        outflow_after_incident=p.outflow_incident, outflow_from_other_vault=p.outflow_contagion,
        grc_wear_off=p.delta,
        grc_on_losses=effect(spends[0], p.a_A, p.rho_A),
        grc_on_spread=effect(spends[1], p.a_B, p.rho_B),
        grc_on_incidents=effect(spends[2], p.a_J, p.rho_J),
        grc_on_sale_discount=effect(spends[3], p.a_L, p.rho_L),
    )


def to_profile(model: ModelFirm, like: FirmProfile) -> FirmProfile:
    """The model's firm as a profile. The statements are made at the same
    points as in `like`: its two raise amounts, the rate its loan demand is
    stated at, and its four GRC spends. `to_profile(to_model(x), like=x)` is x."""
    p, scale = model.params, model.scale
    k = like.risk
    return FirmProfile(
        facts=_facts_from(p, model.state, scale, (like.facts.small_raise.amount, like.facts.large_raise.amount)),
        market=_market_from(p, scale, like.market.loan_demand.rate),
        risk=_risk_from(p, scale, (k.grc_on_losses.spend, k.grc_on_spread.spend,
                                   k.grc_on_incidents.spend, k.grc_on_sale_discount.spend)),
    )


# -- illustrative assumptions --------------------------------------------------------

ILLUSTRATIVE_LOAN_RATE = 0.10       # the rate the illustrative loan demand is stated at
ILLUSTRATIVE_GRC_SPEND = 0.005      # and the GRC spend, as a share of the dollar vault's capacity


def illustrative(facts: FirmFacts, base: Params = DEFAULTS) -> FirmProfile:
    """A profile with these facts and the model's default assumptions, sized
    to the firm. **Illustrative, not calibrated**: the defaults are those of
    `quant.params`, restated as a user would enter them."""
    scale = facts.dollar_vault_capacity
    p = replace(base, periods_per_year=facts.decisions_per_year)
    spend = ILLUSTRATIVE_GRC_SPEND * scale
    return FirmProfile(
        facts=facts,
        market=_market_from(p, scale, ILLUSTRATIVE_LOAN_RATE),
        risk=_risk_from(p, scale, (spend, spend, spend, spend)),
    )


def example_facts() -> FirmFacts:
    """A made-up firm, for examples and tests: a $100m dollar vault and a
    $200m BTC vault, with the model's default terms and costs of raising
    equity (very expensive: 20% all in to raise $5m, 50% to raise $20m)."""
    return FirmFacts(
        dollar_vault=80e6, dollar_vault_capacity=100e6, btc_vault=150e6, btc_vault_capacity=200e6,
        equity=20e6, grc_budget=0.5e6, operating_costs=4e6, payroll_share=0.60, runway_months=12.0,
        interest_share=0.40, management_fee=0.01, performance_fee=0.15,
        dollar_vault_stake=0.05, btc_vault_stake=0.05, compensation=0.5, own_funds=0.02,
        liquidation_threshold=0.80, liquidation_penalty=0.05, ltv_min=0.30, ltv_max=0.75,
        btc_security_cost=0.015, required_return=0.06,
        small_raise=RaiseCost(amount=5e6, cost=0.20), large_raise=RaiseCost(amount=20e6, cost=0.50),
        decisions_per_year=4,
    )


def profile_fields() -> dict[str, tuple[str, ...]]:
    """The names of a profile's inputs, by part."""
    return {
        "facts": tuple(x.name for x in fields(FirmFacts)),
        "market": tuple(x.name for x in fields(MarketAssumptions)),
        "risk": tuple(x.name for x in fields(RiskAssumptions)),
    }
