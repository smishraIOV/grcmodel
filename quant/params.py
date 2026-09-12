"""Model parameters, as plain Python floats.

Separated from the tensor code on purpose. Parameters are scalars that belong
to the economics; tensors belong to a NumericsProfile (quant/numerics.py).
Keeping them apart is what lets a profile own *every* tensor construction,
which is how the silent-float32 bug is prevented rather than merely fixed.

Nothing here may hold a torch.Tensor, and nothing here may import torch.

Every value is illustrative, not calibrated — which is why quant/threshold.py
reports the effectiveness a programme must reach rather than a spend
recommendation (docs/quant-model.md section 8).
"""

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GrcAlphas:
    """Mitigation effectiveness per risk family: how much of the family's
    exposure one unit of GRC spend removes.

    These are the parameters nothing in this repo can currently calibrate,
    which is why quant/threshold.py reports break-even values for them rather
    than trusting a point estimate.
    """

    # Units of 1/money, so this moves with the loss scale. It was 0.3 when a
    # quarter's expected loss was 9 against equity of 16 -- a one-shot
    # magnitude. At the recalibrated scale (SamplerParams below) the same
    # effectiveness per unit of loss needs a larger alpha.
    #
    # 1.5 puts meaningful mitigation at a GRC stock near 1, which is where a
    # programme costing 5-10% of revenue settles given quarterly depreciation.
    credit: float = 1.5
    operational: float = 1.5
    compliance: float = 1.5


@dataclass(frozen=True)
class FirmParams:
    """Balance-sheet and technology parameters shared by every stage.

    These lived in quant/static.py, which meant the Monte Carlo stage imported
    them from the four-state model it was meant to be independent of.

    A caveat that matters now that there is a horizon: the loss and production
    parameters are *per period*, not annual rates. Only the discount and the
    GRC depreciation below are expressed annually and converted. Until the
    rest are rate-based, changing `periods_per_year` does not rescale the
    economics and the model cannot be checked for time-scale invariance --
    the same class of silent bug as the dimensional one in
    docs/static-model-debug-notes.md section 4, one axis over.
    """

    initial_equity: float = 16.0
    financing_convexity: float = 2.0
    distress_reference: float = 5.0
    # A and S in F(I) = A*S*(1 - exp(-I/S)). These were 3.0 and 10.0, which put
    # the per-quarter operating surplus at 9.01 against equity of 16 -- the firm
    # earned more than half its equity every quarter, which is not a bank, and
    # made its franchise worth roughly 473 against a book value of 16. That
    # single ratio is what left the wind-down option permanently out of the
    # money and the financing friction inert: nothing on the balance sheet
    # could matter next to an unconditional earnings stream that size.
    #
    # A just above 1 makes the margin thin; a large S keeps the marginal return
    # above 1 well past what the firm can fund, so the funding constraint keeps
    # biting instead of being outgrown. Deploying the ~23.5 it can fund, the
    # firm earns 0.70 a quarter on equity of 16 -- about 19% a year, which is
    # a high-margin intermediary rather than a miracle.
    production_scale: float = 1.05
    production_curvature: float = 600.0

    # GRC capital the firm already has, per family, at the start of a run.
    # Zero describes a firm building a control function from nothing, which is
    # a real case -- it is close to the pre-revenue variant of the maturity
    # fork in docs/quant-model.md section 5 -- but it is not the mature going
    # concern the current stages are about. Starting from zero left the firm
    # facing un-mitigated base hazards for several quarters while it built a
    # stock, which put the annual failure probability at 27.5%, outside the
    # band where survival comparative statics mean anything.
    #
    # The value is not tuned to taste: it is the self-consistent steady state,
    # the level at which the firm's own optimal maintenance spend exactly
    # replaces depreciation, so G_0 = g*(G_0) / delta. Found by bisection at
    # 5.18; a firm starting here neither builds nor runs down its control
    # function, which is what "mature going concern" should mean. It puts the
    # annual failure probability at 7.2%, inside the band, with a material
    # 1.07 per quarter of spend.
    #
    # Both neighbouring regimes are degenerate and it is worth knowing why.
    # Below about 3 the firm is rebuilding from a hole and dies from tail
    # events while it does. Above about 9 it has inherited so much capital that
    # optimal spend collapses to ~0.002 per quarter and the model has nothing
    # to say about budgets at all -- a "never binds" regime of the kind
    # docs/static-model-debug-notes.md section 6 warns about, in a new place.
    # Re-solved at the recalibrated loss and production scales: the fixed
    # point moved from 5.18 to 0.650 when a quarter's expected loss went from
    # 9.00 to 0.475. Same construction as before -- the level at which the
    # firm's own optimal maintenance spend exactly replaces depreciation.
    initial_grc_stock: float = 0.650

    # What creditors and shareholders recover when the firm fails, as a
    # fraction of whatever positive equity is left at that moment. Limited
    # liability floors it at zero: a firm that died owing money is worth
    # nothing to its owners, not a negative number.
    #
    # GRC does not appear here, deliberately, and that asymmetry is the point
    # of this channel. A programme can make failure rarer -- it acts on the
    # intensity (quant/hazard.py) -- but it cannot make a failure cheaper once
    # it happens. Losing a licence costs what it costs. Modelling GRC as
    # reducing both would let one parameter buy the same protection twice.
    #
    # 0.4 is illustrative but has anchors, unlike alpha: FDIC loss-given-
    # failure, and crypto bankruptcy recoveries in the 30-70c range.
    failure_recovery: float = 0.4

    # What an *orderly* wind-down recovers, as the same kind of fraction.
    # Higher than a disorderly failure because the firm chose the moment:
    # positions are unwound rather than liquidated into a panic, the licence is
    # surrendered rather than revoked, and counterparties are paid in sequence.
    #
    # This is where docs/framework.md section 1's Governance pillar -- "a named
    # authority who can halt activity" -- finally does work in the model.
    # Risk and Compliance both act by making bad outcomes rarer or smaller.
    # Governance acts by converting one kind of ending into another, which is a
    # different thing and had no representation here until now.
    orderly_recovery: float = 0.7

    # How much external finance the firm can raise, as a multiple of the
    # internal wealth it has left after the quarter's losses -- and whether the
    # market is open to it at all.
    #
    # This is the constraint that makes the balance sheet matter. Without it
    # the firm deploys its unconstrained optimum every quarter whatever its
    # capital, the franchise is worth ~473 against equity of 16, and both the
    # financing friction and the wind-down option sit inert. A bank cannot lend
    # more than it funds.
    #
    # It is also the Froot-Stein underinvestment channel in its hard form. A
    # bad draw leaves less internal wealth, less wealth means less funding,
    # less funding means forgone investment -- so risk management protects the
    # firm's capacity to invest, not merely its cash. The static model priced
    # that with a smooth convex premium; this states it as a constraint, which
    # is closer to how funding actually withdraws.
    #
    # Superseded by FundingParams below when the liability side is switched on,
    # where the same role is played by a deposit stock that persists, costs
    # interest and can run. Kept because it is what every pre-liability result
    # in docs/quant-model.md was produced under.
    external_funding_multiple: float = 0.5
    # Market access degrades as the firm weakens, rather than vanishing at a
    # threshold: a sigmoid in internal wealth over opening equity. Smooth
    # because a step would be one more place the gradient dies, and because
    # funding does in fact dry up gradually before it dries up suddenly.
    market_access_ratio: float = 0.15
    market_access_scale: float = 0.08

    periods_per_year: int = 4  # quarterly
    annual_discount_rate: float = 0.08
    # A control installed today still works in a year's time, but not forever:
    # staff turn over, systems age, regulations move. Illustrative, like
    # everything else here.
    annual_grc_depreciation: float = 0.25

    def discount(self) -> float:
        """Per-period discount factor."""
        return (1.0 + self.annual_discount_rate) ** (-1.0 / self.periods_per_year)

    def grc_depreciation(self) -> float:
        """Per-period fraction of the GRC stock that decays away."""
        return 1.0 - (1.0 - self.annual_grc_depreciation) ** (1.0 / self.periods_per_year)


@dataclass(frozen=True)
class ShockParams:
    """The four-state shock in quant/static.py.

    `credit_spread` widens the gap between the good and bad credit outcome
    while holding the mean at `credit_mean`, so sweeping it is a
    mean-preserving spread. Compliance failure is licence-to-operate risk
    (docs/framework.md section 1): rare, and severe enough to dwarf a bad
    credit year when it lands.
    """

    credit_mean: float = 4.0
    credit_spread: float = 2.0
    op_severity: float = 7.0
    compliance_probability: float = 0.05
    compliance_severity: float = 30.0


@dataclass(frozen=True)
class SamplerParams:
    """Monte Carlo distribution parameters (quant/simulate.py).

    Chosen to match the expected loss of the four-state shock above
    (4.0 credit + 3.5 operational + 1.5 compliance) so the two stages are
    comparable. Credit is a continuous loss; operational incidents are rare
    but heavy-tailed when they land; a compliance breach is rarer still and
    carries a fixed severity, because losing a licence costs what it costs.
    """

    # Scaled down roughly twentyfold from the static model's magnitudes, which
    # were a single shock absorbed once rather than a rate. A gross expected
    # loss of 9.00 a quarter against equity of 16 kills the firm inside two
    # quarters; at 0.475 it is about two thirds of the operating surplus before
    # any GRC, which is a risky business rather than a doomed one.
    #
    # The 4 : 3.5 : 1.5 weighting across families is preserved, so every
    # exposure-relative result carries over.
    credit_loss_mean: float = 0.20
    op_probability: float = 0.25
    op_severity_mean: float = 0.70
    compliance_probability: float = 0.05
    compliance_severity: float = 2.0


@dataclass(frozen=True)
class FundingParams:
    """The liability side: what funds the book, what it costs, and what it does
    when the firm weakens.

    Until this existed the firm had no liabilities at all. It funded its book
    out of equity plus a multiple of equity that cost nothing, was repaid
    inside the period, and could not leave. That is not a bank, and it deleted
    three of the risks a bank exists to manage: leverage, the cost of funding,
    and the possibility that the funding walks out. A hazard channel named
    "depositors leave" was a label on a constant, because there were no
    depositors in the model to leave.

    What this adds is a **stock**, not a flow. Deposits outstanding at the end
    of one period are outstanding at the start of the next, which is the whole
    difference: a liability that is repaid within the period cannot run, and a
    run is the event the channel is named after.

    The economics it buys, in the order they bite:

    - **Leverage amplifies asset losses onto equity.** A 1% loss on a book
      funded five-to-one against capital is a 5% loss of capital. That is the
      textbook mechanism by which ordinary credit losses -- not exotic ones --
      kill intermediaries, and the model could not express it.
    - **Funding is not free.** Deposits pay interest whether or not the book
      earns, so leverage is a decision with a price rather than something to
      max out.
    - **Funding withdraws as the firm weakens.** The deposit base the firm can
      carry shrinks with its capital ratio, so a bad quarter cuts next
      quarter's lending capacity. This is the Froot-Stein underinvestment
      channel arriving through the liability side rather than being asserted
      as a constraint on the asset side.

    What is *not* here yet, and is the next stage: withdrawals as a shock, and
    the fire-sale cost of meeting them out of a book that has not matured.
    Deposits currently adjust smoothly toward capacity, which is a run in slow
    motion -- enough to make leverage a real decision, not enough to make
    liquidity one. The reserve the firm holds against withdrawals is already
    here and already priced; what is missing is anything to hold it against.
    """

    # Deposits the firm can carry per unit of equity, at full market access.
    # 4.0 is a five-to-one balance sheet: assets = (1 + 4) x equity. Banks run
    # between 10x and 20x on risk-weighted measures and nearer 10x on a plain
    # leverage ratio; a crypto intermediary funding itself on demandable
    # deposits would not be granted that, and the point of the number is to put
    # the firm somewhere leverage plainly matters rather than to match a
    # jurisdiction.
    deposit_capacity: float = 4.0

    # What the deposits cost, per year. Below the asset yield, which is the
    # entire reason an intermediary exists -- it is paid for maturity and
    # credit transformation, and the spread is where that payment shows up.
    annual_deposit_rate: float = 0.02

    # What funding the firm has *not* lent earns while it sits there.
    #
    # Without this, deposits the firm cannot profitably deploy are a pure
    # deadweight loss, and the model punishes a bank for having a franchise:
    # measured at 2.5x leverage on the pre-liability opportunity, firm value
    # fell 5% purely because the firm was made to carry funding it had no use
    # for. Real intermediaries park surplus deposits in something liquid; they
    # do not set fire to them.
    #
    # Below the deposit rate, so idle funding carries a small negative spread
    # rather than being free. That half a point is doing real work later: it is
    # what stops a liquidity buffer being free insurance once withdrawals
    # exist, and a buffer that costs nothing would tell us nothing about how
    # big one should be.
    annual_reserve_rate: float = 0.015

    # How fast the deposit base moves toward the capacity the firm's capital
    # supports, per year. Not instantaneous, because a stock that re-solves
    # itself every period is a flow again and cannot run.
    #
    # 2.0 a year is a half-life of about four months: a firm that loses capital
    # sheds the corresponding deposits over two or three quarters rather than
    # on the day. Faster than that and the liability side stops being a state
    # variable in any meaningful sense; much slower and a firm that has lost
    # its capital keeps funding a book it can no longer support, which is the
    # opposite of the mechanism.
    annual_adjustment_speed: float = 2.0

    # Market access, as it applies to the deposit base rather than to a
    # within-period borrowing. Same sigmoid and the same reasoning as
    # FirmParams.market_access_*: funding dries up gradually before it dries up
    # suddenly, and a step would be one more place the gradient dies.
    #
    # Applied to the *capacity*, so a weakened firm is not merely constrained
    # in what it may deploy -- its existing funding is leaving.
    access_ratio: float = 0.15
    access_scale: float = 0.08

    def period_deposit_rate(self, periods_per_year: int) -> float:
        """Interest per period. A rate that compounds down, not one divided
        down: at four periods a year the difference is immaterial, at
        fifty-two it is not, and every other annual figure here converts the
        same way."""
        return (1.0 + self.annual_deposit_rate) ** (1.0 / periods_per_year) - 1.0

    def period_reserve_rate(self, periods_per_year: int) -> float:
        """What idle funding earns per period. Compounds down like the rest."""
        return (1.0 + self.annual_reserve_rate) ** (1.0 / periods_per_year) - 1.0

    def period_adjustment(self, periods_per_year: int) -> float:
        """Fraction of the gap to capacity closed in one period.

        A Poisson-style conversion, 1 - exp(-theta/ppy), for the same reason
        the cliff rate uses one: it is a rate, it must land inside [0, 1) for
        any rate at all, and the obvious 1 - (1 - theta)**(1/ppy) returns a
        complex number for theta > 1 -- which this default is.
        """
        return 1.0 - math.exp(-self.annual_adjustment_speed / periods_per_year)


@dataclass(frozen=True)
class HazardParams:
    """Intensity of firm failure, as a function of how well capitalized it is.

    A logistic in the capital ratio rather than a power law. Every argument to
    the exponential must be dimensionless, and a ratio over a dimensionless
    scale is that by construction; a power of a money quantity is how the
    financing cost acquired its units bug (docs/static-model-debug-notes.md
    section 4).

    The base rate is the intensity a maximally distressed firm faces, not the
    one a healthy firm faces. At `capital_target` the hazard is half of it.
    Illustrative, like everything else here -- but unlike alpha, these have
    external anchors that could be used: bank failure rates by capital ratio,
    rating-agency default rates, observed crypto-lender failures.
    """

    annual_base_rate: float = 0.5
    capital_target: float = 0.4   # ratio of opening equity at which hazard equals its base
    capital_scale: float = 0.2    # how sharply hazard responds around that point

    # Two further channels, each reducible by the GRC stock of its family.
    # Competing risks compose additively in intensity, so these sum with the
    # capital channel above rather than multiplying.
    #
    # Operational: an incident becomes public and depositors leave. This is how
    # a crypto intermediary actually fails -- a bridge exploit or a custody
    # failure, not an accounting threshold crossed on a particular Tuesday.
    # Licence: a compliance breach escalates to revocation, which ends the firm
    # regardless of its balance sheet.
    #
    # Rates below are at ZERO GRC stock, so they are what the firm faces with no
    # programme at all, not what it faces in practice.
    annual_operational_rate: float = 0.15
    annual_licence_rate: float = 0.08


@dataclass(frozen=True)
class CliffParams:
    """A severe but survivable hit: the state worth most avoiding.

    A moderate loss and outright failure both leave management with few real
    choices -- absorb it, or it is over. The cliff is the case in between: the
    firm is badly impaired and still alive, holding a decision about whether to
    rebuild, run down, or wind up, with a hazard that has risen sharply and a
    balance sheet that no longer funds its opportunity. That is where the
    decision problem actually lives, and a model without it has no state in
    which GRC's option value is doing anything.

    GRC reduces the **frequency** and not the severity. A bridge is either
    drained or it is not; controls make the exploit less likely, they do not
    make it smaller. That is the opposite of the operational loss channel,
    where controls contain an incident that happens anyway, and the two are
    modelled separately because they are different claims.

    Severity is drawn as a fraction of opening equity, so it scales with the
    firm and stays invariant to the currency it is denominated in.
    """

    period_probability: float = 0.025      # at zero operational GRC stock
    mean_severity_fraction: float = 0.35   # of opening equity, exponential, capped at 1
    # Temperature of the straight-through relaxation used to differentiate the
    # occurrence probability (docs/static-model-debug-notes.md section 7).
    #
    # Chosen by measuring the gradient against the exact analytic mixture,
    # which is affordable at one period. Bias in dE[loss]/dG, large-sample:
    #
    #     tau     1.00   0.50   0.25   0.12   0.06   0.03
    #     ratio   3.40   1.48   1.11   1.03   1.02   1.01
    #
    # and the variance runs the other way -- relative standard deviation of the
    # gradient across scenario draws at 512 paths is 0.24 at tau = 0.5 and 1.41
    # at 0.03. 0.1 sits where the bias has flattened out and the variance has
    # not yet taken over. The sign was never wrong at any temperature tested,
    # which is the property that actually matters for a descent direction.
    relaxation_temperature: float = 0.1


@dataclass(frozen=True)
class ModelParams:
    """Everything the static and Monte Carlo stages need, in one place."""

    firm: FirmParams = field(default_factory=FirmParams)
    alphas: GrcAlphas = field(default_factory=GrcAlphas)
    shock: ShockParams = field(default_factory=ShockParams)
    sampler: SamplerParams = field(default_factory=SamplerParams)
    hazard: HazardParams = field(default_factory=HazardParams)
    cliff: CliffParams = field(default_factory=CliffParams)
    funding: FundingParams = field(default_factory=FundingParams)


@dataclass(frozen=True)
class AnnualRates:
    """The economics, stated without reference to how often the firm decides.

    Everything here is either a rate per year or a per-event magnitude, so it
    means the same thing whatever the decision frequency is. `model_at` turns
    it into a per-period parameter set.

    This exists because the model silently assumed its decision frequency. The
    discount, the GRC depreciation and all three hazard rates were already
    annual and divide down correctly; the loss means, the event probabilities,
    the cliff rate and the production parameters were per-period and did not.
    Switching from quarterly to weekly decisions without this would have handed
    the firm thirteen times its annual losses and thirteen times its annual
    return -- the dimensional bug of section 4 of the debug notes, relocated
    to the time axis and just as silent.
    """

    # Losses. A *rate* per year for how often, a magnitude per event for how
    # bad -- severities do not scale with frequency, because an incident is the
    # size it is however often you look.
    # Credit loss is a RATE on the book -- a fraction of what is deployed,
    # lost per year -- not a money amount. 3.4% against the ~23.5 the firm
    # funds reproduces the 0.80 a year it used to lose at that book size, so
    # the level is unchanged and only its dependence on the book is new.
    credit_loss_rate: float = 0.034    # fraction of deployed book lost per year
    op_events: float = 1.0             # operational incidents per year
    op_severity: float = 0.70          # per incident
    compliance_events: float = 0.20    # breaches per year
    compliance_severity: float = 2.0   # per breach
    cliff_events: float = 0.101272     # cliff strikes per year at zero GRC

    # Production. The gross return is annual and compounds down; the curvature
    # scales *up* with frequency so that the unconstrained optimum
    # I* = S ln(A) is the same amount of capital however often it is re-decided.
    annual_return: float = 1.2155      # = 1.05 ** 4, the old quarterly figure
    # Sets how much capital the firm wants to deploy: I* = S ln A, with
    # S = curvature_per_period x periods_per_year.
    #
    # 150 put I* at 29.3 against a balance sheet of 24, which was right for a
    # firm funding itself out of equity plus a small costless multiple. Once
    # the liability side arrived the balance sheet grew to about 80 and the
    # firm's *opportunity* did not, so it stopped wanting what it could fund:
    # measured, the share of quarters in which funding bound fell from 71% to
    # 0.0%, which is the Froot-Stein underinvestment channel switching itself
    # off. A bank that cannot use its deposits is not levered, it is merely
    # paying for storage.
    #
    # 500 puts I* at 97.6 against a funding capacity near 80 -- the same 1.2:1
    # ratio of wanted to fundable that the pre-liability calibration ran at, so
    # the channel bites as hard as it did and for the same reason. Chosen by
    # sweeping it: at 300 the firm is comfortable, at 750 it is permanently
    # capital-starved and the comparative statics flatten out.
    curvature_per_period: float = 500.0


ANNUAL = AnnualRates()


def model_at(periods_per_year: int, rates: AnnualRates = ANNUAL) -> ModelParams:
    """A consistent parameter set for a given decision frequency."""
    return ModelParams(
        firm=FirmParams(
            periods_per_year=periods_per_year,
            production_scale=rates.annual_return ** (1.0 / periods_per_year),
            production_curvature=rates.curvature_per_period * periods_per_year,
        ),
        sampler=SamplerParams(
            # A per-period loss rate on the book, not a money amount.
            credit_loss_mean=rates.credit_loss_rate / periods_per_year,
            op_probability=rates.op_events / periods_per_year,
            op_severity_mean=rates.op_severity,
            compliance_probability=rates.compliance_events / periods_per_year,
            compliance_severity=rates.compliance_severity,
        ),
        cliff=CliffParams(
            # A Poisson rate, so the per-period probability of at least one
            # strike is 1 - exp(-lambda/ppy). The obvious-looking
            # 1 - (1-lambda)**(1/ppy) treats the rate as a probability: it
            # agrees to three decimals for small lambda, and for lambda > 1 it
            # raises a negative number to a fractional power and returns a
            # complex number. Found by sweeping the rate up to 1.5.
            period_probability=1.0 - math.exp(-rates.cliff_events / periods_per_year),
        ),
    )


DEFAULTS = model_at(4)    # quarterly decisions
WEEKLY = model_at(52)     # weekly decisions

# Pinned, and deliberately not shared with DEFAULTS.
#
# The four-state world and the closed-form benchmark exist to be *exactly
# solvable*, not to be plausible. They are the only thing in this project that
# knows a right answer without solving anything, so every approximate solver is
# ultimately measured against them.
#
# Sharing parameters with the dynamic model meant that recalibrating the
# dynamic model for plausibility silently moved the oracle -- which is how a
# single change to production_scale took out five regression tests and both
# closed-form checks at once. An oracle that moves when the thing it is
# checking moves is not an oracle.
#
# These are the static model's original values and they should stay put. Their
# job is to keep ln(alpha * X) / alpha away from its degenerate corners, not to
# describe a bank.
ORACLE = ModelParams(
    firm=FirmParams(production_scale=3.0, production_curvature=10.0),
    alphas=GrcAlphas(credit=0.3, operational=0.3, compliance=0.3),
    shock=ShockParams(),
)
SPREAD_SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0]
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
