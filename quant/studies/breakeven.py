"""Break-evens: what the model says when it refuses to quote a number.

Nothing in this repo can calibrate alpha -- how much of a risk family's
exposure a unit of GRC spend removes -- and no amount of downstream machinery
fixes that. Reporting an optimal budget computed from an invented alpha implies
a precision that does not exist, so the question is inverted: not "spend this
much" but "this programme is worth running provided you believe X", where X is
something the reader can agree or disagree with.

The survival framing improves the question rather than answering it. Alpha is
still uncalibrated, but the quantity a programme has to move is now the annual
probability of failure, and *that* has external anchors alpha never had -- bank
failure rates, rating-agency default rates, observed crypto-lender failures,
and the price of the D&O and cyber cover that insures the same risk. The model
can be checked against something.

Four break-evens, in descending order of how directly a board can argue with
them:

1. `hazard_breakeven`     by how many basis points must this cut annual
                          failure probability to pay for itself? No alpha in
                          the answer at all.
2. `capitalization_band`  over what range of capitalization is a programme
                          worth running? Below it, wind down instead.
3. `franchise_breakeven`  how large must the franchise be before the programme
                          pays? A valuation question a firm already answers.
4. `alpha_breakeven`      the original question, per family, now including the
                          survival term.
"""

from dataclasses import dataclass, replace

from quant.env.env import EnvConfig, FirmEnv, evaluate
from quant.env.reward import PerpetuityValue
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.numerics import DEFAULT_PROFILE, NumericsProfile
from quant.params import DEFAULTS, FirmParams, GrcAlphas, ModelParams
from quant.solvers.pathwise import optimize_constant

FAMILIES = ("credit", "operational", "compliance")
# Below this much spend a programme is not a programme -- it is a rounding
# error on someone's budget line, so treat it as "not worth running".
#
# **Stated per YEAR, and compared against annualized spend.** `EvalResult.grc`
# is a per-*period* flow, so a threshold in per-period units silently means
# something different at every decision frequency: the same programme, decided
# monthly instead of quarterly, spends a third as much per period and would read
# as immaterial. Every row of every sweep would flip to "uneconomic" and nothing
# would error.
#
# 0.40 a year is the old 0.10 a quarter, so quarterly verdicts are unchanged.
ANNUAL_MATERIALITY = 0.40


def is_material(result, firm: FirmParams) -> bool:
    """Is this programme large enough to be worth calling a programme?

    Annualizes before comparing, which is the whole point -- see
    ANNUAL_MATERIALITY. Takes the firm rather than a bare number of periods so
    the caller cannot forget which frequency the result came from.
    """
    return result.total_grc * firm.periods_per_year > ANNUAL_MATERIALITY
# A programme has to add at least this share of firm value to count as worth
# running, which is the value-side counterpart of the spend threshold above.
MATERIALITY_VALUE = 0.01


# The franchise EnvConfig.at_frequency installs by default. Named here
# because the capitalization sweep has to scale it with the firm.
DEFAULT_FRANCHISE = 20.0


def solve(
    firm: FirmParams,
    crn,
    periods: int,
    steps: int,
    params: ModelParams = DEFAULTS,
    alphas=None,
    **config,
):
    """One constant-policy solve. The right policy class for a budget question:
    it answers "how much should we spend", not "how should we react".

    The config and the sampler are both built from `params`, which is the seam
    `standard_env` exists to close: this function used to pin
    `EnvConfig.quarterly` and `MonteCarloSampler(DEFAULTS.sampler)`
    independently, so pointing it at any other decision frequency would have
    given the firm a quarter's losses per period while everything else
    converted correctly.
    """
    settings = dict(firm=firm, **config)
    if alphas is not None:
        settings["alphas"] = alphas
    env = FirmEnv(
        EnvConfig.at_frequency(params, periods, **settings),
        MonteCarloSampler(params.sampler),
    )
    return optimize_constant(env, crn, n_steps=steps)[1]


@dataclass
class HazardBreakeven:
    annual_cost: float
    franchise: float
    basis_points: float

    def sentence(self) -> str:
        return (
            f"a programme costing {self.annual_cost:.2f} a year, against a franchise of "
            f"{self.franchise:.2f}, must cut the annual probability of failure by at least "
            f"{self.basis_points:.0f} basis points to pay for itself"
        )


def hazard_breakeven(firm, crn, periods=60, steps=2500, params=DEFAULTS) -> HazardBreakeven:
    """The headline, and the only one with no alpha in it.

        delta_h* = annual cost / franchise value

    A programme is worth its cost if it removes at least that much expected
    annual destruction of the going concern. Both inputs are things a board
    already has a view on -- what the programme costs, and what the business is
    worth -- so the whole claim can be argued with without touching the model's
    uncalibrated parameters.
    """
    result = solve(firm, crn, periods, steps, params)
    annual_cost = result.total_grc * firm.periods_per_year
    franchise = result.value * result.going_concern_share
    return HazardBreakeven(annual_cost, franchise, 1e4 * annual_cost / franchise)


def capitalization_band(firm, crn, equities, periods=60, steps=2000, params=DEFAULTS):
    """Over what range of capitalization does a programme earn its keep?

    Non-monotone by construction, and that is the useful part. A firm with
    little capital left has little franchise to protect and should be winding
    down rather than spending; a firm with plenty faces so little hazard that
    the programme is uneconomic. The band between them is where GRC is a
    decision rather than a formality.
    """
    rows = []
    for equity in equities:
        ratio = equity / firm.initial_equity
        scaled = replace(
            firm,
            initial_equity=equity,
            # The control function scales with the firm; holding it fixed would
            # confound capitalization with how much GRC is already in place.
            initial_grc_stock=firm.initial_grc_stock * ratio,
        )
        # The franchise scales for exactly the same reason, and it did not
        # before. Held fixed at 20 against an equity of 4, the going concern is
        # worth five times its own book, so a barely-capitalized firm reads as
        # having everything to protect and the band comes out monotone --
        # reporting the constant rather than the capitalization. Once the
        # liability side made the balance sheet scale with equity this stopped
        # being a second-order confound.
        rows.append((
            equity,
            solve(scaled, crn, periods, steps, params,
                  terminal=PerpetuityValue(franchise=DEFAULT_FRANCHISE * ratio)),
        ))
    return rows


def franchise_breakeven(firm, crn, franchises, periods=60, steps=2000, params=DEFAULTS):
    """How much business must be at stake before the programme pays?

    Swept through the going-concern value itself rather than through the
    productivity that generates it. Sweeping productivity leaves the terminal
    franchise pinned, so the firm always has something worth protecting and
    every row reports "worth running" -- which is what it did before this was
    corrected.

    Useful because this is a valuation question a firm already answers, and
    alpha is not: *this programme pays if you believe the business is worth at
    least X beyond its book value.*
    """
    rows = []
    for franchise in franchises:
        rows.append((
            franchise,
            solve(firm, crn, periods, steps, params, terminal=PerpetuityValue(franchise=franchise)),
        ))
    return rows


def alpha_breakeven(
    firm, crn, family: str, low=0.02, high=1.0, iterations=8, periods=60,
    steps=1500, params=DEFAULTS,
) -> float | None:
    """Smallest effectiveness at which this family's programme earns its keep.

    Measured as the *value* the programme adds over not having it -- the same
    firm solved with this family's alpha set to zero, so spend buys nothing and
    the optimizer drives that budget to zero on its own.

    Deliberately not "the smallest alpha at which the budget is material",
    which was the static model's test and is the wrong question here. Optimal
    spend is non-monotone in effectiveness: the closed form ln(alpha X)/alpha
    rises and then falls, because a very effective programme needs very little
    spending on. Testing budget materiality therefore reports "no alpha works"
    for a family whose programme is enormously worthwhile -- which is what it
    did here before this was corrected.

    Value added is monotone in alpha, which is what a bisection needs.
    """
    baseline = solve(
        firm, crn, periods, steps, params, alphas=replace(params.alphas, **{family: 0.0})
    ).value
    threshold = MATERIALITY_VALUE * baseline

    def gain(alpha: float) -> float:
        alphas = replace(DEFAULTS.alphas, **{family: alpha})
        return solve(firm, crn, periods, steps, params, alphas=alphas).value - baseline

    if gain(high) < threshold:
        return None
    if gain(low) >= threshold:
        return low
    for _ in range(iterations):
        middle = (low + high) / 2
        if gain(middle) >= threshold:
            high = middle
        else:
            low = middle
    return high


def value_curvature(firm, crn, equities, periods=60, steps=2000, params=DEFAULTS):
    """Second difference of firm value in opening equity.

    The diagnostic to run before quoting any comparative static. A survival
    barrier makes the value function convex where failure is close and concave
    where it is not, so the firm is risk-averse when comfortably capitalized
    and risk-*loving* near failure -- gambling for resurrection, which is
    economically real and produces "cut GRC in a crisis" recommendations that
    are correct within the model and indefensible out of it.

    Positive entries mark the convex region.
    """
    values = [solve(replace(firm, initial_equity=e), crn, periods, steps, params).value for e in equities]
    curvature = []
    for index in range(1, len(equities) - 1):
        left, middle, right = equities[index - 1], equities[index], equities[index + 1]
        # Non-uniform-spacing second difference.
        slope_low = (values[index] - values[index - 1]) / (middle - left)
        slope_high = (values[index + 1] - values[index]) / (right - middle)
        curvature.append((middle, 2.0 * (slope_high - slope_low) / (right - left)))
    return values, curvature
