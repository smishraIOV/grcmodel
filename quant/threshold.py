"""Break-even analysis: the decision-useful form of the model.

Nothing in this repo can calibrate `alpha` -- how much of a risk family's
exposure one unit of GRC spend actually removes. That parameter, not the
mathematics, is what stands between this model and a real recommendation, so
reporting a single optimal budget computed from an invented alpha implies a
precision that does not exist.

Inverting the question makes the output robust to that gap. Instead of "spend
this much", the model answers "this programme is worth running provided you
believe a unit of spend removes at least this much exposure" -- a claim a risk
owner can actually agree or disagree with.

Two thresholds matter, and they differ:

- The risk-neutral break-even, alpha = 1/exposure, has a closed form. Below it
  a firm facing no financing friction spends nothing on the family, because a
  unit of spend buys back less than a unit of expected loss.
- The frictional firm keeps spending below that point, because GRC also
  narrows the spread of internal wealth and so protects its ability to fund
  investment. That gap is the Froot-Stein premium, expressed as a threshold.
"""

from dataclasses import dataclass, replace

from quant.cli import parse_profile, print_header
from quant.model import family_exposures, optimize_policy
from quant.numerics import DEFAULT_PROFILE, NumericsProfile
from quant.static import ALPHAS, build_model, build_shock

FAMILIES = ("credit", "operational", "compliance")
ALPHA_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.8]
# Below this much spend a programme is not a programme -- it is a rounding
# error on someone's budget line, so treat it as "not worth running".
MATERIALITY = 0.10


@dataclass
class ThresholdRow:
    alpha: float
    budget: float
    value: float
    value_without: float

    @property
    def net_gain(self) -> float:
        return self.value - self.value_without


def run_family_sweep(
    family: str,
    alphas: list[float] | None = None,
    n_steps: int = 3000,
    profile: NumericsProfile = DEFAULT_PROFILE,
) -> list[ThresholdRow]:
    """Sweep one family's mitigation effectiveness, holding the others fixed.

    The counterfactual "no programme" value is obtained by setting the family's
    alpha to zero: spend then buys no mitigation, so the optimizer drives that
    budget to zero on its own rather than needing it pinned there.
    """
    draw = build_shock(profile=profile)
    without = optimize_policy(
        build_model(alphas=replace(ALPHAS, **{family: 0.0})),
        draw,
        n_steps=n_steps,
        profile=profile,
    ).value

    rows = []
    for alpha in alphas if alphas is not None else ALPHA_SWEEP:
        result = optimize_policy(
            build_model(alphas=replace(ALPHAS, **{family: alpha})),
            draw,
            n_steps=n_steps,
            profile=profile,
        )
        rows.append(
            ThresholdRow(
                alpha=alpha,
                budget=getattr(result, family),
                value=result.value,
                value_without=without,
            )
        )
    return rows


def budget_at(
    family: str,
    alpha: float,
    n_steps: int = 2000,
    profile: NumericsProfile = DEFAULT_PROFILE,
) -> float:
    result = optimize_policy(
        build_model(alphas=replace(ALPHAS, **{family: alpha})),
        build_shock(profile=profile),
        n_steps=n_steps,
        profile=profile,
    )
    return getattr(result, family)


def materiality_threshold(
    family: str,
    low: float = 0.01,
    high: float = 1.0,
    iterations: int = 10,
    profile: NumericsProfile = DEFAULT_PROFILE,
) -> float | None:
    """Smallest alpha at which the family earns a material budget, by bisection.

    Reported instead of the lowest grid point that clears MATERIALITY, because
    a grid point says only "somewhere below here" and invites reading the
    sweep's resolution as the model's precision.
    """
    if budget_at(family, high, profile=profile) < MATERIALITY:
        return None
    if budget_at(family, low, profile=profile) >= MATERIALITY:
        return low
    for _ in range(iterations):
        midpoint = (low + high) / 2
        if budget_at(family, midpoint, profile=profile) >= MATERIALITY:
            high = midpoint
        else:
            low = midpoint
    return high


def main(profile: NumericsProfile | None = None) -> None:
    profile = profile or parse_profile(__doc__.splitlines()[0])
    print_header(profile)
    exposures = family_exposures(build_shock(profile=profile))

    for family in FAMILIES:
        exposure = exposures[family]
        risk_neutral = 1.0 / exposure
        rows = run_family_sweep(family, profile=profile)
        threshold = materiality_threshold(family, profile=profile)

        print(f"{family.upper()}  (expected exposure {exposure:.2f})")
        print(f"  risk-neutral break-even alpha = 1/exposure = {risk_neutral:.3f}")
        print(f"  {'alpha':>7} | {'optimal spend':>13} | {'firm value':>10} | {'net gain':>9}")
        previous_alpha = 0.0
        for row in rows:
            crosses = previous_alpha < risk_neutral <= row.alpha
            marker = "  <- risk-neutral break-even" if crosses else ""
            print(
                f"  {row.alpha:>7.2f} | {row.budget:>13.4f} | {row.value:>10.4f} | "
                f"{row.net_gain:>9.4f}{marker}"
            )
            previous_alpha = row.alpha
        if threshold is None:
            print("  -> no plausible alpha justifies a material programme")
        else:
            print(f"  -> worth running once you believe alpha >= {threshold:.3f}")
            if threshold < risk_neutral:
                print(
                    f"     that is {risk_neutral - threshold:.3f} below the risk-neutral "
                    f"break-even: the programme pays at effectiveness levels that pure "
                    f"expected-loss reduction would reject, because it also protects "
                    f"investment capacity. That gap is the Froot-Stein premium."
                )
            else:
                print(
                    "     at or above the risk-neutral break-even: this family's losses "
                    "carry too little spread for the financing friction to add much."
                )
        print()


if __name__ == "__main__":
    main()
