"""Static Froot-Stein model (docs/quant-model.md section 5.1).

A four-state world -- business good/bad crossed with compliance breach/no
breach -- carrying explicit probabilities. Small enough to reason about by
hand, but with genuine uncertainty in it, which the Froot-Stein mechanism
requires: it is a Jensen's-inequality effect, so there must be spread for the
convex financing premium to act on.

The two results this module reports:

1. `run_friction_premium` -- the same firm solved with and without a financing
   friction. The gap between the two GRC policies is the Froot-Stein premium:
   spend that exists only because external finance is costly. That gap is the
   quantitative form of readme.md's claim that GRC adds value.

2. `run_spread_sweep` -- a mean-preserving spread of the credit shock. Optimal
   credit GRC rises with spread at constant mean, which is the mechanism
   itself rather than a by-product of any particular cost-function shape.

An earlier version of this module swept the financing convexity instead. That
result did not survive a dimensional correction to the cost function; see
docs/critical-review.md F1-F2.

Every tensor here is built through a NumericsProfile. Before that, these
constructions carried no `dtype=` at all and so were silently float32, upcast
to float64 later inside optimize_policy -- which meant the exposures the
closed-form benchmark is measured against had been rounded through float32
first (quant/numerics.py).
"""

from quant.cli import parse_profile, print_header
from quant.model import (
    FirmValueModel,
    PolicyResult,
    RiskDraw,
    frictionless_benchmark,
    optimize_policy,
)
from quant.numerics import DEFAULT_PROFILE, NumericsProfile
from quant.params import DEFAULTS, SPREAD_SWEEP, FirmParams, GrcAlphas, ShockParams

ALPHAS: GrcAlphas = DEFAULTS.alphas


def build_shock(
    credit_spread: float | None = None,
    compliance_severity: float | None = None,
    compliance_probability: float | None = None,
    shock: ShockParams = DEFAULTS.shock,
    profile: NumericsProfile = DEFAULT_PROFILE,
) -> RiskDraw:
    """Four states: (good, bad) business crossed with (no breach, breach).

    `credit_spread` widens the gap between the good and bad credit outcome
    while holding its mean at `shock.credit_mean`, so sweeping it is a
    mean-preserving spread.
    """
    spread = shock.credit_spread if credit_spread is None else credit_spread
    severity = shock.compliance_severity if compliance_severity is None else compliance_severity
    probability = (
        shock.compliance_probability
        if compliance_probability is None
        else compliance_probability
    )

    business_weight = profile.tensor([0.5, 0.5])
    credit_by_state = profile.tensor([shock.credit_mean - spread, shock.credit_mean + spread])
    op_occurs_by_state = profile.tensor([0.0, 1.0])
    op_severity_by_state = profile.tensor([0.0, shock.op_severity])

    breach_weight = profile.tensor([1.0 - probability, probability])
    breach_by_state = profile.tensor([0.0, 1.0])

    return RiskDraw(
        credit_loss=credit_by_state.repeat_interleave(2),
        op_occurs=op_occurs_by_state.repeat_interleave(2),
        op_severity=op_severity_by_state.repeat_interleave(2),
        compliance_occurs=breach_by_state.repeat(2),
        compliance_severity=profile.full((4,), severity),
        base_weight=business_weight.repeat_interleave(2) * breach_weight.repeat(2),
        compliance_base_probability=probability,
    )


def build_model(
    financing_scale: float = 1.0, firm: FirmParams = DEFAULTS.firm, **overrides
) -> FirmValueModel:
    params = dict(
        initial_equity=firm.initial_equity,
        financing_convexity=firm.financing_convexity,
        distress_reference=firm.distress_reference,
        production_scale=firm.production_scale,
        production_curvature=firm.production_curvature,
        alphas=ALPHAS,
        financing_scale=financing_scale,
    )
    params.update(overrides)
    return FirmValueModel(**params)


def run_friction_premium(
    draw: RiskDraw | None = None,
    profile: NumericsProfile = DEFAULT_PROFILE,
    **model_overrides,
) -> tuple[PolicyResult, PolicyResult]:
    """Solve the same firm with the financing friction off and on.

    Returns (frictionless, frictional). Every unit by which the frictional
    budgets exceed the frictionless ones is Froot-Stein premium.
    """
    draw = draw if draw is not None else build_shock(profile=profile)
    frictionless = optimize_policy(
        build_model(financing_scale=0.0, **model_overrides), draw, profile=profile
    )
    frictional = optimize_policy(
        build_model(financing_scale=1.0, **model_overrides), draw, profile=profile
    )
    return frictionless, frictional


def run_spread_sweep(
    spreads: list[float] | None = None, profile: NumericsProfile = DEFAULT_PROFILE
) -> list[tuple[float, PolicyResult]]:
    """Mean-preserving spread of the credit shock: mean fixed, spread rising."""
    model = build_model()
    return [
        (
            spread,
            optimize_policy(
                model, build_shock(credit_spread=spread, profile=profile), profile=profile
            ),
        )
        for spread in (spreads if spreads is not None else SPREAD_SWEEP)
    ]


def main(profile: NumericsProfile | None = None) -> None:
    profile = profile or parse_profile(__doc__.splitlines()[0])
    print_header(profile)

    draw = build_shock(profile=profile)
    benchmark = frictionless_benchmark(draw, ALPHAS)
    frictionless, frictional = run_friction_premium(draw, profile=profile)

    print("Froot-Stein premium -- the same firm, financing friction off vs on")
    print(f"{'budget':>14} | {'no friction':>11} | {'closed form':>11} | {'friction':>9} | {'premium':>8}")
    for name, attr in (("credit", "credit"), ("operational", "operational"), ("compliance", "compliance")):
        off, on, closed = getattr(frictionless, attr), getattr(frictional, attr), getattr(benchmark, attr)
        print(f"{name:>14} | {off:>11.4f} | {closed:>11.4f} | {on:>9.4f} | {on - off:>8.4f}")
    print(f"{'total':>14} | {frictionless.total:>11.4f} | {benchmark.total():>11.4f} | "
          f"{frictional.total:>9.4f} | {frictional.total - frictionless.total:>8.4f}")
    print(f"\n  firm value: {frictionless.value:.4f} (no friction) -> {frictional.value:.4f} (friction)")
    print(f"  finance constraint binds on {frictional.constrained_fraction:.1%} of probability mass")

    print("\nMean-preserving spread of the credit shock (mean held at "
          f"{DEFAULTS.shock.credit_mean}, convexity {DEFAULTS.firm.financing_convexity})")
    print(f"{'spread':>8} | {'credit GRC':>10} | {'total GRC':>10} | {'firm value':>10}")
    for spread, result in run_spread_sweep(profile=profile):
        print(f"{spread:>8.2f} | {result.credit:>10.4f} | {result.total:>10.4f} | {result.value:>10.4f}")


if __name__ == "__main__":
    main()
