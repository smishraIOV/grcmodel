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
"""

import torch

from quant.device import get_device
from quant.model import (
    FirmValueModel,
    GrcAlphas,
    PolicyResult,
    RiskDraw,
    frictionless_benchmark,
    optimize_policy,
)

INITIAL_EQUITY = 16.0
FINANCING_CONVEXITY = 2.0
DISTRESS_REFERENCE = 5.0
PRODUCTION_SCALE = 3.0
PRODUCTION_CURVATURE = 10.0

CREDIT_MEAN = 4.0
CREDIT_SPREAD = 2.0
OP_SEVERITY = 7.0
# Compliance failure is licence-to-operate risk (docs/framework.md section 1):
# rare, and severe enough to dwarf a bad credit year when it lands.
COMPLIANCE_PROBABILITY = 0.05
COMPLIANCE_SEVERITY = 30.0

ALPHAS = GrcAlphas(credit=0.3, operational=0.3, compliance=0.3)
SPREAD_SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0]


def build_shock(
    credit_spread: float = CREDIT_SPREAD,
    compliance_severity: float = COMPLIANCE_SEVERITY,
    compliance_probability: float = COMPLIANCE_PROBABILITY,
) -> RiskDraw:
    """Four states: (good, bad) business crossed with (no breach, breach).

    `credit_spread` widens the gap between the good and bad credit outcome
    while holding its mean at CREDIT_MEAN, so sweeping it is a mean-preserving
    spread.
    """
    business_weight = torch.tensor([0.5, 0.5])
    credit_by_state = torch.tensor([CREDIT_MEAN - credit_spread, CREDIT_MEAN + credit_spread])
    op_occurs_by_state = torch.tensor([0.0, 1.0])
    op_severity_by_state = torch.tensor([0.0, OP_SEVERITY])

    breach_weight = torch.tensor([1.0 - compliance_probability, compliance_probability])
    breach_by_state = torch.tensor([0.0, 1.0])

    return RiskDraw(
        credit_loss=credit_by_state.repeat_interleave(2),
        op_occurs=op_occurs_by_state.repeat_interleave(2),
        op_severity=op_severity_by_state.repeat_interleave(2),
        compliance_occurs=breach_by_state.repeat(2),
        compliance_severity=torch.full((4,), compliance_severity),
        base_weight=business_weight.repeat_interleave(2) * breach_weight.repeat(2),
        compliance_base_probability=compliance_probability,
    )


def build_model(financing_scale: float = 1.0, **overrides) -> FirmValueModel:
    params = dict(
        initial_equity=INITIAL_EQUITY,
        financing_convexity=FINANCING_CONVEXITY,
        distress_reference=DISTRESS_REFERENCE,
        production_scale=PRODUCTION_SCALE,
        production_curvature=PRODUCTION_CURVATURE,
        alphas=ALPHAS,
        financing_scale=financing_scale,
    )
    params.update(overrides)
    return FirmValueModel(**params)


def run_friction_premium(
    draw: RiskDraw | None = None, **model_overrides
) -> tuple[PolicyResult, PolicyResult]:
    """Solve the same firm with the financing friction off and on.

    Returns (frictionless, frictional). Every unit by which the frictional
    budgets exceed the frictionless ones is Froot-Stein premium.
    """
    draw = draw if draw is not None else build_shock()
    device = get_device()
    frictionless = optimize_policy(build_model(financing_scale=0.0, **model_overrides), draw, device=device)
    frictional = optimize_policy(build_model(financing_scale=1.0, **model_overrides), draw, device=device)
    return frictionless, frictional


def run_spread_sweep(spreads: list[float] | None = None) -> list[tuple[float, PolicyResult]]:
    """Mean-preserving spread of the credit shock: mean fixed, spread rising."""
    device = get_device()
    model = build_model()
    return [
        (spread, optimize_policy(model, build_shock(credit_spread=spread), device=device))
        for spread in (spreads if spreads is not None else SPREAD_SWEEP)
    ]


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")

    draw = build_shock()
    benchmark = frictionless_benchmark(draw, ALPHAS)
    frictionless, frictional = run_friction_premium(draw)

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
          f"{CREDIT_MEAN}, convexity {FINANCING_CONVEXITY})")
    print(f"{'spread':>8} | {'credit GRC':>10} | {'total GRC':>10} | {'firm value':>10}")
    for spread, result in run_spread_sweep():
        print(f"{spread:>8.2f} | {result.credit:>10.4f} | {result.total:>10.4f} | {result.value:>10.4f}")


if __name__ == "__main__":
    main()
