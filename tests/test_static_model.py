"""Tests for the corrected Froot-Stein model.

These replace an earlier test asserting that optimal GRC investment rises with
financing-cost convexity. That assertion passed only because the cost function
was dimensionally inconsistent and the firm was parameterized into permanent
insolvency; it fails once either is fixed, so it was defending a bug rather
than the economics (docs/critical-review.md F1, F2, F5).

What is tested here instead is the mechanism: that the model's answers do not
depend on the currency unit, that switching the financing friction off recovers
a closed-form risk-neutral benchmark, and that GRC investment responds to the
*spread* of losses at constant mean -- which is what Froot-Stein actually
claims.
"""

import math
from dataclasses import replace

import torch

from quant.model import (
    FirmValueModel,
    GrcAlphas,
    GrcBudgets,
    RiskDraw,
    family_exposures,
    frictionless_benchmark,
    optimize_policy,
)
from quant.static import (
    ALPHAS,
    build_model,
    build_shock,
    run_friction_premium,
    run_spread_sweep,
)

FAMILIES = ("credit", "operational", "compliance")


def test_zero_friction_recovers_closed_form_benchmark():
    """With no financing friction the firm is risk-neutral over wealth, the
    objective separates across families, and each budget has a closed form:
    g* = max(0, ln(alpha*exposure)/alpha). Any deviation means the objective or
    the optimizer disagrees with the analysis."""
    draw = build_shock()
    benchmark = frictionless_benchmark(draw, ALPHAS)
    result = optimize_policy(build_model(financing_scale=0.0), draw, n_steps=6000)

    for family in FAMILIES:
        expected = getattr(benchmark, family)
        actual = getattr(result, family)
        # A budget whose benchmark is exactly zero is only approached
        # asymptotically through softplus, so it gets a looser tolerance than
        # one sitting at an interior optimum.
        tolerance = 1e-3 if expected == 0.0 else 1e-6
        assert abs(actual - expected) < tolerance, f"{family}: {actual} != {expected}"


def test_friction_premium_is_positive():
    """The Froot-Stein claim in readme.md: costly external finance makes a
    risk-neutral firm invest in risk management beyond what expected-loss
    reduction alone justifies."""
    frictionless, frictional = run_friction_premium()

    assert frictional.total > frictionless.total
    for family in FAMILIES:
        assert getattr(frictional, family) >= getattr(frictionless, family) - 1e-6

    # The compliance programme is the sharpest case: a friction-free firm would
    # not run one at all at these parameters, so its entire budget is premium.
    assert frictionless.compliance < 1e-3
    assert frictional.compliance > 0.1


def test_mean_preserving_spread_raises_grc():
    """The mechanism itself. Holding the credit loss mean fixed and widening
    its spread must raise optimal credit GRC -- a convex financing cost makes
    narrowing the spread of internal wealth valuable, independent of any
    particular cost-function shape."""
    results = [(spread, result.credit) for spread, result in run_spread_sweep()]
    budgets = [credit for _spread, credit in results]

    for earlier, later in zip(budgets, budgets[1:]):
        assert later >= earlier - 1e-6, f"not monotone: {budgets}"
    assert budgets[-1] > budgets[0] + 0.1

    # At zero spread there is nothing for the convex premium to act on through
    # the credit channel, so credit GRC falls back to the risk-neutral level.
    benchmark = frictionless_benchmark(build_shock(credit_spread=0.0), ALPHAS)
    assert abs(budgets[0] - benchmark.credit) < 1e-3


def test_optimal_policy_is_unit_invariant():
    """Redenominating the firm -- dollars to cents -- must scale the optimal
    budgets by exactly the same factor and change nothing real. The previous
    cost function failed this, which is what made its convexity result an
    artifact (docs/critical-review.md F1)."""

    def solve(k: float) -> float:
        draw = build_shock()
        scaled = RiskDraw(
            credit_loss=draw.credit_loss * k,
            op_occurs=draw.op_occurs,
            op_severity=draw.op_severity * k,
            compliance_occurs=draw.compliance_occurs,
            compliance_severity=draw.compliance_severity * k,
            base_weight=draw.base_weight,
            compliance_base_probability=draw.compliance_base_probability,
        )
        model = build_model(
            initial_equity=16.0 * k,
            distress_reference=5.0 * k,
            production_curvature=10.0 * k,
            alphas=GrcAlphas(credit=0.3 / k, operational=0.3 / k, compliance=0.3 / k),
        )
        return optimize_policy(model, scaled, n_steps=6000).total / k

    baseline = solve(1.0)
    for k in (0.1, 10.0):
        assert abs(solve(k) - baseline) < 1e-3 * max(1.0, baseline), f"k={k} broke invariance"


def test_each_family_responds_to_its_own_effectiveness():
    """The risk taxonomy has to be load-bearing. Previously the three loss
    fields were summed and mitigated uniformly, so the split was decorative
    (docs/critical-review.md F4). Making one family's GRC more effective should
    move that family's budget most."""
    draw = build_shock()
    baseline = optimize_policy(build_model(), draw)

    for family in FAMILIES:
        doubled = optimize_policy(
            build_model(alphas=replace(ALPHAS, **{family: 2 * getattr(ALPHAS, family)})), draw
        )
        own_change = abs(getattr(doubled, family) - getattr(baseline, family))
        others = [f for f in FAMILIES if f != family]
        other_changes = [abs(getattr(doubled, f) - getattr(baseline, f)) for f in others]
        assert own_change > max(other_changes), (
            f"{family}: own budget moved {own_change:.4f}, others {other_changes}"
        )


def test_compliance_probability_channel_is_unbiased_and_differentiable():
    """Compliance GRC acts on breach probability, which cannot be pushed
    through a Bernoulli sample. The likelihood-ratio reweighting that replaces
    it must reproduce the analytic expectation and carry a correct gradient."""
    p0, alpha, severity, n_paths = 0.05, 0.4, 15.0, 200_000
    generator = torch.Generator().manual_seed(0)
    occurs = (torch.rand(n_paths, generator=generator, dtype=torch.float64) < p0).to(torch.float64)

    draw = RiskDraw(
        credit_loss=torch.zeros(n_paths, dtype=torch.float64),
        op_occurs=torch.zeros(n_paths, dtype=torch.float64),
        op_severity=torch.zeros(n_paths, dtype=torch.float64),
        compliance_occurs=occurs,
        compliance_severity=torch.full((n_paths,), severity, dtype=torch.float64),
        base_weight=torch.ones(n_paths, dtype=torch.float64),
        compliance_base_probability=p0,
    )
    alphas = GrcAlphas(credit=0.0, operational=0.0, compliance=alpha)

    g = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    budgets = GrcBudgets(
        credit=torch.zeros((), dtype=torch.float64),
        operational=torch.zeros((), dtype=torch.float64),
        compliance=g,
    )
    weights = draw.path_weights(budgets, alphas)
    expected_loss = (weights * draw.compliance_occurs * draw.compliance_severity).sum()

    analytic = p0 * math.exp(-alpha * 2.0) * severity
    assert abs(expected_loss.item() - analytic) < 0.02 * analytic

    expected_loss.backward()
    analytic_gradient = -alpha * analytic
    assert abs(g.grad.item() - analytic_gradient) < 0.05 * abs(analytic_gradient)


def test_default_parameters_sit_in_the_constrained_regime():
    """The model only has content where the finance constraint binds in some
    states but not all. Permanent insolvency (the old parameterization) and
    permanent slack both make GRC's effect on survival vacuous, and both
    silently reverse or flatten the headline result (docs/critical-review.md F2)."""
    _frictionless, frictional = run_friction_premium()
    assert 0.05 < frictional.constrained_fraction < 0.95
    assert frictional.value > 0, "a firm worth running should have positive value"


def test_exposures_match_the_constructed_shock():
    """Guards the closed-form benchmark: if exposures drift from what
    build_shock actually contains, every benchmark assertion above is
    measuring the wrong thing."""
    exposures = family_exposures(build_shock())
    assert exposures["credit"] == 4.0
    assert exposures["operational"] == 3.5
    assert exposures["compliance"] == 1.5


def test_production_has_the_expected_unconstrained_optimum():
    """I* = S*ln(A) is used to reason about which states are constrained."""
    model: FirmValueModel = build_model()
    assert abs(model.unconstrained_investment() - 10.0 * math.log(3.0)) < 1e-12
