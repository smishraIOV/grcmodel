"""Rung 0: numerics properties that hold independently of any solver.

Three agreeing solvers on a badly-configured numerics stack agree wrongly, so
these run before anything about the economics is asserted. They pin the three
rules quant/numerics.py exists to enforce -- randomness drawn once on CPU,
reductions accumulated wide, and no tensor built without a dtype.
"""

import re
from pathlib import Path

import pytest
import torch

from quant.model import optimize_policy
from quant.numerics import CPU_FAST, FAST, PROFILES, REFERENCE, get_profile
from quant.params import DEFAULTS
from quant.simulate import build_model, sample_risk_draw
from quant.static import build_shock

QUANT = Path(__file__).resolve().parent.parent / "quant"
CONSTRUCTORS = ("tensor", "zeros", "ones", "full", "rand", "randn", "empty", "arange", "linspace")


def _calls(source: str, name: str) -> list[str]:
    """Every `torch.<name>(...)` call in `source`, with its arguments, spanning
    however many lines it takes."""
    found = []
    for match in re.finditer(rf"torch\.{name}\(", source):
        depth, index = 0, match.end() - 1
        while index < len(source):
            depth += {"(": 1, ")": -1}.get(source[index], 0)
            if depth == 0:
                break
            index += 1
        found.append(source[match.start() : index + 1])
    return found


def test_no_tensor_is_constructed_without_a_dtype():
    """The bug this whole module exists to prevent.

    quant/static.py used to build its shock with bare `torch.tensor([...])`,
    which silently produced float32 that optimize_policy then upcast to
    float64 -- so the exposures the closed-form benchmark is measured against
    had been rounded through float32 first. Nothing caught it because the
    result still looked like a float64 pipeline from the outside.

    quant/numerics.py is exempt: it is the one place a dtype is chosen.
    """
    offenders = []
    for path in sorted(QUANT.glob("*.py")):
        if path.name == "numerics.py":
            continue
        source = path.read_text()
        for name in CONSTRUCTORS:
            offenders += [
                f"{path.name}: {call.splitlines()[0]}"
                for call in _calls(source, name)
                if "dtype=" not in call
            ]
    assert not offenders, "build these through a NumericsProfile:\n" + "\n".join(offenders)


@pytest.mark.parametrize("profile", list(PROFILES.values()), ids=list(PROFILES))
def test_generator_lives_on_the_cpu(profile):
    """Rule 1. An MPS generator uses a different algorithm from CPU's, so the
    same seed would give different scenarios on different devices."""
    assert profile.rng_device.type == "cpu"
    assert profile.generator(0).device.type == "cpu"


def test_scenario_set_is_identical_across_profiles():
    """The payoff from rule 1: a profile change alters arithmetic rounding
    only, never which paths were averaged over.

    Asserted bitwise, not approximately. Both profiles run the inverse-CDF
    transform at float64 and round once at the end, so the float32 draw must
    be exactly the float32 rounding of the float64 one. Anything weaker would
    let a drifting sampler hide behind a tolerance.
    """
    reference = sample_risk_draw(4096, DEFAULTS.sampler, REFERENCE.generator(0), REFERENCE)
    fast = sample_risk_draw(4096, DEFAULTS.sampler, CPU_FAST.generator(0), CPU_FAST)

    for field in ("credit_loss", "op_occurs", "op_severity", "compliance_occurs"):
        wide = getattr(reference, field)
        narrow = getattr(fast, field)
        assert torch.equal(wide.to(torch.float32), narrow), field


def test_same_seed_reproduces_the_same_draw():
    first = sample_risk_draw(1024, DEFAULTS.sampler, REFERENCE.generator(7), REFERENCE)
    second = sample_risk_draw(1024, DEFAULTS.sampler, REFERENCE.generator(7), REFERENCE)
    assert torch.equal(first.credit_loss, second.credit_loss)
    assert torch.equal(first.op_severity, second.op_severity)


def test_reductions_accumulate_wider_than_they_compute():
    """Rule 2. Summing many small contributions into a large running total is
    where float32 loses first, and a Monte Carlo value is exactly that shape.

    Every input here is exactly representable in float32, so the error being
    measured is the accumulation's alone.
    """
    n = 2**16
    x = torch.cat([torch.tensor([2.0**24], dtype=torch.float32), torch.ones(n, dtype=torch.float32)])
    exact = 2.0**24 + n

    assert abs(x.sum().item() - exact) > 1.0, "input no longer exercises the failure"
    assert CPU_FAST.sum(x).item() == exact
    assert CPU_FAST.sum(x).dtype is torch.float32, "sum returns the input dtype"


def test_unknown_and_unavailable_profiles_are_refused():
    """No silent fallback: a run that quietly drops from MPS to CPU reports
    timings and tolerances for a configuration nobody chose."""
    with pytest.raises(ValueError, match="unknown profile"):
        get_profile("float128-please")
    if not FAST.available():
        with pytest.raises(RuntimeError, match="not available"):
            get_profile("fast")


@pytest.mark.skipif(not FAST.available(), reason="MPS not available on this machine")
def test_profiles_agree():
    """float32 on MPS must reach the same answer as float64 on CPU.

    Meaningful only because of rule 1: both solves see the same scenarios, so
    any disagreement is arithmetic rather than a different sample. Tolerances
    are the ones docs/quant-model.md quotes budgets to, not machine epsilon --
    the point is that the FAST profile is usable for the dynamic model's
    rollouts, not that it is bit-identical.

    The value tolerance was 1e-5 relative, which was tighter than that
    rationale and tighter than any number the documentation quotes. It held
    only by luck: raising the production curvature for the levered balance
    sheet took the disagreement from 2e-7 to 1.5e-5 and through it. The
    arithmetic is 3000 Adam steps accumulating in float32, which is what this
    test exists to bound rather than to eliminate -- `production` was rewritten
    with `expm1` on the strength of this failure and moved it by half a
    percent, so the residue is the optimizer, not one cancellation.

    1e-4 is still two orders of magnitude tighter than the precision anything
    downstream claims, and the per-family budgets below are unchanged at 1e-3
    absolute -- those are the numbers a reader would act on.
    """
    reference = optimize_policy(
        build_model(), build_shock(profile=REFERENCE), n_steps=3000, profile=REFERENCE
    )
    fast = optimize_policy(
        build_model(), build_shock(profile=FAST), n_steps=3000, profile=FAST
    )

    assert abs(fast.value - reference.value) < 1e-4 * abs(reference.value)
    for family in ("credit", "operational", "compliance"):
        assert abs(getattr(fast, family) - getattr(reference, family)) < 1e-3, family
    assert abs(fast.constrained_fraction - reference.constrained_fraction) < 1e-3
