"""Multi-seed studies, and the separation that makes them honest."""

import pytest

from quant.env.env import EnvConfig, FirmEnv
from quant.env.shocks import MonteCarloSampler
from quant.numerics import REFERENCE
from quant.params import DEFAULTS
from quant.studies.seeds import HELD_OUT_OFFSET, edge, run_seeds

SEEDS = (0, 1, 2)


def study():
    env = FirmEnv(EnvConfig.quarterly(3), MonteCarloSampler(DEFAULTS.sampler))
    return run_seeds(
        env, seeds=SEEDS, train_paths=256, test_paths=512,
        constant_steps=500, neural_steps=400,
    )


def test_training_and_evaluation_use_disjoint_scenarios():
    """A policy graded on the paths it was fitted to is marking its own
    homework, and one of these solvers fits a free parameter per path."""
    assert HELD_OUT_OFFSET > 1000
    assert not set(SEEDS) & {s + HELD_OUT_OFFSET for s in SEEDS}


def test_every_seed_is_recorded_separately():
    """The point of the study is the spread, so the individual runs have to
    survive to be reported rather than being averaged on the way in."""
    studies = study()
    for name, result in studies.items():
        assert len(result.held_out) == len(SEEDS), name
        assert len(result.in_sample) == len(SEEDS), name
        assert result.spread()[0] <= result.median() <= result.spread()[1]


# `overfit()` is a difference of two Monte Carlo estimates -- 256 training paths
# against 512 held-out ones -- so it carries sampling noise of its own and can
# come out slightly negative for a policy that has barely fitted anything.
#
# The tolerance is now *derived* rather than picked, having been picked twice
# and failed twice. Held-out values across the three seeds span about 0.4 on a
# value near 35.7, so a single estimate carries roughly 0.8% of seed noise and
# a difference of two of them carries more. Any tolerance inside that was going
# to fail eventually; -1e-9 (machine epsilon on a Monte Carlo quantity) failed
# first, then -1e-3.
#
# 1% is the scale the measurement can actually resolve. Scenario leakage drives
# this *up*, toward a zero gap, not down, and a training loop scoring the wrong
# draw would be far larger than this.
OVERFIT_NOISE = 1e-2


def test_in_sample_values_are_optimistic():
    """Both solvers are fitted against the training draw, so scoring them there
    should flatter them. If it does not by more than sampling noise, training
    and evaluation are sharing scenarios somewhere they should not be."""
    studies = study()
    assert studies["constant"].overfit() > -OVERFIT_NOISE
    assert studies["neural"].overfit() > -OVERFIT_NOISE


def test_a_single_run_cannot_separate_the_solvers_at_a_short_horizon():
    """The finding this module exists to make sayable: at a short horizon the
    two solvers differ by less than the spread across seeds, so any single run
    reporting one as better is reporting its seed."""
    studies = study()
    advantage = abs(edge(studies))
    noise = max(s.half_width() / abs(s.median()) for s in studies.values())
    assert advantage < max(noise, 0.02), (
        f"edge {advantage:.3%} vs seed noise {noise:.3%} -- if this now separates "
        "cleanly the short-horizon verdict in docs/quant-model.md needs revisiting"
    )


def test_the_learner_clips_its_gradient():
    """Guards a measured failure, not a precaution.

    At the five-year horizon one run in six used to destroy itself: 500 steps of
    normal convergence with the gradient norm falling, then a single step at
    2.7e5 -- four hundred thousand times the previous -- and a worthless policy
    eight steps later. The amplifier is the run channel's straight-through
    relaxation, and Adam makes it worse rather than better, because a spike
    against a small running variance produces a step far larger than the
    learning rate.

    The threshold is where the distributions separate: steady-state norms are
    0.6-2, the cold start peaks near 82, the spike is 2.7e5. On a healthy
    seed the clip fires zero times in 600 steps.
    """
    import inspect

    from quant.solvers.neural import train_pathwise

    source = inspect.getsource(train_pathwise)
    assert "clip_grad_norm_" in source, "the learner has lost its gradient clip"

    signature = inspect.signature(train_pathwise)
    clip = signature.parameters["grad_clip"].default
    assert clip is not None, "clipping must be on by default"
    # Well above the cold start's ~82, far below the 2.7e5 spike.
    assert 82.0 < clip < 1e4, f"clip {clip} does not separate the two regimes"
