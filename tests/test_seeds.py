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


def test_in_sample_values_are_optimistic():
    """Both solvers are fitted against the training draw, so scoring them there
    should flatter them. If it does not, training and evaluation are sharing
    scenarios somewhere they should not be."""
    studies = study()
    assert studies["constant"].overfit() > -1e-9
    assert studies["neural"].overfit() > -1e-9


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
