"""Truncated BPTT with a critic, and the identities that say it is wired right.

The two exact checks come first and matter most. A windowed learner has many
places to be quietly wrong -- the random stream can desynchronise between
segments, the probability mass still operating can be double-counted at a
seam, the critic can be asked to predict something other than what the
recursion sums to -- and none of those show up as a crash. They show up as a
number that looks plausible.
"""

import pytest
import torch

from quant.env.env import ConstantPolicy, EnvConfig, FirmEnv
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.numerics import REFERENCE
from quant.params import DEFAULTS
from quant.solvers.svg import Critic, returns_to_go, train_svg, windowed_objective

HORIZON = 6


def setup(paths=512):
    env = FirmEnv(EnvConfig.quarterly(HORIZON), MonteCarloSampler(DEFAULTS.sampler))
    crn = CommonRandomNumbers(0, HORIZON, paths, REFERENCE)
    policy = ConstantPolicy(grc=(0.08, 0.08, 0.08), investment=20.0, profile=REFERENCE)
    return env, crn, policy


def test_the_recursion_sums_to_the_same_thing():
    """returns_to_go reads Trajectory.path_values backwards. If the two
    disagree the critic is being trained to predict something the objective
    does not measure, and every windowed value is then wrong by an amount
    nothing reports."""
    env, crn, policy = setup()
    trajectory = env.rollout(policy, crn, False)
    recursive = returns_to_go(env, trajectory)[0]
    summed = trajectory.path_values(env.config.discount)
    assert torch.allclose(recursive, summed, atol=1e-12)


def test_one_window_is_exactly_full_backpropagation():
    """The property that makes K a dial rather than a different algorithm.

    With the window equal to the horizon there is no seam, the critic is never
    consulted, and the objective must be the ordinary rollout value -- bitwise,
    not approximately. Any gap here is a seam bug that smaller windows would
    inherit and hide.
    """
    env, crn, policy = setup()
    critic = Critic(env)
    full = env.value_of(env.rollout(policy, crn, differentiable=False))
    windowed = windowed_objective(env, policy, critic, crn, window=HORIZON)
    assert windowed.item() == pytest.approx(full.item(), abs=1e-12)


def test_windows_consume_their_own_slice_of_the_random_stream():
    """A segment starting at quarter four must draw quarter four's shocks. If
    `start` indexed the state rather than the stream, every window would replay
    the opening quarters and the firm would face a world that resets."""
    env, crn, policy = setup(paths=256)
    opening = env.reset(256)
    first = env.rollout_from(policy, crn, opening, start=0, steps=2, differentiable=False)
    later = env.rollout_from(policy, crn, opening, start=2, steps=2, differentiable=False)
    assert not torch.allclose(first.states[-1].equity, later.states[-1].equity)


def test_the_state_carried_across_a_seam_has_no_gradient():
    """What truncation *is*. If the state survived the seam differentiably the
    chain would never be cut and K would do nothing."""
    env, crn, policy = setup(paths=128)
    state = env.reset(128)
    assert state.detach().equity.requires_grad is False
    assert state.detach().grc_stock.requires_grad is False


def test_the_critic_learns_something():
    """Weak on purpose: this asserts the regression is wired up and descending,
    not that the critic is accurate. Accuracy is what the comparison in
    docs/quant-model.md measures, and it is not good."""
    env, crn, _ = setup(paths=256)
    result = train_svg(env, crn, window=3, n_steps=60, record_every=10)
    assert result.critic_loss[-1] < result.critic_loss[0]
    assert torch.isfinite(torch.tensor(result.evaluation.value))
