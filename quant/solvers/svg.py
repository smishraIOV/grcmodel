"""Truncated backpropagation through time, with a learned critic.

The plan called this SVG(K) and stage 5 declined to build it, on the evidence
that full backpropagation through the rollout was stable to thirty-two quarters
-- the neural policy's edge grew with horizon rather than degrading, which is
the opposite of the instability truncation exists to fix.

That evidence has since weakened. After the recalibration the thirty-two
quarter case is degenerate (the firm winds down in the first quarter for a
horizon-accounting reason, see docs/quant-model.md section 6), so it no longer
tests gradient stability at all. This module exists so the two approaches can
be compared directly rather than argued about.

**What truncation changes.** Full BPTT differentiates the whole episode: an
action in quarter one carries gradient through every subsequent quarter's
state. That is exact, and it is also a product of T Jacobians -- the thing that
explodes or vanishes when T is large. Truncation cuts the chain every K steps
and replaces everything past the cut with a learned estimate of what follows.
The gradient becomes biased, because the critic is wrong, and bounded, because
the chain is short.

**What the critic is.** A value function trained by regression on realized
returns from the actor's own rollouts. Not a baseline for a score-function
estimator -- there is no score function here -- but a boundary condition, which
is a different job and a more forgiving one: an inaccurate critic biases the
gradient rather than inflating its variance.

The two knobs interact in the obvious way. K = horizon recovers full BPTT
exactly and needs no critic. K = 1 is a one-step method that leans entirely on
the critic. Anything between trades exactness for a shorter chain.
"""

from dataclasses import dataclass, field

import torch

from quant.env.env import EvalResult, FirmEnv, evaluate
from quant.env.shocks import CommonRandomNumbers
from quant.env.state import FirmState, Trajectory
from quant.solvers.neural import NeuralPolicy


class Critic(torch.nn.Module):
    """Estimates what a state is worth, per unit of probability mass still
    operating when it is reached.

    Scaled by opening equity on the way out. The targets are firm values of
    order tens while the observation is order one, and a network asked to
    produce the former from the latter spends its early training learning the
    scale rather than the shape.
    """

    def __init__(self, env: FirmEnv, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        profile = env.config.profile
        self.env = env
        self.scale = env.config.firm.initial_equity
        width = env.observe(env.reset(1)).shape[-1]

        layers: list[torch.nn.Module] = []
        for _ in range(n_layers):
            layers += [torch.nn.Linear(width, hidden), torch.nn.Tanh()]
            width = hidden
        layers.append(torch.nn.Linear(width, 1))
        self.net = torch.nn.Sequential(*layers).to(
            device=profile.device, dtype=profile.dtype
        )

    def forward(self, state: FirmState) -> torch.Tensor:
        return self.scale * self.net(self.env.observe(state)).squeeze(-1)

    def __call__(self, state: FirmState) -> torch.Tensor:
        return self.forward(state)


def returns_to_go(env: FirmEnv, trajectory: Trajectory) -> list[torch.Tensor]:
    """What each state was worth, backwards along a realized trajectory.

    Per unit of probability mass entering the period, so it is directly what
    the critic is asked to predict:

        G_T = terminal(s_T)
        G_t = p_t * orderly_t
              + (1 - p_t) * [ s_t * beta * (reward_t + G_{t+1})
                              + (1 - s_t) * beta * failure_t ]

    with p_t the probability of winding down before operating and s_t the
    probability of surviving the period. Same accounting as
    Trajectory.path_values, read as a recursion rather than a sum.
    """
    discount = env.config.discount
    horizon = len(trajectory.infos)
    values = [None] * (horizon + 1)
    values[horizon] = trajectory.terminal_value

    for step in reversed(range(horizon)):
        survived = torch.exp(trajectory.log_survivals[step])
        abandon = trajectory.abandon_probs[step]
        carried = survived * discount * (trajectory.rewards[step] + values[step + 1])
        failed = (1.0 - survived) * discount * trajectory.failure_values[step]
        values[step] = abandon * trajectory.orderly_values[step] + (1.0 - abandon) * (
            carried + failed
        )
    return values


@dataclass
class SvgResult:
    policy: NeuralPolicy
    critic: Critic
    evaluation: EvalResult
    history: list[float] = field(default_factory=list)
    critic_loss: list[float] = field(default_factory=list)


def windowed_objective(
    env: FirmEnv,
    policy: NeuralPolicy,
    critic: Critic,
    crn: CommonRandomNumbers,
    window: int,
) -> torch.Tensor:
    """Firm value, with the gradient cut every `window` quarters.

    Each window is rolled forward from a *detached* state and valued by the
    critic at its far edge, except the last, which sees the real terminal
    value. The mass still operating and the accumulated path weights are
    carried between windows as numbers rather than as differentiable paths --
    that carry is precisely what truncation removes.
    """
    profile = env.config.profile
    horizon, discount = env.config.horizon, env.config.discount

    state = env.reset(crn.batch)
    active = torch.ones_like(state.equity)
    weight = torch.ones_like(state.equity)
    total = torch.zeros_like(state.equity)

    for start in range(0, horizon, window):
        steps = min(window, horizon - start)
        final = start + steps >= horizon
        segment = env.rollout_from(
            policy,
            crn,
            state.detach(),
            start=start,
            steps=steps,
            boundary=None if final else critic,
            differentiable=True,
        )

        total = total + (discount**start) * active * segment.path_values(discount)
        for step_weight in segment.weights:
            weight = weight * step_weight.detach()
        active = (active * segment.cumulative_survival()[-1]).detach()
        state = segment.states[-1]

    return profile.sum(total * weight / weight.sum())


def train_svg(
    env: FirmEnv,
    crn: CommonRandomNumbers,
    window: int = 4,
    n_steps: int = 1500,
    lr: float = 3e-3,
    critic_lr: float = 3e-3,
    critic_steps: int = 2,
    seed: int = 0,
    record_every: int = 100,
) -> SvgResult:
    """Alternate critic regression and a truncated-BPTT actor update.

    The critic is fitted to the actor's own realized returns, so it chases a
    moving target; `critic_steps` per actor step keeps it from falling behind
    without making the whole thing a critic-fitting exercise.
    """
    torch.manual_seed(seed)
    policy = NeuralPolicy(env)
    critic = Critic(env)
    actor_opt = torch.optim.Adam(policy.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    history: list[float] = []
    losses: list[float] = []

    for step in range(n_steps):
        with torch.no_grad():
            realized = env.rollout(policy, crn, differentiable=False)
            targets = returns_to_go(env, realized)

        for _ in range(critic_steps):
            critic_opt.zero_grad()
            loss = sum(
                ((critic(realized.states[t]) - targets[t]) ** 2).mean()
                for t in range(env.config.horizon)
            ) / env.config.horizon
            loss.backward()
            critic_opt.step()

        actor_opt.zero_grad()
        value = windowed_objective(env, policy, critic, crn, window)
        (-value).backward()
        actor_opt.step()

        if step % record_every == 0:
            history.append(value.item())
            losses.append(loss.item())

    return SvgResult(policy, critic, evaluate(policy, env, crn), history, losses)
