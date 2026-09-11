"""The environment, and the one function that scores a policy against it.

Two things here carry most of the design weight.

**Non-anticipativity is a property of the policy, not of the dynamics.**
`rollout` hands the policy the current state and nothing else, then draws the
shock. A policy that sees only `state` therefore cannot condition on what has
not happened yet, whatever the solver does internally. This is what lets one
environment serve both an honest state-feedback policy and the deliberately
clairvoyant perfect-information bound in quant/solvers/pathwise.py: the bound
cheats by holding a free parameter per path, and the cheat is visible in the
policy's type rather than buried in the transition. tests/test_env.py asserts
the property directly by perturbing the future of the random stream.

**Every policy is scored by `evaluate`.** The commonest way a solver
comparison goes wrong is the baseline and the learner being measured by
different code, so there is exactly one scoring path and all of them use it.
"""

import contextlib
from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch

from quant.env.actions import FirmAction
from quant.env.reward import LiquidationValue, TerminalValue
from quant.env.shocks import CommonRandomNumbers, Shock
from quant.env.state import N_FAMILIES, FirmState, Trajectory
from quant.numerics import DEFAULT_PROFILE, NumericsProfile
from quant.params import DEFAULTS, FirmParams, GrcAlphas

Sampler = Callable[[dict[str, torch.Tensor], NumericsProfile], Shock]


class Policy(Protocol):
    """State in, action out. The only interface a solver has to satisfy."""

    def __call__(self, state: FirmState) -> FirmAction: ...


@dataclass(frozen=True)
class ConstantPolicy:
    """The same action every period, regardless of state.

    The canonical trivial Policy, and a genuine baseline: a learner that
    cannot beat a well-chosen constant has not learned anything about the
    state, only about the average.
    """

    grc: tuple[float, float, float]
    investment: float
    profile: NumericsProfile = DEFAULT_PROFILE

    def __call__(self, state: FirmState) -> FirmAction:
        batch = state.batch()
        return FirmAction(
            grc=self.profile.tensor(list(self.grc)).expand(batch, N_FAMILIES),
            investment=self.profile.full((batch,), self.investment),
        )


@dataclass(frozen=True)
class EnvConfig:
    profile: NumericsProfile = DEFAULT_PROFILE
    firm: FirmParams = field(default_factory=lambda: DEFAULTS.firm)
    alphas: GrcAlphas = field(default_factory=lambda: DEFAULTS.alphas)
    horizon: int = 1
    discount: float = 1.0
    financing_scale: float = 1.0
    equity_floor: float = -float("inf")
    terminal: TerminalValue = field(default_factory=LiquidationValue)
    # What a path that crossed the barrier is worth. Distinct from `terminal`
    # because a firm that died does not choose how it is valued.
    failure_value: float = 0.0


class FirmEnv:
    def __init__(self, config: EnvConfig, sampler: Sampler, dynamics=None):
        from quant.env.dynamics import StandardDynamics

        self.config = config
        self.sampler = sampler
        self.dynamics = dynamics or StandardDynamics(
            firm=config.firm,
            alphas=config.alphas,
            profile=config.profile,
            financing_scale=config.financing_scale,
            equity_floor=config.equity_floor,
        )

    def reset(self, batch: int) -> FirmState:
        profile = self.config.profile
        return FirmState(
            equity=profile.full((batch,), self.config.firm.initial_equity),
            grc_stock=profile.full((batch, N_FAMILIES), 0.0),
            alive=torch.ones(batch, dtype=torch.bool, device=profile.device),
            t=0,
        )

    def observe(self, state: FirmState) -> torch.Tensor:
        """Normalized features, (B, 4). The only place state is rescaled.

        Divided by initial equity so a learner sees O(1) inputs and so the
        observation is invariant to the currency the firm is denominated in --
        the same requirement the cost functions carry
        (docs/static-model-debug-notes.md section 4).
        """
        scale = self.config.firm.initial_equity
        return torch.cat(
            [(state.equity / scale).unsqueeze(-1), state.grc_stock / scale], dim=-1
        )

    def terminal_value(self, state: FirmState) -> torch.Tensor:
        alive_value = self.config.terminal(state)
        dead_value = torch.full_like(alive_value, self.config.failure_value)
        return torch.where(state.alive, alive_value, dead_value)

    def rollout(
        self, policy: Policy, crn: CommonRandomNumbers, differentiable: bool = True
    ) -> Trajectory:
        state = self.reset(crn.batch)
        states, rewards, weights, infos = [state], [], [], []

        context = contextlib.nullcontext() if differentiable else torch.no_grad()
        with context:
            for t in range(self.config.horizon):
                action = policy(state)  # state only -- see the module docstring
                shock = self.sampler(crn.at(t), self.config.profile)
                result = self.dynamics.step(state, action, shock)
                state = result.state
                states.append(state)
                rewards.append(result.reward)
                weights.append(result.weight)
                infos.append(result.info)

            terminal = self.terminal_value(state)

        return Trajectory(
            states=states,
            rewards=rewards,
            weights=weights,
            terminal_value=terminal,
            infos=infos,
        )

    def value_of(self, trajectory: Trajectory) -> torch.Tensor:
        """Probability-weighted expected discounted return. Differentiable."""
        weights = trajectory.path_weights()
        return self.config.profile.sum(
            trajectory.path_values(self.config.discount) * weights
        )


@dataclass
class EvalResult:
    """What a policy is worth, plus the diagnostics that say whether to believe it."""

    value: float
    grc: tuple[float, float, float]
    constrained_fraction: float
    survival_rate: float
    effective_sample_size: float

    @property
    def total_grc(self) -> float:
        return sum(self.grc)


def evaluate(policy: Policy, env: FirmEnv, crn: CommonRandomNumbers) -> EvalResult:
    """Score a policy. The single measurement path for every solver."""
    profile = env.config.profile
    with torch.no_grad():
        trajectory = env.rollout(policy, crn, differentiable=False)
        weights = trajectory.path_weights()
        value = env.value_of(trajectory)

        # "Is the firm actually raising external finance" needs a money-valued
        # threshold: `external` comes from softplus so it is never exactly
        # zero, and a fixed absolute epsilon is float64-scaled -- under float32
        # it sits at the noise floor and every path reads as constrained.
        negligible = env.config.firm.distress_reference * torch.finfo(profile.dtype).eps ** 0.5
        constrained = profile.sum(
            (trajectory.infos[0]["external"] > negligible).to(profile.dtype) * weights
        )
        grc = profile.sum(trajectory.states[1].grc_stock * weights.unsqueeze(-1), dim=0)
        survival = profile.sum(trajectory.states[-1].alive.to(profile.dtype) * weights)

    return EvalResult(
        value=value.item(),
        grc=(grc[0].item(), grc[1].item(), grc[2].item()),
        constrained_fraction=constrained.item(),
        survival_rate=survival.item(),
        effective_sample_size=trajectory.effective_sample_size(),
    )
