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
from quant.params import DEFAULTS, FirmParams, GrcAlphas, HazardParams

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
    abandon: float = 0.0
    profile: NumericsProfile = DEFAULT_PROFILE

    def __call__(self, state: FirmState) -> FirmAction:
        batch = state.batch()
        return FirmAction(
            grc=self.profile.tensor(list(self.grc)).expand(batch, N_FAMILIES),
            investment=self.profile.full((batch,), self.investment),
            abandon=self.profile.full((batch,), self.abandon),
        )


@dataclass(frozen=True)
class EnvConfig:
    profile: NumericsProfile = DEFAULT_PROFILE
    firm: FirmParams = field(default_factory=lambda: DEFAULTS.firm)
    alphas: GrcAlphas = field(default_factory=lambda: DEFAULTS.alphas)
    # The defaults are the *reduction*, not the intended model: one period,
    # no discounting, GRC stock equal to flow. That configuration has a closed
    # form and reproduces the published static results, so it is what the
    # regression tests run against. `EnvConfig.quarterly` is the real model.
    horizon: int = 1
    discount: float = 1.0
    grc_depreciation: float = 1.0
    financing_scale: float = 1.0
    equity_floor: float = -float("inf")
    hazard: HazardParams | None = None
    allow_abandonment: bool = False
    terminal: TerminalValue = field(default_factory=LiquidationValue)

    @classmethod
    def quarterly(cls, horizon: int, firm: FirmParams | None = None, **overrides) -> "EnvConfig":
        """The dynamic model: quarterly periods, discounted, GRC accumulating.

        Discount and depreciation come from the annual rates in FirmParams, so
        they stay consistent with each other and with `periods_per_year`.

        The insolvency barrier is on by default here, and it is not decoration.
        The convex financing cost is unbounded below: the premium is
        K(e/K)^gamma with e = I - wealth, so once equity goes negative the
        shortfall grows, the premium grows faster, and equity squares each
        period. Measured on this model it reaches -2e7 by quarter three and
        overflows float64 by quarter eight. One period cannot compound, which
        is why the static model never showed it.

        A real firm does not run to minus ten-to-the-three-hundred; it fails.
        So the barrier is what makes the multi-period problem well posed at
        all, and it is the first place survival does real work in this model --
        an argument for the stage 4 framing that arrived from the arithmetic
        rather than from the economics.
        """
        firm = firm or DEFAULTS.firm
        settings = dict(
            firm=firm,
            horizon=horizon,
            discount=firm.discount(),
            grc_depreciation=firm.grc_depreciation(),
            # Not an economic barrier any more -- the hazard is. This sits far
            # enough below zero that survival there is already exactly zero, so
            # freezing the path costs no gradient; it exists only to stop the
            # unbounded financing premium from overflowing float64.
            equity_floor=-5.0 * firm.initial_equity,
            hazard=DEFAULTS.hazard,
            allow_abandonment=True,
        )
        settings.update(overrides)  # an explicit override wins over the derived rate
        return cls(**settings)


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
            grc_depreciation=config.grc_depreciation,
            hazard=config.hazard,
            allow_abandonment=config.allow_abandonment,
        )

    def reset(self, batch: int) -> FirmState:
        profile = self.config.profile
        return FirmState(
            equity=profile.full((batch,), self.config.firm.initial_equity),
            grc_stock=profile.full((batch, N_FAMILIES), self.config.firm.initial_grc_stock),
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
        """What a surviving firm is worth at the horizon.

        Only the surviving value: paths that failed are accounted for by the
        survival weighting in Trajectory.path_values, which collects
        `failure_value` at the date of failure rather than at the horizon. The
        mask is a numerical guard -- a path frozen at the barrier can hold a
        large negative equity, and multiplying it by a zero survival weight is
        cleaner than relying on the arithmetic.
        """
        value = self.config.terminal(state)
        return torch.where(state.alive, value, torch.zeros_like(value))

    def rollout(
        self, policy: Policy, crn: CommonRandomNumbers, differentiable: bool = True
    ) -> Trajectory:
        state = self.reset(crn.batch)
        states, rewards, weights, infos = [state], [], [], []
        survivals, recoveries, abandons, orderlies = [], [], [], []

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
                survivals.append(result.log_survival)
                recoveries.append(result.failure_value)
                abandons.append(result.abandon_prob)
                orderlies.append(result.orderly_value)
                infos.append(result.info)

            terminal = self.terminal_value(state)

        return Trajectory(
            states=states,
            rewards=rewards,
            weights=weights,
            log_survivals=survivals,
            failure_values=recoveries,
            abandon_probs=abandons,
            orderly_values=orderlies,
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
    """What a policy is worth, plus the diagnostics that say whether to believe it.

    `grc` is the per-period GRC *flow*, averaged over periods and paths -- the
    spend an executive would budget. At a one-period horizon with no
    depreciation it coincides with the stock, which is why it matches the
    static model's reported budgets.
    """

    value: float
    grc: tuple[float, float, float]
    grc_stock: tuple[float, float, float]
    constrained_fraction: float
    survival_rate: float
    # Probability the firm chooses to wind down at some point over the horizon,
    # as opposed to failing or reaching the end still operating.
    orderly_exit_rate: float
    effective_sample_size: float
    # (V - Lambda) / V at the opening state: the share of firm value that is
    # going-concern rather than liquidation. If this is near zero, death is
    # costless and there is no survival motive left to model -- the objective
    # has quietly become expected-loss minimization again.
    going_concern_share: float

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
        horizon = len(trajectory.infos)
        per_path = weights.unsqueeze(-1)

        # Diagnostics count only paths that were alive when the action was
        # taken. A dead path still has a policy evaluated on its frozen state
        # and that action still lands in `info`, but it changes nothing and
        # averaging it in makes the reported budget meaningless -- it reads as
        # whatever the policy happens to output in a region it is never graded
        # on.
        survival = trajectory.cumulative_survival()
        live = [survival[step] for step in range(len(trajectory.infos))]
        live_mass = sum(profile.sum(mask * weights) for mask in live)

        constrained = sum(
            profile.sum((info["external"] > negligible).to(profile.dtype) * mask * weights)
            for info, mask in zip(trajectory.infos, live)
        ) / live_mass
        flow = sum(
            profile.sum(info["grc_flow"] * (mask * weights).unsqueeze(-1), dim=0)
            for info, mask in zip(trajectory.infos, live)
        ) / live_mass
        # Survival-weighted, for the same reason the flow is: a failed path's
        # state is frozen and the policy is still evaluated on it, so its
        # action is whatever the network happens to emit in a region it is
        # never graded on. Unweighted, that read as an end stock of 9.8 in a
        # configuration where the stock cannot exceed one quarter's spend.
        survives = profile.sum(survival[-1] * weights)
        exited = sum(
            profile.sum(survival[step] * trajectory.abandon_probs[step] * weights)
            for step in range(len(trajectory.infos))
        )
        opening = trajectory.states[0]
        liquidation = env.config.firm.failure_recovery * torch.clamp(opening.equity, min=0.0)
        liquidation_value = profile.sum(liquidation * weights)
        final = (survival[-1] * weights).unsqueeze(-1)
        stock = profile.sum(trajectory.states[-1].grc_stock * final, dim=0) / survives

    return EvalResult(
        value=value.item(),
        grc=(flow[0].item(), flow[1].item(), flow[2].item()),
        grc_stock=(stock[0].item(), stock[1].item(), stock[2].item()),
        constrained_fraction=constrained.item(),
        survival_rate=survives.item(),
        orderly_exit_rate=exited.item(),
        effective_sample_size=trajectory.effective_sample_size(),
        going_concern_share=(1.0 - liquidation_value / value).item(),
    )
