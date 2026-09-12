"""Risk realizations, and the pre-drawn randomness behind them.

Two samplers, because the project needs two different things from a shock:

- `MonteCarloSampler` draws continuous losses and Bernoulli events, the
  realistic case, with paths equally likely.
- `FourStateSampler` enumerates business good/bad crossed with breach/no
  breach and carries explicit probabilities. Small enough that expected
  exposures are exact rather than estimated, which is what lets the
  closed-form benchmark in quant/solvers/analytic.py be asserted to 1e-6
  instead of to sampling error.

Randomness is pre-drawn, not drawn on demand. `CommonRandomNumbers` holds the
whole (T, B) uniform tensor per named channel, so every solver compared
against another sees the identical scenario set and the comparison is paired
rather than a noisy A/B. It is also what makes the no-lookahead test possible:
perturb the future slice and assert the past did not move.

Channels are seeded independently by name, so adding a channel later does not
shift the draws of the ones already there and invalidate every recorded
number.
"""

import zlib
from dataclasses import dataclass

import torch

from quant.numerics import NumericsProfile
from quant.params import SamplerParams, ShockParams

CHANNELS = (
    "credit",
    "op_occurs",
    "op_severity",
    "compliance_occurs",
    # Added after the others. Because channels are seeded independently by
    # name, adding them left every existing draw bit-identical -- which is the
    # reason that design was chosen rather than one shared stream.
    "cliff_occurs",
    "cliff_severity",
)


@dataclass(frozen=True)
class Shock:
    """One period's risk realization for a batch of paths.

    Each family is represented by the moment its GRC budget actually moves
    (docs/framework.md section 1): credit a continuous loss whose MEAN falls,
    operational an indicator times a SEVERITY that falls, compliance an
    indicator whose PROBABILITY falls.
    """

    credit_loss: torch.Tensor
    op_occurs: torch.Tensor
    op_severity: torch.Tensor
    compliance_occurs: torch.Tensor
    compliance_severity: torch.Tensor
    base_weight: torch.Tensor          # (B,) path probability before GRC shifts it
    compliance_base_probability: float
    # The cliff channel carries its *raw uniform*, not a drawn indicator. Its
    # occurrence probability depends on the GRC stock, which the sampler does
    # not know and must not: the threshold is applied in the dynamics, where
    # the relaxation that makes it differentiable can see the control.
    cliff_uniform: torch.Tensor
    # Severity is GRC-independent, so it is drawn here like any other shock.
    cliff_fraction: torch.Tensor


class CommonRandomNumbers:
    """The full (T, B) uniform draw per channel, held on the RNG device.

    Uniforms stay at the profile's `rng_dtype` on CPU: samplers transform them
    at full precision and cast last, per the rule in quant/numerics.py.
    """

    def __init__(
        self,
        seed: int,
        horizon: int,
        batch: int,
        profile: NumericsProfile,
        channels: tuple[str, ...] = CHANNELS,
    ):
        self.seed, self.horizon, self.batch = seed, horizon, batch
        self.uniforms = {
            channel: profile.draw_uniform(
                (horizon, batch), profile.generator(_channel_seed(seed, channel))
            )
            for channel in channels
        }

    def at(self, t: int) -> dict[str, torch.Tensor]:
        return {channel: draw[t] for channel, draw in self.uniforms.items()}


def _channel_seed(seed: int, channel: str) -> int:
    """Stable across processes -- Python's str hash is per-process randomized."""
    return (seed * 1_000_003 + zlib.crc32(channel.encode())) % (2**31 - 1)


def cliff_fraction(
    uniform: torch.Tensor, profile: NumericsProfile, mean: float = 0.35
) -> torch.Tensor:
    """Share of opening equity a cliff event destroys, given it happens.

    Exponential with the given mean, capped at one: mostly severe, sometimes
    total, and heavy enough in the middle that a firm often survives badly
    impaired rather than cleanly dying. That middle is the point of the
    channel.
    """
    return profile.to(torch.clamp(-mean * torch.log1p(-uniform), max=1.0))


@dataclass(frozen=True)
class MonteCarloSampler:
    """Exponential losses by inverse CDF, Bernoulli events by thresholding."""

    params: SamplerParams

    def __call__(self, uniforms: dict[str, torch.Tensor], profile: NumericsProfile) -> Shock:
        def exponential(uniform: torch.Tensor, mean: float) -> torch.Tensor:
            return profile.to(-mean * torch.log1p(-uniform))

        def bernoulli(uniform: torch.Tensor, probability: float) -> torch.Tensor:
            return profile.to((uniform < probability).to(uniform.dtype))

        batch = uniforms["credit"].shape[0]
        return Shock(
            credit_loss=exponential(uniforms["credit"], self.params.credit_loss_mean),
            op_occurs=bernoulli(uniforms["op_occurs"], self.params.op_probability),
            op_severity=exponential(uniforms["op_severity"], self.params.op_severity_mean),
            compliance_occurs=bernoulli(
                uniforms["compliance_occurs"], self.params.compliance_probability
            ),
            compliance_severity=profile.full((batch,), self.params.compliance_severity),
            base_weight=profile.full((batch,), 1.0 / batch),
            compliance_base_probability=self.params.compliance_probability,
            cliff_uniform=profile.to(uniforms["cliff_occurs"]),
            cliff_fraction=cliff_fraction(uniforms["cliff_severity"], profile),
        )


@dataclass(frozen=True)
class FourStateSampler:
    """Business good/bad crossed with breach/no breach, explicitly weighted.

    Ignores the uniforms: every state is enumerated and carries its own
    probability, so the expectation is exact. `credit_spread` widens the gap
    between the good and bad credit outcome at constant mean, which is what
    makes sweeping it a mean-preserving spread.
    """

    params: ShockParams
    batch = 4

    def __call__(self, uniforms: dict[str, torch.Tensor], profile: NumericsProfile) -> Shock:
        p = self.params
        business_weight = profile.tensor([0.5, 0.5])
        credit = profile.tensor([p.credit_mean - p.credit_spread, p.credit_mean + p.credit_spread])
        op_occurs = profile.tensor([0.0, 1.0])
        op_severity = profile.tensor([0.0, p.op_severity])
        breach_weight = profile.tensor([1.0 - p.compliance_probability, p.compliance_probability])
        breach = profile.tensor([0.0, 1.0])

        return Shock(
            credit_loss=credit.repeat_interleave(2),
            op_occurs=op_occurs.repeat_interleave(2),
            op_severity=op_severity.repeat_interleave(2),
            compliance_occurs=breach.repeat(2),
            compliance_severity=profile.full((4,), p.compliance_severity),
            base_weight=business_weight.repeat_interleave(2) * breach_weight.repeat(2),
            compliance_base_probability=p.compliance_probability,
            # No cliff in the enumerated world: it exists to keep the
            # closed-form benchmark exactly solvable, and a heavy-tailed jump
            # is the opposite of that. A uniform of 1.0 never fires.
            cliff_uniform=profile.full((4,), 1.0),
            cliff_fraction=profile.full((4,), 0.0),
        )
