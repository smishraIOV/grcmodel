"""What the firm is worth when the rollout stops.

A strategy object rather than a constant, because the three cases the project
needs are genuinely different and the choice is a modelling decision rather
than an implementation detail:

- `Zero` -- the horizon is the end of the world. Used by the degenerate
  checks, where a known closed form has to be reproduced with nothing after it.
- `LiquidationValue` -- the firm is wound up and equity distributed. This is
  what makes a one-period rollout reproduce the static model exactly.
- `PerpetuityValue` -- a going concern with a continuing franchise. This is
  the one that carries the tail, and it is what keeps the horizon short enough
  for pathwise gradients and a grid solver to both stay usable.

`failure_value` is separate from all three: a firm that died does not get to
choose how it is valued. It is the seam where an orderly wind-down recovery
(worth more than a disorderly failure) enters in stage 4.
"""

from dataclasses import dataclass
from typing import Protocol

import torch

from quant.env.state import FirmState


class TerminalValue(Protocol):
    def __call__(self, state: FirmState) -> torch.Tensor: ...


@dataclass(frozen=True)
class Zero:
    def __call__(self, state: FirmState) -> torch.Tensor:
        return torch.zeros_like(state.equity)


@dataclass(frozen=True)
class LiquidationValue:
    """Equity, net of a proportional cost of winding up."""

    recovery: float = 1.0

    def __call__(self, state: FirmState) -> torch.Tensor:
        return self.recovery * state.equity


@dataclass(frozen=True)
class PerpetuityValue:
    """Equity plus a multiple of it, standing in for the continuing franchise.

    A placeholder shape, not a calibrated one: the franchise should grow with
    the deposit base rather than with equity, which is why deposits become
    state in a later stage. Kept explicit so the assumption is visible.
    """

    multiple: float = 0.0
    recovery: float = 1.0

    def __call__(self, state: FirmState) -> torch.Tensor:
        return self.recovery * (1.0 + self.multiple) * state.equity
