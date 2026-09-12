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
    """Equity, plus the value of the business that is still there.

    The franchise is **added**, not multiplied. That distinction is the whole
    of this class and getting it wrong broke three separate channels.

    Written as `(1 + m) * equity`, a unit of equity retained to the horizon is
    worth `(1 + m)` while a unit distributed today is worth 1 -- a pure
    arbitrage the firm will always take. At m = 1.0 that drove the payout
    share to zero, dominated the wind-down option (why accept 0.7 x equity now
    when waiting pays 2 x equity?), and left the objective so dominated by
    terminal equity that every other control's gradient was noise beside it.

    Additive, retaining a unit yields exactly one unit at the horizon,
    discounted -- so distributing wins on time value and has to be argued
    against on the merits: capital funds investment the firm cannot otherwise
    fund, and lowers the hazard. That is a real trade-off rather than a free
    lunch.

    The franchise is a going concern's value beyond its book, collected only by
    the probability mass still trading at the horizon, so death destroys it.
    Not proportional to equity, because the business is worth what the
    opportunity is worth, not what cash the firm happens to be holding on the
    last day.
    """

    franchise: float = 0.0
    recovery: float = 1.0

    def __call__(self, state: FirmState) -> torch.Tensor:
        return self.recovery * state.equity + self.franchise
