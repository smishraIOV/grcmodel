"""The firm's controls, and the single place raw parameters become them.

Every solver -- the pathwise optimizer, a grid's inner maximization, a neural
policy -- produces unconstrained reals and has to map them onto a non-negative
spend and investment. That map is defined once, here. Two solvers squashing
differently would make their values incomparable while still looking like they
disagreed about the economics.

Always `softplus`, never `torch.clamp`: clamp's gradient is exactly zero at the
boundary, so a control initialized at zero -- the natural start for a spend
level -- never receives a signal to move off it
(docs/static-model-debug-notes.md section 1).
"""

from dataclasses import dataclass

import torch

from quant.env.state import N_FAMILIES

N_RAW = N_FAMILIES + 1  # three GRC budgets plus investment


@dataclass(frozen=True)
class FirmAction:
    """Per-path controls for one period.

    Both fields carry a batch dimension even when a solver holds the same
    budget across every path. Whether a control may vary by path is a
    statement about what the *policy* is allowed to see, not about the
    dynamics -- see quant/env/env.py on non-anticipativity.
    """

    grc: torch.Tensor         # (B, 3) spend by family
    investment: torch.Tensor  # (B,)

    def total_grc(self) -> torch.Tensor:
        return self.grc.sum(dim=-1)


class ActionSpec:
    """Raw reals -> FirmAction. The one squashing map."""

    n_raw = N_RAW

    @staticmethod
    def from_raw(raw: torch.Tensor) -> FirmAction:
        """`raw` is (B, 4): three budgets then investment."""
        positive = torch.nn.functional.softplus(raw)
        return FirmAction(grc=positive[..., :N_FAMILIES], investment=positive[..., N_FAMILIES])
