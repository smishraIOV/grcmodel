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

N_RAW = N_FAMILIES + 2  # three GRC budgets, investment, and the exit decision

# Raw initialization for the exit decision. Zero would mean sigmoid(0) = 0.5 --
# a firm that starts out planning to wind down with even odds every quarter,
# which is a terrible place to begin a search and slow to climb out of. -4
# gives about 1.8%: the option is present and has a gradient, but the firm
# starts out intending to stay in business.
ABANDON_INIT = -4.0


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
    abandon: torch.Tensor     # (B,) in [0, 1], the probability of exiting now

    def total_grc(self) -> torch.Tensor:
        return self.grc.sum(dim=-1)


class ActionSpec:
    """Raw reals -> FirmAction. The one squashing map."""

    n_raw = N_RAW

    @staticmethod
    def from_raw(raw: torch.Tensor) -> FirmAction:
        """`raw` is (B, 5): three budgets, investment, then the exit decision.

        Spend and investment go through softplus, as everywhere. The exit
        decision goes through a sigmoid and is treated as a *probability* of
        winding down rather than a hard choice.

        That relaxation costs nothing, which is worth stating because relaxing
        a binary decision usually does. Firm value is **linear** in this
        probability -- it is a convex combination of exiting now and carrying
        on -- and a linear function on [0, 1] attains its maximum at an
        endpoint. So the optimizer drives it to 0 or 1 on its own and the
        relaxed optimum equals the discrete one. What it buys is a gradient
        everywhere in between, which an argmax would not have.
        """
        positive = torch.nn.functional.softplus(raw[..., : N_FAMILIES + 1])
        return FirmAction(
            grc=positive[..., :N_FAMILIES],
            investment=positive[..., N_FAMILIES],
            abandon=torch.sigmoid(raw[..., N_FAMILIES + 1]),
        )
