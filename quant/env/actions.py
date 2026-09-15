"""The firm's controls, and the single place raw parameters become them.

Every solver -- the pathwise optimizer, a grid's inner maximization, a neural
policy -- produces unconstrained reals and has to map them onto a non-negative
spend and investment. That map is defined once, here. Two solvers squashing
differently would make their values incomparable while still looking like they
disagreed about the economics.

Always `softplus`, never `torch.clamp`: clamp's gradient is exactly zero at the
boundary, so a control initialized at zero -- the natural start for a spend
level -- never receives a signal to move off it
(docs/quant-model.md appendix A1).
"""

import math
from dataclasses import dataclass

import torch

from quant.env.state import N_FAMILIES

N_RAW = N_FAMILIES + 3  # GRC budgets, investment, the exit decision, the payout

# Raw initialization for the exit decision, as a *cumulative* probability of
# winding down at some point over the horizon rather than a per-quarter one.
#
# This was a flat -4.0, about 1.8% a quarter. That is 13% cumulative over eight
# quarters and 44% over thirty-two, so at longer horizons the firm began its
# search already halfway out the door -- and exit is absorbing, so once the
# mass has left there is no continuing business to generate a gradient for
# staying. Measured, a firm allowed to exit converged to 11.20 at
# twenty-four quarters while the same firm forbidden to exit reached 19.68:
# not an economic judgement, an optimizer trapped in an absorbing action.
#
# Holding the *cumulative* figure fixed instead makes the initialization mean
# the same thing at every horizon.
# 0.1%, not 2%. Measured at thirty-two quarters, a 2% cumulative start still
# collapsed into the absorbing exit (value 11.20 against a no-exit reference of
# 17.90) and more training did not rescue it -- 4000 steps landed in the same
# place, so it is a basin of attraction rather than a budget. At 0.1% the same
# firm converges to 17.90 with an exit rate of exactly zero.
#
# The option remains discoverable: a firm with nothing left to protect still
# finds and takes it (tests/test_env.py). Starting nearly out of the door is
# what prevents the search, not starting nearly in it.
ABANDON_INIT_CUMULATIVE = 0.001


def abandon_init(horizon: int, cumulative: float = ABANDON_INIT_CUMULATIVE) -> float:
    """Raw value whose sigmoid gives `cumulative` exit probability over `horizon`."""
    per_period = 1.0 - (1.0 - cumulative) ** (1.0 / max(horizon, 1))
    return math.log(per_period / (1.0 - per_period))

# Payout starts low for the same reason: sigmoid(-2) is about 12%, so the firm
# begins retaining most of what it earns and has to discover that distributing
# is worth more. Starting at half would have it giving away capital it needs to
# fund investment before it has learned that funding is scarce.
PAYOUT_INIT = -2.0


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
    payout: torch.Tensor      # (B,) in [0, 1], share of the quarter's equity distributed

    def total_grc(self) -> torch.Tensor:
        return self.grc.sum(dim=-1)


class ActionSpec:
    """Raw reals -> FirmAction. The one squashing map."""

    n_raw = N_RAW

    @staticmethod
    def from_raw(raw: torch.Tensor) -> FirmAction:
        """`raw` is (B, 6): three budgets, investment, the exit decision, the payout.

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
            payout=torch.sigmoid(raw[..., N_FAMILIES + 2]),
        )
