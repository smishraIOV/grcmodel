"""Firm state and the per-step record of what happened to it.

Batch-first throughout: every field carries a leading path dimension and `t`
is a plain int, because all paths step in lockstep. Two reasons, and both
bite later rather than now -- static shapes keep MPS from recompiling per
step, and a shared time index is what lets one pre-drawn random tensor be
replayed identically across solvers (quant/env/shocks.py).

Death is represented as a mask, never as removal from the batch. A dead path
keeps stepping with its state frozen and contributes zero reward. Compacting
the batch instead would make shapes dynamic and silently desynchronise every
path's position in the random stream.
"""

from dataclasses import dataclass, replace

import torch

N_FAMILIES = 3  # credit, operational, compliance -- docs/framework.md section 1


@dataclass(frozen=True)
class FirmState:
    """(equity, GRC stock, alive) for a batch of paths at one point in time.

    `grc_stock` is carried now but not yet depreciated or accumulated: at a
    one-period horizon a stock and a flow are the same thing. It becomes a
    real state variable in stage 2, which is what makes the dynamic problem
    more than the static one repeated.
    """

    equity: torch.Tensor      # (B,)
    grc_stock: torch.Tensor   # (B, 3)
    alive: torch.Tensor       # (B,) bool
    t: int = 0

    def batch(self) -> int:
        return self.equity.shape[0]

    def advance(self, **changes) -> "FirmState":
        """Next state, with `t` incremented and anything unnamed carried over."""
        return replace(self, t=self.t + 1, **changes)

    def freeze_dead(self, previous: "FirmState") -> "FirmState":
        """Hold every already-dead path at the value it died with.

        Applied after the transition, so the dynamics never has to reason
        about liveness: it computes the update for every path and this
        discards it where it does not apply.
        """
        alive = previous.alive
        return replace(
            self,
            equity=torch.where(alive, self.equity, previous.equity),
            grc_stock=torch.where(alive.unsqueeze(-1), self.grc_stock, previous.grc_stock),
            alive=self.alive & alive,
        )


@dataclass(frozen=True)
class StepResult:
    """One period's outcome.

    `weight` is the per-step probability weight a path carries, which is how
    the compliance channel acts: a breach is discrete, so its budget cannot
    shrink a drawn indicator and instead reweights the path
    (docs/static-model-debug-notes.md section 5). It is per-step and must be
    multiplied along a trajectory, which is exactly why it does not survive a
    long horizon -- see Trajectory.effective_sample_size.
    """

    state: FirmState
    reward: torch.Tensor       # (B,)
    weight: torch.Tensor       # (B,)
    terminated: torch.Tensor   # (B,) bool, died during THIS step
    info: dict


@dataclass
class Trajectory:
    """A rollout, kept unreduced so any solver can take its own expectation."""

    states: list[FirmState]
    rewards: list[torch.Tensor]
    weights: list[torch.Tensor]
    terminal_value: torch.Tensor
    infos: list[dict]

    def path_weights(self) -> torch.Tensor:
        """Normalized probability weight per path, compounded over the horizon."""
        weight = self.weights[0]
        for step_weight in self.weights[1:]:
            weight = weight * step_weight
        return weight / weight.sum()

    def path_values(self, discount: float) -> torch.Tensor:
        """Discounted return per path, terminal value included."""
        total = torch.zeros_like(self.terminal_value)
        for step, reward in enumerate(self.rewards):
            total = total + (discount**step) * reward
        return total + (discount ** len(self.rewards)) * self.terminal_value

    def effective_sample_size(self) -> float:
        """Kish ESS of the compounded weights, as a fraction of the batch.

        The diagnostic that says when likelihood-ratio reweighting has stopped
        working. Per-step ratios multiply, so their variance compounds and ESS
        decays geometrically in the horizon; at that point the compliance
        channel has to move into the dynamics as an event intensity rather
        than stay a reweighting.
        """
        weight = self.path_weights()
        return (1.0 / (weight**2).sum()).item() / weight.numel()
