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
    """(equity, deposits, GRC stock, alive) for a batch of paths at one point
    in time.

    `grc_stock` is carried now but not yet depreciated or accumulated: at a
    one-period horizon a stock and a flow are the same thing. It becomes a
    real state variable in stage 2, which is what makes the dynamic problem
    more than the static one repeated.

    `deposits` is `None` in every configuration without a liability side, which
    is all of the reduced ones: the closed-form benchmark, the static
    regression, and the grid solver's restriction. That is a sentinel rather
    than a zero tensor on purpose. Zero deposits is a firm that *has* a
    liability side and happens to be funding nothing through it, and the two
    should not be spelled the same way -- the arithmetic agrees but the
    accounting identity the tests assert does not, because a firm with no
    liability side pays no interest and has no capacity beyond its own equity.
    Same convention as `hazard: HazardParams | None` in the dynamics: a channel
    that is off says so.
    """

    equity: torch.Tensor      # (B,)
    grc_stock: torch.Tensor   # (B, 3)
    alive: torch.Tensor       # (B,) bool
    deposits: torch.Tensor | None = None  # (B,), None when there is no liability side
    t: int = 0

    def assets(self) -> torch.Tensor:
        """Equity plus deposits: what the firm has to deploy before it spends.

        The balance-sheet identity, and the one place the liability side is
        allowed to be implicit about being switched off.
        """
        return self.equity if self.deposits is None else self.equity + self.deposits

    def batch(self) -> int:
        return self.equity.shape[0]

    def advance(self, **changes) -> "FirmState":
        """Next state, with `t` incremented and anything unnamed carried over."""
        return replace(self, t=self.t + 1, **changes)

    def detach(self) -> "FirmState":
        """The same state with the gradient cut.

        Where a truncated-BPTT window begins: the learner carries the state
        forward as a *value* but not as a path the gradient can flow back
        along (`quant/solvers/svg.py`, on the `svg-critic` branch).
        """
        return replace(
            self,
            equity=self.equity.detach(),
            grc_stock=self.grc_stock.detach(),
            alive=self.alive.detach(),
            deposits=None if self.deposits is None else self.deposits.detach(),
        )

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
            deposits=(
                None
                if self.deposits is None
                else torch.where(alive, self.deposits, previous.deposits)
            ),
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
    reward: torch.Tensor        # (B,)
    weight: torch.Tensor        # (B,)
    log_survival: torch.Tensor  # (B,) log P(survive this period), <= 0
    failure_value: torch.Tensor # (B,) what is recovered if it fails this period
    abandon_prob: torch.Tensor  # (B,) P(wind down voluntarily, before operating)
    orderly_value: torch.Tensor # (B,) what an orderly wind-down recovers
    terminated: torch.Tensor    # (B,) bool, crossed the hard barrier THIS step
    info: dict


@dataclass
class Trajectory:
    """A rollout, kept unreduced so any solver can take its own expectation."""

    states: list[FirmState]
    rewards: list[torch.Tensor]
    weights: list[torch.Tensor]
    base_weight: torch.Tensor
    log_survivals: list[torch.Tensor]
    failure_values: list[torch.Tensor]
    abandon_probs: list[torch.Tensor]
    orderly_values: list[torch.Tensor]
    terminal_value: torch.Tensor
    infos: list[dict]

    def cumulative_survival(self) -> list[torch.Tensor]:
        """[A_0 = 1, A_1, ..., A_T]: the probability a path is still operating.

        Two ways to stop, and they are different events. The firm may wind down
        voluntarily at the start of a period, before it operates, and collect
        the orderly recovery; or it may operate and fail, and collect the
        disorderly one. Both remove mass from the continuing business:

            A_{t+1} = A_t * (1 - p_t) * s_t

        Accumulated in logs and exponentiated once. A product of per-period
        probabilities underflows float32 well inside a horizon this model cares
        about, and the sum does not.
        """
        running = torch.zeros_like(self.log_survivals[0])
        active = [torch.exp(running)]
        for survived, abandon in zip(self.log_survivals, self.abandon_probs):
            # clamped off 1.0 so a path that has certainly exited gives -inf
            # rather than nan
            stays = torch.log1p(-abandon.clamp(max=1.0 - 1e-12))
            running = running + stays + survived
            active.append(torch.exp(running))
        return active

    def path_weights(self) -> torch.Tensor:
        """Normalized probability weight per path, compounded over the horizon.

        The prior is applied once and the per-step likelihood ratios are
        accumulated in **log space**, for the same reason survival is: a
        product of T factors underflows long before its logarithm does. Done
        multiplicatively with the prior folded in per step, this reached 1e-344
        at 104 steps and normalized to 0/0.

        The shift by the maximum before exponentiating is the standard
        log-sum-exp guard: it cannot change the normalized result and it keeps
        the largest weight at exactly 1.
        """
        log_weight = torch.log(self.base_weight)
        for ratio in self.weights:
            log_weight = log_weight + torch.log(ratio)
        shifted = torch.exp(log_weight - log_weight.max())
        return shifted / shifted.sum()

    def path_values(self, discount: float) -> torch.Tensor:
        """Discounted return per path, survival-weighted.

        Death is not sampled. Each path carries the *probability* it is still
        alive, so value is an expectation over survival rather than an average
        over drawn failures: the surviving mass earns the rewards and the
        terminal value, and the mass that failed during a period collects the
        failure value at that date.

        That is what makes the objective differentiable in everything that
        drives the hazard. A sampled death is a step function of equity and
        carries no gradient; a survival probability carries one everywhere.
        """
        active = self.cumulative_survival()
        total = torch.zeros_like(self.terminal_value)
        for step, reward in enumerate(self.rewards):
            entering = active[step]
            abandon = self.abandon_probs[step]
            operating = entering * (1.0 - abandon)

            # Exits are collected at the date they happen, not at the horizon:
            # winding down in quarter one and failing in quarter eight are not
            # worth the same thing.
            total = total + (discount**step) * entering * abandon * self.orderly_values[step]
            # Dividends are paid at the end of the quarter, out of what it
            # left, so only the mass that operated *and* survived collects
            # them -- and they are discounted to that date, like the recoveries
            # below rather than like the exit above.
            total = total + (discount ** (step + 1)) * active[step + 1] * reward

            failed = operating - active[step + 1]
            total = total + (discount ** (step + 1)) * failed * self.failure_values[step]
        return total + (discount ** len(self.rewards)) * active[-1] * self.terminal_value

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
