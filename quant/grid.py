"""Grids over the state, and policy iteration on them (design.md section 5).

Shared by both business forms. The state is a product of one-dimensional
axes; **axis 0 is always the firm's equity**, because that is the axis with
economic rules off the grid. The other axes (the GRC stock, assets under
management, ...) are clamped to their range.

**Interpolation is multilinear.** Linear interpolation of a concave function is
concave; a smoother cubic can bulge into small convex patches near kinks, and
the solver gambles on them (design.md section 11). Off the grid in equity, the
value of an action the firm can always take:

- above the top, pay the excess out: slope one;
- below the bottom, issue the shortfall and continue as at the bottom, floored
  at zero by walk-away: V(lo) - x - C(x), x = lo - w. Issuing the gap is a
  lower bound on what the firm can do (one issue of the total costs more than
  two smaller ones, as C is convex), so it errs toward charging the downside.
  Clamping instead capped the downside of any gamble and made the first solve
  diverge.

**Policy iteration** (`policy_iteration`): Adam on every grid point's raw
controls at once, from two starts each step -- the last policy and a fresh one,
keeping the better per grid point, because a squashed control pushed to its
bound has no gradient left to come back -- then repeated evaluation of the
fixed policy, and a final slow polish. The problem supplies only its objective:
flow plus discounted expected value, before the walk-away floor.
"""

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import torch

from quant.numerics import NumericsProfile


@dataclass(frozen=True)
class StateGrid:
    """A product grid. `axes[0]` is equity; each axis is uniform."""

    axes: tuple[torch.Tensor, ...]

    @classmethod
    def uniform(
        cls, bounds: list[tuple[float, float, int]], profile: NumericsProfile
    ) -> "StateGrid":
        axes = []
        for lo, hi, n in bounds:
            if n < 2:
                raise ValueError("every axis needs at least two points")
            axes.append(
                torch.linspace(lo, hi, n, dtype=profile.dtype, device=profile.device)
            )
        return cls(axes=tuple(axes))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(a.shape[0] for a in self.axes)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    def lo(self, k: int) -> float:
        return float(self.axes[k][0])

    def hi(self, k: int) -> float:
        return float(self.axes[k][-1])

    def step(self, k: int) -> float:
        return (self.hi(k) - self.lo(k)) / (self.shape[k] - 1)

    def states(self) -> list[torch.Tensor]:
        """Every grid point, one flattened tensor per axis, row-major."""
        mesh = torch.meshgrid(*self.axes, indexing="ij")
        return [m.reshape(-1) for m in mesh]

    def interpolate(
        self,
        V: torch.Tensor,
        points: list[torch.Tensor],
        issuance_cost: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """V (shaped like the grid) at arbitrary points, one tensor per axis.

        Points broadcast together. Multilinear inside; equity's off-grid rules
        outside (module docstring); other axes clamped.
        """
        points = list(torch.broadcast_tensors(*points))
        w = points[0]
        excess = torch.relu(w - self.hi(0))
        shortfall = torch.relu(self.lo(0) - w)

        index, frac = [], []
        for k, x in enumerate(points):
            u = (torch.clamp(x, self.lo(k), self.hi(k)) - self.lo(k)) / self.step(k)
            i = torch.clamp(torch.floor(u), 0, self.shape[k] - 2)
            index.append(i.long())
            frac.append(u - i)

        strides = [math.prod(self.shape[k + 1:]) for k in range(len(self.shape))]
        flat = V.reshape(-1)
        inside = torch.zeros_like(w)
        for corner in itertools.product((0, 1), repeat=len(self.shape)):
            position = sum((index[k] + c) * strides[k] for k, c in enumerate(corner))
            weight = 1.0
            for k, c in enumerate(corner):
                weight = weight * (frac[k] if c else 1.0 - frac[k])
            inside = inside + weight * flat[position]
        return torch.relu(inside + excess - shortfall - issuance_cost(shortfall))


@dataclass
class IterationResult:
    V: torch.Tensor
    raw: list[torch.Tensor]
    residual: float
    iterations: int
    history: list[float]


def policy_iteration(
    objective: Callable[[torch.Tensor, list[torch.Tensor]], torch.Tensor],
    initial_raw: list[torch.Tensor],
    V0: torch.Tensor,
    beta: float,
    tol: float = 1e-7,
    max_iterations: int = 200,
    improve_steps: int = 40,
    first_improve_steps: int = 400,
    evaluate_steps: int | None = None,
    polish_steps: int = 300,
    lr: float = 0.05,
    verbose: bool = False,
) -> IterationResult:
    """Solve V = max{0, max_raw objective(V, raw)} on a grid.

    `objective(V, raw)` returns, per grid point (flattened), the flow plus the
    discounted expected continuation under the controls `raw` maps to. The
    walk-away floor is applied here.

    Policy evaluation contracts by beta per step, so the number of evaluation
    steps it needs grows like 1 / (1 - beta): about 50 annually, about 200
    quarterly. A fixed 30 left the quarterly solve at its iteration cap.
    """
    if evaluate_steps is None:
        evaluate_steps = max(30, math.ceil(3.0 / (1.0 - beta)))
    shape = V0.shape

    def bellman(V, raw):
        return torch.relu(objective(V, raw)).reshape(shape)

    def ascend(V, start, steps, rate):
        params = [z.detach().clone().requires_grad_(True) for z in start]
        optimizer = torch.optim.Adam(params, lr=rate)
        for _ in range(steps):
            optimizer.zero_grad()
            (-objective(V, params).sum()).backward()
            optimizer.step()
        found = [z.detach() for z in params]
        with torch.no_grad():
            value = objective(V, found)
        return found, value

    def evaluate(V, raw):
        residual = float("inf")
        with torch.no_grad():
            for _ in range(evaluate_steps):
                V_next = bellman(V, raw)
                residual = float((V_next - V).abs().max())
                V = V_next
        return V, residual

    V = V0.clone()
    fresh = [z.clone() for z in initial_raw]
    raw = fresh
    history: list[float] = []
    residual = float("inf")
    it = 0
    for it in range(max_iterations):
        steps = first_improve_steps if it == 0 else improve_steps
        warm, warm_value = ascend(V, raw, steps, lr)
        if it == 0:
            raw = warm
        else:
            restart, restart_value = ascend(V, fresh, steps, lr)
            better = restart_value > warm_value
            raw = [torch.where(better, r, k) for r, k in zip(restart, warm)]
        V, residual = evaluate(V, raw)
        history.append(residual)
        if verbose:
            print(f"iteration {it:3d}  residual {residual:.3e}  max V {float(V.max()):.6f}")
        if residual < tol and it > 0:
            break

    # Polish: one long, slow ascent from the converged policy, then a final
    # evaluation. The value is nearly flat in some controls near their optimum,
    # so Adam at the working rate jitters by about its step size there.
    raw, _ = ascend(V, raw, polish_steps, lr / 10.0)
    V, residual = evaluate(V, raw)
    return IterationResult(V=V, raw=raw, residual=residual, iterations=it + 1, history=history)
