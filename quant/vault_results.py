"""What a solved vault firm says (design.md sections 1 and 2).

Mirrors `quant.results` for the bank form. Everything reads a `VaultSolution`;
only `grc_value` re-solves.

- **Effective risk aversion** of the firm's own equity, measured as the price
  of risk from first derivatives (design.md section 10), on the next-period
  distribution of its equity.
- **The allocation's marginal values**: the gradient of the firm's objective
  with respect to each strategy's share, holding the rest of the policy fixed.
  At an optimum every strategy the vault holds has the same marginal value,
  and every strategy it does not hold has a lower one -- the first-order
  condition the allocation must satisfy. This is the vault's counterpart of
  the hurdle check.
- **Thresholds**: where the firm walks away and where it starts paying out.
- **The target GRC stock**, and **GRC's value** to the firm: the firm with a
  program against the same firm without one.
"""

from dataclasses import dataclass, replace

import torch

from quant.model import issuance_cost, next_grc
from quant.params import Params
from quant.vault import VaultControls, vault_outcomes
from quant.vault_solver import VaultSpec, VaultSolution, solve_vault, vault_continuation


@dataclass(frozen=True)
class VaultRiskMap:
    """Per grid state, flattened in the grid's (w, S, A) order."""

    shadow: torch.Tensor     # E[V_w]: the value of a unit of the firm's equity next period
    G: torch.Tensor          # effective risk aversion of the firm's equity


def risk_map(sol: VaultSolution) -> VaultRiskMap:
    p, grid = sol.params, sol.grid
    _, s, A = grid.states()
    with torch.no_grad():
        c = sol.controls()
        s_next = next_grc(p, s, c.g)
        out = vault_outcomes(p, A, c, s_next, sol.rule, sol.jumps)
    point = out.wealth.clone().requires_grad_(True)
    value = grid.interpolate(
        sol.V, [point, s_next.unsqueeze(-1), out.aum], lambda x: issuance_cost(p, x)
    )
    (slope,) = torch.autograd.grad(value.sum(), point)
    E = out.expect
    shadow = E(slope)
    centered = out.wealth - E(out.wealth).unsqueeze(-1)
    G = -E(slope * centered) / (shadow * torch.clamp(E(centered * centered), min=1e-300))
    return VaultRiskMap(shadow=shadow, G=G)


def allocation_gradient(sol: VaultSolution) -> torch.Tensor:
    """d objective / d a_i at the solved policy, per state and strategy: (n, 3)."""
    p, grid = sol.params, sol.grid
    w, s, A = grid.states()
    with torch.no_grad():
        c = sol.controls()
    alloc = c.alloc.clone().requires_grad_(True)
    moved = VaultControls(K=c.K, g=c.g, q=c.q, H=c.H, alloc=alloc)
    value = vault_continuation(p, sol.spec, grid, sol.rule, sol.jumps, sol.V, w, s, A, moved)
    (grad,) = torch.autograd.grad(value.sum(), alloc)
    return grad


@dataclass(frozen=True)
class VaultThresholds:
    """Per (S, A) grid point: where the firm walks away and where it pays out."""

    walk_away: torch.Tensor  # highest w at which V = 0 (nan if none)
    payout: torch.Tensor     # lowest w at which K < w (nan if none)


def thresholds(sol: VaultSolution, tol: float = 1e-6) -> VaultThresholds:
    w = sol.grid.axes[0]
    V = sol.V
    K = sol.controls().K.reshape(sol.grid.shape)
    nan = torch.tensor(float("nan"), dtype=w.dtype)
    walk = torch.full(V.shape[1:], float("nan"), dtype=w.dtype)
    pay = torch.full(V.shape[1:], float("nan"), dtype=w.dtype)
    for j in range(V.shape[1]):
        for k in range(V.shape[2]):
            dead = V[:, j, k] <= tol
            walk[j, k] = w[dead].max() if bool(dead.any()) else nan
            paying = K[:, j, k] < w - tol
            pay[j, k] = w[paying].min() if bool(paying.any()) else nan
    return VaultThresholds(walk_away=walk, payout=pay)


def target_stock(sol: VaultSolution, at_w: float | None = None, at_A: float | None = None) -> float:
    """The GRC stock the firm converges to at one (w, A): the fixed point of
    S -> (1 - delta) S + g(w, S, A) along the S axis."""
    grid = sol.grid
    w_axis, s_axis, A_axis = grid.axes
    i = int(torch.argmin((w_axis - (at_w if at_w is not None else w_axis[-1])).abs()))
    k = int(torch.argmin((A_axis - (at_A if at_A is not None else sol.params.capacity)).abs()))
    g = sol.controls().g.reshape(grid.shape)[i, :, k]
    gap = next_grc(sol.params, s_axis, g) - s_axis
    if bool((gap <= 0).all()):
        return 0.0
    for j in range(len(s_axis) - 1):
        if gap[j] >= 0 >= gap[j + 1]:
            a, b = float(gap[j]), float(gap[j + 1])
            t = a / (a - b) if a != b else 0.0
            return float(s_axis[j] + t * (s_axis[j + 1] - s_axis[j]))
    return float("nan")


@dataclass(frozen=True)
class VaultGrcValue:
    """What a GRC program is worth to the firm at one (w, A), starting from no
    program: V with GRC minus V with GRC that does nothing."""

    gain: float
    stock: float


def grc_value(
    p: Params, profile, spec: VaultSpec = VaultSpec(), at_w: float = 0.1,
    at_A: float | None = None, **iteration_args,
) -> VaultGrcValue:
    def value(q):
        sol = solve_vault(q, profile, spec, **iteration_args)
        w_axis, _, A_axis = sol.grid.axes
        i = int(torch.argmin((w_axis - at_w).abs()))
        k = int(torch.argmin((A_axis - (at_A if at_A is not None else q.capacity)).abs()))
        return float(sol.V[i, 0, k]), sol

    with_grc, sol = value(p)
    without, _ = value(replace(p, a_A=0.0, a_B=0.0, a_J=0.0))
    return VaultGrcValue(gain=with_grc - without, stock=target_stock(sol, at_w, at_A))
