"""What a solved bank says: the v0 outputs (design.md sections 1, 7 and appendix B).

Everything here reads a `Solution`; nothing re-solves except `grc_value`,
which compares solves by construction. Expectations run over every outcome,
jumps included.

**Effective risk aversion without second derivatives.** The paper's
G = -E[P''] / E[P'] needs the value function's curvature, and a linearly
interpolated value function has none between grid points. So G is measured
the way the paper uses it -- as the price of risk -- from first derivatives
only:

    G(w, S) = -E[V_w(w') (w' - E w')] / (E[V_w(w')] Var(w'))

the required return per unit of variance for scaling up the bank's own risk.
Under normal shocks Stein's lemma makes this exactly -E[V_ww] / E[V_w]; under
the Student-t it is the natural generalization. The same construction gives
the **hurdle** for a small new position with payoff shock eps:

    hurdle(eps) = -E[V_w(w') eps] / E[V_w(w')]

which is eq. (10)'s G cov(eps, eps_P) under normality, and exact in general.
V_w comes from autograd through the interpolant: the slope of each segment,
averaged over the quadrature nodes, which is smooth in the state.
"""

from dataclasses import dataclass, replace

import torch

from quant.model import next_grc, outcomes
from quant.params import Params
from quant.solver import GridSpec, Solution, interpolate, solve


def _next_period(sol: Solution):
    """Every next-period outcome for every grid state, and V_w at each."""
    p, grid = sol.params, sol.grid
    _, s = grid.states()
    with torch.no_grad():
        c = sol.controls()
        s_next = next_grc(p, s, c.g)
        out = outcomes(p, c, s_next, sol.rule, sol.jumps)
    point = out.wealth.clone().requires_grad_(True)
    value = interpolate(p, sol.V, grid, point, s_next.unsqueeze(-1))
    (slope,) = torch.autograd.grad(value.sum(), point)
    return out, slope


@dataclass(frozen=True)
class RiskMap:
    """Per grid state, flattened in the solver's (w, S) order."""

    shadow: torch.Tensor     # E[V_w(w')]: the value of a unit of equity next period
    G: torch.Tensor          # effective risk aversion
    hurdle_N: torch.Tensor   # hurdle for a small unit of the new product's risk
    hurdle_P: torch.Tensor   # hurdle for a small unit of risk shaped like the book's


def risk_map(sol: Solution) -> RiskMap:
    p = sol.params
    out, slope = _next_period(sol)
    E = out.expect
    shadow = E(slope)
    centered = out.wealth - E(out.wealth).unsqueeze(-1)
    variance = E(centered * centered)
    G = -E(slope * centered) / (shadow * torch.clamp(variance, min=1e-300))
    xi_P, xi_N = out.shocks[:, 0], out.shocks[:, 1]
    hurdle_N = -E(slope * (p.s_N * xi_N)) / shadow
    hurdle_P = -E(slope * (p.s_P * xi_P)) / shadow
    return RiskMap(shadow=shadow, G=G, hurdle_N=hurdle_N, hurdle_P=hurdle_P)


@dataclass(frozen=True)
class Thresholds:
    """Per GRC grid level: where the bank stops recapitalizing and starts paying out."""

    walk_away: torch.Tensor  # highest w at which V = 0 (nan if none on the grid)
    payout: torch.Tensor     # lowest w at which K < w, i.e. a dividend is paid


def thresholds(sol: Solution, tol: float = 1e-6) -> Thresholds:
    w = sol.grid.w
    K = sol.on_grid(sol.controls().K)
    walk, pay = [], []
    for j in range(sol.grid.spec.n_s):
        dead = sol.V[:, j] <= tol
        walk.append(w[dead].max() if bool(dead.any()) else torch.tensor(float("nan"), dtype=w.dtype))
        paying = K[:, j] < w - tol
        pay.append(w[paying].min() if bool(paying.any()) else torch.tensor(float("nan"), dtype=w.dtype))
    return Thresholds(walk_away=torch.stack(walk), payout=torch.stack(pay))


def target_stock(sol: Solution, at_w: float | None = None) -> float:
    """The GRC stock the bank converges to, at an equity level well above payout.

    The fixed point of S -> S' = (1 - delta) S + g(w, S) along one row of the
    grid, found where S' - S changes sign, by linear interpolation.
    """
    grid = sol.grid
    row = int(torch.argmin((grid.w - (at_w if at_w is not None else grid.w[-1])).abs()))
    g = sol.on_grid(sol.controls().g)[row]
    gap = next_grc(sol.params, grid.s, g) - grid.s
    for j in range(len(grid.s) - 1):
        if gap[j] >= 0 >= gap[j + 1]:
            a, b = float(gap[j]), float(gap[j + 1])
            t = a / (a - b) if a != b else 0.0
            return float(grid.s[j] + t * (grid.s[j + 1] - grid.s[j]))
    return float("nan") if bool((gap < 0).all()) is False else 0.0


@dataclass(frozen=True)
class GrcValue:
    """What a GRC program is worth to this bank (design.md section 1).

    Value is V at one equity level and zero GRC stock -- a bank deciding
    whether to start a program -- compared with the same bank without one.

    No counterfactual benchmark is computed. A "premium" measured against a
    bank that raises equity for free, or one forced to recapitalize after any
    loss, is an academic construction: neither bank exists, so the number has
    no quantitative meaning, and the qualitative point is already known. See
    design.md section 11, the decisions log, for what was tried and why it was
    removed.
    """

    gain: float    # V with a GRC program minus V without one
    stock: float   # the GRC stock the bank converges to


def grc_value(p: Params, profile, spec: GridSpec, at_w: float = 0.1, **solve_args) -> GrcValue:
    def value(q):
        sol = solve(q, profile, spec, **solve_args)
        row = int(torch.argmin((sol.grid.w - at_w).abs()))
        return float(sol.V[row, 0]), sol

    with_grc, sol = value(p)
    without, _ = value(replace(p, a_A=0.0, a_B=0.0, a_J=0.0))
    return GrcValue(gain=with_grc - without, stock=target_stock(sol))
