"""What a solved bank says: the v0 outputs (design.md sections 2, 3.3, 6).

Everything here reads a `Solution`; nothing re-solves except `grc_value`,
which compares solves by construction.

**Effective risk aversion without second derivatives.** The paper's
G = -E[P''] / E[P'] needs the value function's curvature, and a linearly
interpolated value function has none between grid points. So G is measured
the way the paper uses it -- as the price of risk -- from first derivatives
only:

    G(w, S) = -E[V_w(w') (w' - E w')] / (E[V_w(w')] Var(w'))

the required return per unit of variance for scaling up the bank's own risk.
Under normal shocks Stein's lemma makes this exactly -E[V_ww] / E[V_w]; under
the Student-t it is the natural generalisation. The same construction gives
the **hurdle** for a small new position with payoff shock eps:

    hurdle(eps) = -E[V_w(w') eps] / E[V_w(w')]

which is eq. (10)'s G cov(eps, eps_P) under normality, and exact in general.
V_w comes from autograd through the interpolant: the slope of each segment,
averaged over the quadrature nodes, which is smooth in the state.
"""

from dataclasses import dataclass, replace

import torch

from quant.model import next_grc, next_wealth
from quant.params import Params
from quant.solver import GridSpec, Solution, interpolate, solve


def _next_period(sol: Solution):
    """w' at every node for every grid state, and V_w at those points."""
    p, grid, rule = sol.params, sol.grid, sol.rule
    _, s = grid.states()
    with torch.no_grad():
        c = sol.controls()
        s_next = next_grc(p, s, c.g)
        w_next = next_wealth(p, c, s_next, rule.nodes)
    point = w_next.clone().requires_grad_(True)
    value = interpolate(p, sol.V, grid, point, s_next.unsqueeze(-1))
    (slope,) = torch.autograd.grad(value.sum(), point)
    return w_next, slope


@dataclass(frozen=True)
class RiskMap:
    """Per grid state, flattened in the solver's (w, S) order."""

    shadow: torch.Tensor     # E[V_w(w')]: the value of a unit of equity next period
    G: torch.Tensor          # effective risk aversion
    hurdle_N: torch.Tensor   # hurdle for a small unit of the new product's risk
    hurdle_P: torch.Tensor   # hurdle for a small unit of risk shaped like the book's


def risk_map(sol: Solution) -> RiskMap:
    p, rule = sol.params, sol.rule
    w_next, slope = _next_period(sol)
    E = rule.expect
    shadow = E(slope)
    centred = w_next - E(w_next).unsqueeze(-1)
    variance = E(centred * centred)
    G = -E(slope * centred) / (shadow * torch.clamp(variance, min=1e-300))
    xi_P, xi_N = rule.nodes[:, 0], rule.nodes[:, 1]
    hurdle_N = -E(slope * (p.s_N * xi_N)) / shadow
    hurdle_P = -E(slope * (p.s_P * xi_P)) / shadow
    return RiskMap(shadow=shadow, G=G, hurdle_N=hurdle_N, hurdle_P=hurdle_P)


@dataclass(frozen=True)
class Thresholds:
    """Per GRC grid level: where the bank stops recapitalising and starts paying out."""

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
    """What GRC is worth, split by mechanism (design.md section 2).

    Each row is a separate solve with GRC allowed to act through some
    mechanisms and not others. Values are V at one equity level and zero GRC
    stock -- a bank deciding whether to start a programme.
    """

    none: float          # GRC does nothing
    mean_only: float     # mechanism A: smaller expected loss
    spread_only: float   # mechanism B: smaller spread -- the Froot-Stein premium
    both: float
    stock_mean_only: float
    stock_both: float

    @property
    def premium_share(self) -> float:
        """Share of GRC's value that only a capital-constrained bank would pay for."""
        total = self.both - self.none
        return (self.both - self.mean_only) / total if total > 0 else float("nan")


def grc_value(p: Params, profile, spec: GridSpec, at_w: float = 0.1, **solve_args) -> GrcValue:
    def solved(q):
        return solve(q, profile, spec, **solve_args)

    def value(sol):
        row = int(torch.argmin((sol.grid.w - at_w).abs()))
        return float(sol.V[row, 0])

    none = solved(replace(p, a_A=0.0, a_B=0.0))
    mean = solved(replace(p, a_B=0.0))
    spread = solved(replace(p, a_A=0.0))
    both = solved(p)
    return GrcValue(
        none=value(none), mean_only=value(mean), spread_only=value(spread), both=value(both),
        stock_mean_only=target_stock(mean), stock_both=target_stock(both),
    )
