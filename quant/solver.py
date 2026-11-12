"""Infinite-horizon solution of the v0 bank by policy iteration on a grid (design.md section 5).

    V(w, S) = max{ 0, max_c [ flow(w, K) + beta E V(w', S') ] }

on a grid over equity `w` and the GRC stock `S`. (The design writes the stock
as G; here it is S, because G is also the paper's effective risk aversion.)

**Expectations** use the quadrature rule of `quant.quadrature`, so a solve is
deterministic.

**The grid, interpolation and policy iteration** are the shared machinery of
`quant.grid`: multilinear interpolation (linear interpolation of a concave
function stays concave), equity's off-grid rules, and Adam from two starts with
repeated policy evaluation. This module supplies the bank's state, controls and
objective.

**The inner maximization** is gradient ascent (Adam) on unconstrained raw
controls, for every grid point at once. The controls are mapped so every
constraint holds by construction:

    I     = I_max sigmoid(z_I)                 the book
    g     = delta S_hi sigmoid(z_g)            keeps S' on the grid
    q     = q_max sigmoid(z_q)                 the BTC treasury, bought spot
    K     = floor(I, g, q) + softplus(z_K)     capital ratio, cash, no negative equity
    alpha = alpha_max sigmoid(z_alpha)
    H     = 1.2 sigmoid(z_h) (b_BTC I + q)     hedge ratio in [0, 1.2]: perps hedge

The hedge can reduce BTC exposure but not create long exposure. Long exposure
beyond the book's comes only from the spot treasury, which uses cash and
capital; otherwise synthetic exposure through perps would always win and the
treasury would never be held.

The ratio runs to 1.2, a small bounded over-hedge, so that a full hedge is an
interior point. Capped at 1, the full hedge sat at the edge of a squashed
control, where the gradient vanishes, and the solver stalled at a ratio of
0.987.

"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from quant.grid import StateGrid, policy_iteration
from quant.model import (
    Controls,
    capital_floor,
    outcomes,
    issuance_cost,
    next_grc,
    next_wealth,
    shareholder_flow,
    shock_correlation,
)
from quant.numerics import NumericsProfile
from quant.params import Params
from quant.quadrature import JumpRule, ShockRule, pareto_jump_rule, student_t_rule


@dataclass(frozen=True)
class GridSpec:
    w_lo: float = -1.0
    w_hi: float = 0.6
    n_w: int = 129
    s_hi: float = 0.08     # 0.04 capped spend below what the bank wanted
    n_s: int = 9
    I_max: float = 1.5
    n_hermite: int = 5
    n_mixing: int = 8
    n_jump: int = 8


@dataclass(frozen=True)
class Grid:
    """The bank's grid: equity by GRC stock, over `quant.grid.StateGrid`."""

    spec: GridSpec
    state: StateGrid

    @property
    def w(self) -> torch.Tensor:
        return self.state.axes[0]

    @property
    def s(self) -> torch.Tensor:
        return self.state.axes[1]

    @property
    def dw(self) -> float:
        return self.state.step(0)

    @property
    def ds(self) -> float:
        return self.state.step(1)

    def states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Every (w, S) grid point, flattened in row-major (w, S) order."""
        w, s = self.state.states()
        return w, s


def make_grid(spec: GridSpec, profile: NumericsProfile) -> Grid:
    state = StateGrid.uniform(
        [(spec.w_lo, spec.w_hi, spec.n_w), (0.0, spec.s_hi, spec.n_s)], profile
    )
    return Grid(spec=spec, state=state)


def interpolate(
    p: Params, V: torch.Tensor, grid: Grid, w: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """V at arbitrary (w, s); V is (n_w, n_s). See `quant.grid` for the rules."""
    return grid.state.interpolate(V, [w, s], lambda x: issuance_cost(p, x))


HEDGE_MAX = 1.2  # hedge ratio bound; > 1 so that a full hedge is interior


@dataclass
class RawPolicy:
    """Unconstrained parameters, one per control per grid point."""

    z_I: torch.Tensor
    z_g: torch.Tensor
    z_K: torch.Tensor
    z_alpha: torch.Tensor
    z_h: torch.Tensor
    z_q: torch.Tensor

    @classmethod
    def initial(cls, p: Params, n: int, profile: NumericsProfile) -> "RawPolicy":
        """Lend most of the first-best book, hedge almost fully, hold a thin
        buffer and little BTC treasury."""
        def full(x):
            return torch.full((n,), x, dtype=profile.dtype, device=profile.device)

        return cls(
            z_I=full(0.5), z_g=full(-2.0), z_K=full(-3.0), z_alpha=full(-3.0),
            z_h=full(1.3), z_q=full(-4.0),
        )

    def tensors(self) -> list[torch.Tensor]:
        return [self.z_I, self.z_g, self.z_K, self.z_alpha, self.z_h, self.z_q]

    def controls(self, p: Params, grid: Grid) -> Controls:
        spec = grid.spec
        I = spec.I_max * torch.sigmoid(self.z_I)
        g = p.delta * spec.s_hi * torch.sigmoid(self.z_g)
        q = p.q_max * torch.sigmoid(self.z_q)
        K = capital_floor(p, I, g, q) + F.softplus(self.z_K)
        alpha = p.alpha_max * torch.sigmoid(self.z_alpha)
        H = HEDGE_MAX * torch.sigmoid(self.z_h) * (p.b_BTC * I + q)
        return Controls(K=K, g=g, I=I, alpha=alpha, H=H, q=q)


def continuation(
    p: Params, grid: Grid, rule: ShockRule, jumps: JumpRule | None, V: torch.Tensor,
    w: torch.Tensor, s: torch.Tensor, c: Controls,
) -> torch.Tensor:
    """flow(w, K) + beta E V(w', S'), before the walk-away floor."""
    s_next = next_grc(p, s, c.g)
    out = outcomes(p, c, s_next, rule, jumps)
    future = out.expect(interpolate(p, V, grid, out.wealth, s_next.unsqueeze(-1)))
    return shareholder_flow(p, w, c.K) + p.beta * future


@dataclass
class Solution:
    params: Params                  # per period: what the model ran on
    grid: Grid
    rule: ShockRule
    jumps: JumpRule | None
    V: torch.Tensor                 # (n_w, n_s)
    policy: RawPolicy
    residual: float                 # sup |V - T V| at the end
    iterations: int
    history: list[float] = field(default_factory=list)
    inputs: Params | None = None    # as given, rates per year

    def controls(self) -> Controls:
        return self.policy.controls(self.params, self.grid)

    def on_grid(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.grid.spec.n_w, self.grid.spec.n_s)

    @property
    def lower_edge_is_walk_away(self) -> bool:
        return bool((self.V[0] == 0).all())

    @property
    def upper_edge_pays_out(self) -> bool:
        """At w_hi the bank should be paying a dividend at every stock level."""
        c = self.controls()
        w, _ = self.grid.states()
        return bool((self.on_grid(c.K)[-1] < self.on_grid(w)[-1]).all())


@dataclass(frozen=True)
class RisklessSteadyState:
    """The exact infinite-horizon answer when nothing is random (an oracle).

    With no shocks and GRC that does nothing (a_A = 0), a bank with enough
    equity holds exactly its capital floor L(I), lends I*, and pays out the
    rest every period. Equity closes at the same w_ss each period, so

        V(w_ss) = (w_ss - K*) / (1 - beta),    V(w) = w - K* + beta V(w_ss)

    for any w >= K*. Holding equity above the floor is never worth it: a unit
    costs 1 now and returns beta (1 + r - tau) < 1. I* maximizes
    -L(I) + beta w'(L(I), I), which is concave in I, so golden section finds it.
    """

    I: float
    K: float
    w_ss: float
    beta: float

    def value(self, w: torch.Tensor) -> torch.Tensor:
        return w - self.K + self.beta * (self.w_ss - self.K) / (1.0 - self.beta)


def riskless_steady_state(p: Params, profile: NumericsProfile) -> RisklessSteadyState:
    p = p.per_period()
    if (p.s_P, p.s_N, p.s_BTC, p.mu_BTC, p.a_A, p.jump_rate) != (0.0,) * 6:
        raise ValueError(
            "the riskless oracle needs s_P = s_N = s_BTC = mu_BTC = a_A = jump_rate = 0"
        )

    def closing(I):
        K = capital_floor(p, I, torch.zeros_like(I))
        c = Controls(K=K, g=torch.zeros_like(I), I=I, alpha=p.alpha_max * torch.ones_like(I), H=p.b_BTC * I)
        return K, next_wealth(p, c, torch.zeros_like(I), profile.zeros(1, 3)).squeeze(-1)

    def objective(I):
        K, w_next = closing(I)
        return -K + p.beta * w_next

    golden = (math.sqrt(5.0) - 1.0) / 2.0
    lo, hi = profile.tensor(0.0), profile.tensor(1.0)
    for _ in range(200):
        a, b = hi - golden * (hi - lo), lo + golden * (hi - lo)
        if objective(a) > objective(b):
            hi = b
        else:
            lo = a
    I = 0.5 * (lo + hi)
    K, w_ss = closing(I)
    return RisklessSteadyState(I=float(I), K=float(K), w_ss=float(w_ss), beta=p.beta)


def solve(
    p: Params,
    profile: NumericsProfile,
    spec: GridSpec = GridSpec(),
    tol: float = 1e-7,
    max_iterations: int = 200,
    improve_steps: int = 40,
    first_improve_steps: int = 400,
    evaluate_steps: int | None = None,
    polish_steps: int = 300,
    lr: float = 0.05,
    verbose: bool = False,
) -> Solution:
    inputs = p
    p = p.per_period()  # rates stated per year become rates per period (design.md 2.5)
    grid = make_grid(spec, profile)
    rule = student_t_rule(
        shock_correlation(p), p.nu, profile, n_hermite=spec.n_hermite, n_mixing=spec.n_mixing
    )
    jumps = (
        pareto_jump_rule(p.jump_mean, p.jump_tail, profile, n=spec.n_jump)
        if p.jump_rate > 0 else None
    )
    w, s = grid.states()
    n = w.shape[0]
    # Start risk-neutral: V = w - w_lo is linear, so the first improvements see
    # no curvature at all. Starting from relu(w) -- liquidation value -- put a
    # convex kink at zero, which made gambling look attractive in the first
    # iterations, and squashed controls pushed to their bounds there stayed.
    V0 = (grid.w - spec.w_lo).unsqueeze(-1).expand(-1, spec.n_s).clone()

    def objective(V, raw):
        return continuation(p, grid, rule, jumps, V, w, s, RawPolicy(*raw).controls(p, grid))

    result = policy_iteration(
        objective, RawPolicy.initial(p, n, profile).tensors(), V0, p.beta,
        tol=tol, max_iterations=max_iterations, improve_steps=improve_steps,
        first_improve_steps=first_improve_steps, evaluate_steps=evaluate_steps,
        polish_steps=polish_steps, lr=lr, verbose=verbose,
    )
    V, policy, residual = result.V, RawPolicy(*result.raw), result.residual
    it, history = result.iterations - 1, result.history

    return Solution(
        params=p, grid=grid, rule=rule, jumps=jumps, V=V, policy=policy,
        residual=residual, iterations=it + 1, history=history, inputs=inputs,
    )
