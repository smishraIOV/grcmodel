"""Infinite-horizon solution of the v0 bank by policy iteration on a grid (design.md section 5).

    V(w, S) = max{ 0, max_c [ flow(w, K) + beta E V(w', S') ] }

on a grid over equity `w` and the GRC stock `S`. (The design writes the stock
as G; here it is S, because G is also the paper's effective risk aversion.)

**Expectations** use the quadrature rule of `quant.quadrature`, so a solve is
deterministic.

**Interpolation** is linear in w by default, and linear in S. Linear
interpolation of a concave function is concave; Catmull-Rom (kept as an option)
is smoother but can bulge into a small convex patch near a kink, and the solver
gambles on it. On a 33-point grid that showed up as hedge errors in matched
+/- pairs of 0.03 -- the bank treating a zero exposure as a local minimum --
and a 1.7e-4 error against the exact no-shock answer. Linear gave zero hedge
error and 2e-8 on the same grid. In a model whose subject is the value
function's curvature, an interpolant that invents curvature is the wrong
trade. The effective risk aversion is read from the expectation over shocks,
which is smooth, not from the interpolant.

Off the grid:

- above `w_hi`, V rises with slope one, because extra equity is simply paid
  out. The grid must reach past the payout threshold, and `Solution` reports
  whether it did;
- below `w_lo`, the bank issues equity to get back onto the grid (see
  `interpolate`). `Solution` reports whether `w_lo` is in the walk-away region.

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

Each improvement step warm-starts from the last policy, and between
improvements the fixed policy is evaluated several times (Howard's
improvement), which is where most of the convergence comes from.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

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
    interpolation: str = "linear"  # or "cubic" (Catmull-Rom); see `interpolate`


@dataclass(frozen=True)
class Grid:
    spec: GridSpec
    w: torch.Tensor   # (n_w,)
    s: torch.Tensor   # (n_s,)

    @property
    def dw(self) -> float:
        return (self.spec.w_hi - self.spec.w_lo) / (self.spec.n_w - 1)

    @property
    def ds(self) -> float:
        return self.spec.s_hi / (self.spec.n_s - 1)

    def states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Every (w, S) grid point, flattened in row-major (w, S) order."""
        ww, ss = torch.meshgrid(self.w, self.s, indexing="ij")
        return ww.reshape(-1), ss.reshape(-1)


def make_grid(spec: GridSpec, profile: NumericsProfile) -> Grid:
    if spec.n_s < 2:
        raise ValueError("the GRC grid needs at least two points")
    w = torch.linspace(spec.w_lo, spec.w_hi, spec.n_w, dtype=profile.dtype, device=profile.device)
    s = torch.linspace(0.0, spec.s_hi, spec.n_s, dtype=profile.dtype, device=profile.device)
    return Grid(spec=spec, w=w, s=s)


def interpolate(
    p: Params, V: torch.Tensor, grid: Grid, w: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """V at arbitrary (w, s). V is (n_w, n_s); w and s broadcast together.

    Linear (or Catmull-Rom, if asked) in w on a grid padded with one linearly
    extrapolated ghost row at each end; linear in s.

    Off the grid in w, the value of an action the bank can always take:

    - above `w_hi`, pay the excess out: slope one;
    - below `w_lo`, issue the shortfall and continue as at `w_lo`, floored at
      zero by walk-away: V(w_lo) - x - C(x), x = w_lo - w.

    The lower rule matters. Clamping to V(w_lo) instead -- assuming the edge
    is already the walk-away region -- capped the downside of any gamble at a
    positive number while its upside grew without limit, and the first solve
    took an unbounded factor position and diverged. Issuing the gap is a
    lower bound on what the bank can do (two issues cost less than one of
    the same total, as C is convex), so it errs toward charging the downside.
    """
    spec = grid.spec
    n_w, n_s = spec.n_w, spec.n_s
    padded = torch.cat([2 * V[:1] - V[1:2], V, 2 * V[-1:] - V[-2:-1]], dim=0)  # (n_w + 2, n_s)
    flat = padded.reshape(-1)

    w, s = torch.broadcast_tensors(w, s)
    excess = torch.relu(w - spec.w_hi)
    shortfall = torch.relu(spec.w_lo - w)
    u = (torch.clamp(w, spec.w_lo, spec.w_hi) - spec.w_lo) / grid.dw
    i = torch.clamp(torch.floor(u), 0, n_w - 2)
    t = u - i
    v = torch.clamp(s, 0.0, spec.s_hi) / grid.ds
    j = torch.clamp(torch.floor(v), 0, n_s - 2)
    q = v - j
    i, j = i.long(), j.long()

    def along_w(col):
        # padded row i is grid row i - 1, so rows i..i+3 are grid rows i-1..i+2
        p0 = flat[i * n_s + col]
        p1 = flat[(i + 1) * n_s + col]
        p2 = flat[(i + 2) * n_s + col]
        p3 = flat[(i + 3) * n_s + col]
        if spec.interpolation == "linear":
            return p1 + (p2 - p1) * t
        return 0.5 * (
            2 * p1
            + (p2 - p0) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
            + (3 * p1 - p0 - 3 * p2 + p3) * t * t * t
        )

    inside = (1 - q) * along_w(j) + q * along_w(j + 1)
    return torch.relu(inside + excess - shortfall - issuance_cost(p, shortfall))


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
    if evaluate_steps is None:
        # Policy evaluation contracts by beta per step, so the number of steps
        # it needs grows like 1 / (1 - beta): about 50 annually, about 200
        # quarterly. A fixed 30 left the quarterly solve at its iteration cap,
        # still 3e-6 from converged.
        evaluate_steps = max(30, math.ceil(3.0 / (1.0 - p.beta)))
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
    # iterations; the product and hedge controls ran to their bounds there and
    # stayed, because a saturated sigmoid or tanh has almost no gradient.
    V = (grid.w - spec.w_lo).unsqueeze(-1).expand(-1, spec.n_s).clone()
    fresh = RawPolicy.initial(p, n, profile)
    policy = fresh
    history = []

    def bellman(V_now, c):
        return torch.relu(continuation(p, grid, rule, jumps, V_now, w, s, c)).reshape(spec.n_w, spec.n_s)

    def ascend(V_now, start: RawPolicy, steps: int) -> tuple[RawPolicy, torch.Tensor]:
        return ascend_at(V_now, start, steps, lr)

    def ascend_at(V_now, start: RawPolicy, steps: int, rate: float) -> tuple[RawPolicy, torch.Tensor]:
        """Adam on every grid point's controls at once; returns the policy and its value."""
        params = [z.detach().clone().requires_grad_(True) for z in start.tensors()]
        optimizer = torch.optim.Adam(params, lr=rate)
        for _ in range(steps):
            optimizer.zero_grad()
            c = RawPolicy(*params).controls(p, grid)
            (-continuation(p, grid, rule, jumps, V_now, w, s, c).sum()).backward()
            optimizer.step()
        found = RawPolicy(*[z.detach() for z in params])
        with torch.no_grad():
            value = continuation(p, grid, rule, jumps, V_now, w, s, found.controls(p, grid))
        return found, value

    for it in range(max_iterations):
        # -- improvement, from two starts, keeping the better at each grid point.
        # The warm start carries the last policy forward, which is what makes
        # each step cheap. The fresh start is the escape route: a control whose
        # squashing function has saturated at a bound cannot be pulled back by
        # its own gradient, so without a second start a bad early step is
        # permanent.
        steps = first_improve_steps if it == 0 else improve_steps
        warm, warm_value = ascend(V, policy, steps)
        if it == 0:
            policy = warm
        else:
            restart, restart_value = ascend(V, fresh, steps)
            better = restart_value > warm_value
            policy = RawPolicy(*[
                torch.where(better, r, k) for r, k in zip(restart.tensors(), warm.tensors())
            ])

        # -- evaluation: apply the fixed policy's Bellman operator repeatedly
        with torch.no_grad():
            c = policy.controls(p, grid)
            for _ in range(evaluate_steps):
                V_next = bellman(V, c)
                residual = float((V_next - V).abs().max())
                V = V_next
        history.append(residual)
        if verbose:
            print(f"iteration {it:3d}  residual {residual:.3e}  max V {float(V.max()):.6f}")
        if residual < tol and it > 0:
            break

    # -- polish: one long, slow ascent from the converged policy, then a final
    # evaluation. The value is nearly flat in some controls near their optimum
    # -- a hedge error of 0.03 costs about 2e-6 -- so Adam at the working rate
    # jitters by about its step size there. A smaller rate pins them down.
    policy, _ = ascend_at(V, policy, polish_steps, lr / 10.0)
    with torch.no_grad():
        c = policy.controls(p, grid)
        for _ in range(evaluate_steps):
            V_next = bellman(V, c)
            residual = float((V_next - V).abs().max())
            V = V_next

    return Solution(
        params=p, grid=grid, rule=rule, jumps=jumps, V=V, policy=policy,
        residual=residual, iterations=it + 1, history=history, inputs=inputs,
    )
