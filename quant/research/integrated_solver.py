"""The integrated firm on an infinite horizon: a coarse 4-D grid over
(w, S, A_L, A_B) (design.md sections 2, 3 and 5; docs/plan-integrated.md
step 4).

    V(w, S, A_L, A_B) = max{ 0, max_c [ flow(w, K) + beta E V(w', S', A_L', A_B') ] }

over the firm's equity, its GRC stock, and the two vaults' sizes in dollars.
The grid, interpolation and policy iteration are `quant.research.grid`'s; this module
supplies the decisions and the objective. A grid in four dimensions is coarse
at any affordable size: it is the deterministic reference the learner is
graded against, and a first look at the integrated firm, not the final solver.

Decisions, squashed so every constraint holds by construction:

    g     = delta S_hi sigmoid(z_g)                   keeps S' on the grid
    q     = q_max sigmoid(z_q)                        the firm's own BTC
    K     = floor(A_L, A_B, g, q) + softplus(z_K)     own funds, runway, no negative equity
    H     = hedge_max sigmoid(z_h) (q + co_B A_B)     hedge at most the BTC the firm owns
    rate  = rate_max sigmoid(z_r)                     the borrower rate
    ltv   = ltv_min + (ltv_max - ltv_min) sigmoid(z_l)
    alloc = softmax(z_1, z_2, 0, z_4)                 lending BTC, basis, cold custody, the sleeve

The firm decides with its BTC bias; `market_value` re-evaluates the policy at
the market's expectation.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from quant.core import issuance_cost, next_grc, shareholder_flow
from quant.research.grid import StateGrid, policy_iteration
from quant.integrated import (
    IntegratedControls,
    IntegratedOutcomes,
    integrated_btc_held,
    integrated_capital_floor,
    integrated_outcomes,
    integrated_rule,
)
from quant.numerics import NumericsProfile
from quant.params import Params
from quant.quadrature import JumpRule, ShockRule, pareto_jump_rule


@dataclass(frozen=True)
class IntegratedSpec:
    w_lo: float = -0.5
    w_hi: float = 0.7
    n_w: int = 17
    s_hi: float = 0.08
    n_s: int = 3
    L_lo: float = 0.1       # the dollar vault's size
    L_hi: float = 1.5
    n_L: int = 5
    B_lo: float = 0.2       # the BTC vault's size
    B_hi: float = 3.0
    n_B: int = 5
    n_ops: int = 2          # quadrature points for the lending system's operating-loss shock
    n_strategy: int = 2     # for each active BTC strategy's yield shock
    n_btc: int = 7          # and for BTC's smooth move
    n_mixing: int = 4
    n_crash: int = 5        # BTC crash sizes
    n_jump: int = 3         # incident sizes, per vault
    hedge_max: float = 1.0


@dataclass
class IntegratedRaw:
    """Unconstrained parameters, one per decision per grid point."""

    z_g: torch.Tensor
    z_K: torch.Tensor
    z_q: torch.Tensor
    z_h: torch.Tensor
    z_r: torch.Tensor
    z_l: torch.Tensor
    z_1: torch.Tensor
    z_2: torch.Tensor
    z_4: torch.Tensor

    @classmethod
    def initial(cls, n: int, profile: NumericsProfile) -> "IntegratedRaw":
        """Little GRC, a thin buffer, little BTC mostly hedged, mid-range
        terms, equal shares."""
        def full(x):
            return torch.full((n,), x, dtype=profile.dtype, device=profile.device)

        return cls(
            z_g=full(-2.0), z_K=full(-3.0), z_q=full(-4.0), z_h=full(1.3),
            z_r=full(0.0), z_l=full(0.0), z_1=full(0.0), z_2=full(0.0), z_4=full(0.0),
        )

    def tensors(self) -> list[torch.Tensor]:
        return [self.z_g, self.z_K, self.z_q, self.z_h, self.z_r, self.z_l, self.z_1, self.z_2, self.z_4]

    def controls(
        self, p: Params, spec: IntegratedSpec, A_L: torch.Tensor, A_B: torch.Tensor
    ) -> IntegratedControls:
        g = p.delta * spec.s_hi * torch.sigmoid(self.z_g)
        q = p.q_max * torch.sigmoid(self.z_q)
        K = integrated_capital_floor(p, A_L, A_B, g, q) + F.softplus(self.z_K)
        H = spec.hedge_max * torch.sigmoid(self.z_h) * integrated_btc_held(p, A_B, q)
        rate = p.rate_max * torch.sigmoid(self.z_r)
        ltv = p.ltv_min + (p.ltv_max - p.ltv_min) * torch.sigmoid(self.z_l)
        alloc = torch.softmax(
            torch.stack([self.z_1, self.z_2, torch.zeros_like(self.z_1), self.z_4], dim=-1), dim=-1
        )
        return IntegratedControls(K=K, g=g, q=q, H=H, rate=rate, ltv=ltv, alloc=alloc)


def integrated_continuation(
    p: Params, grid: StateGrid, rule: ShockRule, jumps: JumpRule | None,
    V: torch.Tensor, w: torch.Tensor, s: torch.Tensor, A_L: torch.Tensor, A_B: torch.Tensor,
    c: IntegratedControls, bias: float | None = None,
) -> torch.Tensor:
    """flow(w, K) + beta E V(w', S', A_L', A_B'), before the walk-away floor."""
    s_next = next_grc(p, s, c.g)
    out = integrated_outcomes(p, A_L, A_B, c, s_next, rule, jumps, bias=bias)
    future = out.expect(
        grid.interpolate(
            V, [out.wealth, s_next.unsqueeze(-1), out.aum_l, out.aum_b], lambda x: issuance_cost(p, x)
        )
    )
    return shareholder_flow(p, w, c.K) + p.beta * future


@dataclass
class IntegratedSolution:
    params: Params                  # per period
    spec: IntegratedSpec
    grid: StateGrid
    rule: ShockRule
    jumps: JumpRule | None
    V: torch.Tensor                 # (n_w, n_s, n_L, n_B), at the firm's own expectation
    raw: IntegratedRaw
    residual: float
    iterations: int
    history: list[float] = field(default_factory=list)
    inputs: Params | None = None    # as given, rates per year

    def states(self) -> list[torch.Tensor]:
        return self.grid.states()

    def controls(self) -> IntegratedControls:
        _, _, A_L, A_B = self.grid.states()
        return self.raw.controls(self.params, self.spec, A_L, A_B)

    def outcomes(self, bias: float | None = None) -> IntegratedOutcomes:
        """Every outcome of the solved policy, at every grid point."""
        _, s, A_L, A_B = self.grid.states()
        with torch.no_grad():
            c = self.controls()
            return integrated_outcomes(
                self.params, A_L, A_B, c, next_grc(self.params, s, c.g), self.rule, self.jumps, bias=bias
            )

    @property
    def lower_edge_is_walk_away(self) -> bool:
        return bool((self.V[0] == 0).all())

    @property
    def upper_edge_pays_out(self) -> bool:
        w = self.grid.states()[0]
        K = self.controls().K.reshape(self.grid.shape)
        return bool((K[-1] < w.reshape(self.grid.shape)[-1]).all())


def integrated_grid(spec: IntegratedSpec, profile: NumericsProfile) -> StateGrid:
    return StateGrid.uniform(
        [
            (spec.w_lo, spec.w_hi, spec.n_w), (0.0, spec.s_hi, spec.n_s),
            (spec.L_lo, spec.L_hi, spec.n_L), (spec.B_lo, spec.B_hi, spec.n_B),
        ],
        profile,
    )


def integrated_rules(p: Params, spec: IntegratedSpec, profile: NumericsProfile):
    rule = integrated_rule(
        p, profile, n_ops=spec.n_ops, n_strategy=spec.n_strategy, n_btc=spec.n_btc,
        n_mixing=spec.n_mixing, n_crash=spec.n_crash,
    )
    jumps = (
        pareto_jump_rule(p.jump_mean, p.jump_tail, profile, n=spec.n_jump)
        if p.jump_rate > 0 else None
    )
    return rule, jumps


def solve_integrated(
    p: Params,
    profile: NumericsProfile,
    spec: IntegratedSpec = IntegratedSpec(),  # noqa: B008
    **iteration_args,
) -> IntegratedSolution:
    inputs = p
    p = p.per_period()
    grid = integrated_grid(spec, profile)
    rule, jumps = integrated_rules(p, spec, profile)
    w, s, A_L, A_B = grid.states()
    # Start risk-neutral: linear in equity, flat in the other axes.
    V0 = (w - spec.w_lo).reshape(grid.shape).clone()

    def objective(V, raw):
        c = IntegratedRaw(*raw).controls(p, spec, A_L, A_B)
        return integrated_continuation(p, grid, rule, jumps, V, w, s, A_L, A_B, c)

    result = policy_iteration(
        objective, IntegratedRaw.initial(w.shape[0], profile).tensors(), V0, p.beta, **iteration_args
    )
    return IntegratedSolution(
        params=p, spec=spec, grid=grid, rule=rule, jumps=jumps, V=result.V,
        raw=IntegratedRaw(*result.raw), residual=result.residual, iterations=result.iterations,
        history=result.history, inputs=inputs,
    )


def market_value(sol: IntegratedSolution, tol: float = 1e-10, max_steps: int = 20000) -> torch.Tensor:
    """The solved policy's value at the market's expectation of BTC (no bias):
    the decisions, walk-away included, stay those made with the bias."""
    p, grid = sol.params, sol.grid
    w, s, A_L, A_B = grid.states()
    with torch.no_grad():
        c = sol.controls()
        walked = (sol.V == 0).reshape(-1)
        V = sol.V.clone()
        for _ in range(max_steps):
            value = integrated_continuation(p, grid, sol.rule, sol.jumps, V, w, s, A_L, A_B, c, bias=0.0)
            V_next = torch.where(walked, torch.zeros_like(value), value).reshape(grid.shape)
            gap = float((V_next - V).abs().max())
            V = V_next
            if gap < tol:
                break
    return V
