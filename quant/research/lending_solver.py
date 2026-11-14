"""The lending vault on an infinite horizon: a 3-D grid over (w, S, A)
(design.md sections 2.2, 3 and 5).

    V(w, S, A) = max{ 0, max_c [ flow(w, K) + beta E V(w', S', A') ] }

over the firm's equity, its GRC stock, and the lending vault's size. The grid,
interpolation and policy iteration are `quant.research.grid`'s; this module supplies
the decisions and the objective. The vault's size beyond the grid is clamped.

Decisions, squashed so every constraint holds by construction:

    g    = delta S_hi sigmoid(z_g)                   keeps S' on the grid
    q    = q_max sigmoid(z_q)                        the firm's own BTC
    K    = floor(A, g, q) + softplus(z_K)            own funds, cash, no negative equity
    H    = hedge_max sigmoid(z_h) q                  hedge at most the BTC held
    rate = rate_max sigmoid(z_r)                     the borrower rate
    ltv  = ltv_min + (ltv_max - ltv_min) sigmoid(z_l)

`hedge_max` is 1 in the model (design.md section 3.4). The Prop. 2 test alone
raises it to 1.2, so that a full hedge is an interior point rather than a
bound the optimizer can only approach.

The firm decides with its BTC bias. `market_value` then evaluates the solved
policy with no bias: the firm's value at the market's expectation.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from quant.core import issuance_cost, next_grc, shareholder_flow
from quant.research.grid import StateGrid, policy_iteration
from quant.lending import (
    LendingControls,
    lending_capital_floor,
    lending_outcomes,
    lending_rule,
)
from quant.numerics import NumericsProfile
from quant.params import Params
from quant.quadrature import JumpRule, ShockRule, pareto_jump_rule


@dataclass(frozen=True)
class LendingSpec:
    w_lo: float = -0.5
    w_hi: float = 0.4
    n_w: int = 37
    s_hi: float = 0.08
    n_s: int = 5
    A_lo: float = 0.1
    A_hi: float = 1.5
    n_A: int = 9
    n_ops: int = 3          # quadrature points for the operating-loss shock
    n_btc: int = 7          # and for BTC's smooth move
    n_mixing: int = 6
    n_crash: int = 5        # BTC crash sizes
    n_jump: int = 5         # incident sizes
    hedge_max: float = 1.0


@dataclass
class LendingRaw:
    """Unconstrained parameters, one per decision per grid point."""

    z_g: torch.Tensor
    z_K: torch.Tensor
    z_q: torch.Tensor
    z_h: torch.Tensor
    z_r: torch.Tensor
    z_l: torch.Tensor

    @classmethod
    def initial(cls, n: int, profile: NumericsProfile) -> "LendingRaw":
        """Little GRC, a thin buffer, little BTC mostly hedged, mid-range terms."""
        def full(x):
            return torch.full((n,), x, dtype=profile.dtype, device=profile.device)

        return cls(
            z_g=full(-2.0), z_K=full(-3.0), z_q=full(-4.0), z_h=full(1.3),
            z_r=full(0.0), z_l=full(0.0),
        )

    def tensors(self) -> list[torch.Tensor]:
        return [self.z_g, self.z_K, self.z_q, self.z_h, self.z_r, self.z_l]

    def controls(self, p: Params, spec: LendingSpec, A: torch.Tensor) -> LendingControls:
        g = p.delta * spec.s_hi * torch.sigmoid(self.z_g)
        q = p.q_max * torch.sigmoid(self.z_q)
        K = lending_capital_floor(p, A, g, q) + F.softplus(self.z_K)
        H = spec.hedge_max * torch.sigmoid(self.z_h) * q
        rate = p.rate_max * torch.sigmoid(self.z_r)
        ltv = p.ltv_min + (p.ltv_max - p.ltv_min) * torch.sigmoid(self.z_l)
        return LendingControls(K=K, g=g, q=q, H=H, rate=rate, ltv=ltv)


def lending_continuation(
    p: Params, grid: StateGrid, rule: ShockRule, jumps: JumpRule | None,
    V: torch.Tensor, w: torch.Tensor, s: torch.Tensor, A: torch.Tensor,
    c: LendingControls, bias: float | None = None,
) -> torch.Tensor:
    """flow(w, K) + beta E V(w', S', A'), before the walk-away floor."""
    s_next = next_grc(p, s, c.g)
    out = lending_outcomes(p, A, c, s_next, rule, jumps, bias=bias)
    future = out.expect(
        grid.interpolate(V, [out.wealth, s_next.unsqueeze(-1), out.aum], lambda x: issuance_cost(p, x))
    )
    return shareholder_flow(p, w, c.K) + p.beta * future


@dataclass
class LendingSolution:
    params: Params                  # per period
    spec: LendingSpec
    grid: StateGrid
    rule: ShockRule
    jumps: JumpRule | None
    V: torch.Tensor                 # (n_w, n_s, n_A), at the firm's own expectation
    raw: LendingRaw
    residual: float
    iterations: int
    history: list[float] = field(default_factory=list)
    inputs: Params | None = None    # as given, rates per year

    def states(self) -> list[torch.Tensor]:
        return self.grid.states()

    def controls(self) -> LendingControls:
        _, _, A = self.grid.states()
        return self.raw.controls(self.params, self.spec, A)

    @property
    def lower_edge_is_walk_away(self) -> bool:
        return bool((self.V[0] == 0).all())

    @property
    def upper_edge_pays_out(self) -> bool:
        w, _, _ = self.grid.states()
        K = self.controls().K.reshape(self.grid.shape)
        return bool((K[-1] < w.reshape(self.grid.shape)[-1]).all())


def lending_grid(spec: LendingSpec, profile: NumericsProfile) -> StateGrid:
    return StateGrid.uniform(
        [(spec.w_lo, spec.w_hi, spec.n_w), (0.0, spec.s_hi, spec.n_s), (spec.A_lo, spec.A_hi, spec.n_A)],
        profile,
    )


def lending_rules(p: Params, spec: LendingSpec, profile: NumericsProfile):
    rule = lending_rule(
        p, profile, n_ops=spec.n_ops, n_btc=spec.n_btc, n_mixing=spec.n_mixing, n_crash=spec.n_crash
    )
    jumps = (
        pareto_jump_rule(p.jump_mean, p.jump_tail, profile, n=spec.n_jump)
        if p.jump_rate > 0 else None
    )
    return rule, jumps


def solve_lending(
    p: Params,
    profile: NumericsProfile,
    spec: LendingSpec = LendingSpec(),  # noqa: B008
    **iteration_args,
) -> LendingSolution:
    inputs = p
    p = p.per_period()
    grid = lending_grid(spec, profile)
    rule, jumps = lending_rules(p, spec, profile)
    w, s, A = grid.states()
    # Start risk-neutral: linear in equity, flat in the other axes.
    V0 = (w - spec.w_lo).reshape(grid.shape).clone()

    def objective(V, raw):
        c = LendingRaw(*raw).controls(p, spec, A)
        return lending_continuation(p, grid, rule, jumps, V, w, s, A, c)

    result = policy_iteration(
        objective, LendingRaw.initial(w.shape[0], profile).tensors(), V0, p.beta, **iteration_args
    )
    return LendingSolution(
        params=p, spec=spec, grid=grid, rule=rule, jumps=jumps, V=result.V,
        raw=LendingRaw(*result.raw), residual=result.residual, iterations=result.iterations,
        history=result.history, inputs=inputs,
    )


def market_value(sol: LendingSolution, tol: float = 1e-10, max_steps: int = 20000) -> torch.Tensor:
    """The solved policy's value at the market's expectation of BTC (no bias).

    Policy evaluation: the decisions stay those the firm made with its bias,
    walk-away included, and only the BTC expectation used to value them
    changes. Contracts by beta per step.
    """
    p, grid = sol.params, sol.grid
    w, s, A = grid.states()
    with torch.no_grad():
        c = sol.controls()
        walked = (sol.V == 0).reshape(-1)
        V = sol.V.clone()
        for _ in range(max_steps):
            value = lending_continuation(p, grid, sol.rule, sol.jumps, V, w, s, A, c, bias=0.0)
            V_next = torch.where(walked, torch.zeros_like(value), value).reshape(grid.shape)
            gap = float((V_next - V).abs().max())
            V = V_next
            if gap < tol:
                break
    return V
