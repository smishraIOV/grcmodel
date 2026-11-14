"""The BTC yield vault on an infinite horizon: a 3-D grid over (w, S, A)
(design.md sections 2.3, 3 and 5).

    V(w, S, A) = max{ 0, max_c [ flow(w, K) + beta E V(w', S', A') ] }

over the firm's equity, its GRC stock, and the BTC vault's size in dollars.
The grid, interpolation and policy iteration are `quant.grid`'s; this module
supplies the decisions and the objective. The vault's size beyond the grid is
clamped.

Decisions, squashed so every constraint holds by construction:

    g     = delta S_hi sigmoid(z_g)                   keeps S' on the grid
    q     = q_max sigmoid(z_q)                        the firm's own BTC
    K     = floor(A, g, q) + softplus(z_K)            own funds, runway, no negative equity
    H     = hedge_max sigmoid(z_h) (q + co A)         hedge at most the BTC the firm owns
    alloc = softmax(z_1, z_2, 0)                      lending, basis, cold custody

The firm decides with its BTC bias; `market_value` re-evaluates the policy at
the market's expectation.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from quant.btc_vault import (
    BtcVaultControls,
    btc_held,
    btc_vault_capital_floor,
    btc_vault_outcomes,
    btc_vault_rule,
)
from quant.core import issuance_cost, next_grc, shareholder_flow
from quant.grid import StateGrid, policy_iteration
from quant.numerics import NumericsProfile
from quant.params import Params
from quant.quadrature import JumpRule, ShockRule, pareto_jump_rule


@dataclass(frozen=True)
class BtcVaultSpec:
    w_lo: float = -0.5
    w_hi: float = 0.4
    n_w: int = 37
    s_hi: float = 0.08
    n_s: int = 5
    A_lo: float = 0.1
    A_hi: float = 1.5
    n_A: int = 9
    n_strategy: int = 2     # quadrature points for each active strategy's yield shock
    n_btc: int = 7          # and for BTC's smooth move
    n_mixing: int = 6
    n_crash: int = 5        # BTC crash sizes
    n_jump: int = 5         # incident sizes
    hedge_max: float = 1.0


@dataclass
class BtcVaultRaw:
    """Unconstrained parameters, one per decision per grid point."""

    z_g: torch.Tensor
    z_K: torch.Tensor
    z_q: torch.Tensor
    z_h: torch.Tensor
    z_1: torch.Tensor
    z_2: torch.Tensor

    @classmethod
    def initial(cls, n: int, profile: NumericsProfile) -> "BtcVaultRaw":
        """Little GRC, a thin buffer, little BTC mostly hedged, equal shares."""
        def full(x):
            return torch.full((n,), x, dtype=profile.dtype, device=profile.device)

        return cls(
            z_g=full(-2.0), z_K=full(-3.0), z_q=full(-4.0), z_h=full(1.3),
            z_1=full(0.0), z_2=full(0.0),
        )

    def tensors(self) -> list[torch.Tensor]:
        return [self.z_g, self.z_K, self.z_q, self.z_h, self.z_1, self.z_2]

    def controls(self, p: Params, spec: BtcVaultSpec, A: torch.Tensor) -> BtcVaultControls:
        g = p.delta * spec.s_hi * torch.sigmoid(self.z_g)
        q = p.q_max * torch.sigmoid(self.z_q)
        K = btc_vault_capital_floor(p, A, g, q) + F.softplus(self.z_K)
        H = spec.hedge_max * torch.sigmoid(self.z_h) * btc_held(p, A, q)
        alloc = torch.softmax(
            torch.stack([self.z_1, self.z_2, torch.zeros_like(self.z_1)], dim=-1), dim=-1
        )
        return BtcVaultControls(K=K, g=g, q=q, H=H, alloc=alloc)


def btc_vault_continuation(
    p: Params, grid: StateGrid, rule: ShockRule, jumps: JumpRule | None,
    V: torch.Tensor, w: torch.Tensor, s: torch.Tensor, A: torch.Tensor,
    c: BtcVaultControls, bias: float | None = None,
) -> torch.Tensor:
    """flow(w, K) + beta E V(w', S', A'), before the walk-away floor."""
    s_next = next_grc(p, s, c.g)
    out = btc_vault_outcomes(p, A, c, s_next, rule, jumps, bias=bias)
    future = out.expect(
        grid.interpolate(V, [out.wealth, s_next.unsqueeze(-1), out.aum], lambda x: issuance_cost(p, x))
    )
    return shareholder_flow(p, w, c.K) + p.beta * future


@dataclass
class BtcVaultSolution:
    params: Params                  # per period
    spec: BtcVaultSpec
    grid: StateGrid
    rule: ShockRule
    jumps: JumpRule | None
    V: torch.Tensor                 # (n_w, n_s, n_A), at the firm's own expectation
    raw: BtcVaultRaw
    residual: float
    iterations: int
    history: list[float] = field(default_factory=list)
    inputs: Params | None = None    # as given, rates per year

    def states(self) -> list[torch.Tensor]:
        return self.grid.states()

    def controls(self) -> BtcVaultControls:
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


def solve_btc_vault(
    p: Params,
    profile: NumericsProfile,
    spec: BtcVaultSpec = BtcVaultSpec(),  # noqa: B008
    **iteration_args,
) -> BtcVaultSolution:
    inputs = p
    p = p.per_period()
    grid = StateGrid.uniform(
        [(spec.w_lo, spec.w_hi, spec.n_w), (0.0, spec.s_hi, spec.n_s), (spec.A_lo, spec.A_hi, spec.n_A)],
        profile,
    )
    rule = btc_vault_rule(
        p, profile, n_strategy=spec.n_strategy, n_btc=spec.n_btc,
        n_mixing=spec.n_mixing, n_crash=spec.n_crash,
    )
    jumps = (
        pareto_jump_rule(p.jump_mean, p.jump_tail, profile, n=spec.n_jump)
        if p.jump_rate > 0 else None
    )
    w, s, A = grid.states()
    # Start risk-neutral: linear in equity, flat in the other axes.
    V0 = (w - spec.w_lo).reshape(grid.shape).clone()

    def objective(V, raw):
        c = BtcVaultRaw(*raw).controls(p, spec, A)
        return btc_vault_continuation(p, grid, rule, jumps, V, w, s, A, c)

    result = policy_iteration(
        objective, BtcVaultRaw.initial(w.shape[0], profile).tensors(), V0, p.beta, **iteration_args
    )
    return BtcVaultSolution(
        params=p, spec=spec, grid=grid, rule=rule, jumps=jumps, V=result.V,
        raw=BtcVaultRaw(*result.raw), residual=result.residual, iterations=result.iterations,
        history=result.history, inputs=inputs,
    )


def market_value(sol: BtcVaultSolution, tol: float = 1e-10, max_steps: int = 20000) -> torch.Tensor:
    """The solved policy's value at the market's expectation of BTC (no bias):
    the decisions, walk-away included, stay those made with the bias."""
    p, grid = sol.params, sol.grid
    w, s, A = grid.states()
    with torch.no_grad():
        c = sol.controls()
        walked = (sol.V == 0).reshape(-1)
        V = sol.V.clone()
        for _ in range(max_steps):
            value = btc_vault_continuation(p, grid, sol.rule, sol.jumps, V, w, s, A, c, bias=0.0)
            V_next = torch.where(walked, torch.zeros_like(value), value).reshape(grid.shape)
            gap = float((V_next - V).abs().max())
            V = V_next
            if gap < tol:
                break
    return V


@dataclass(frozen=True)
class RisklessBtcVault:
    """The exact answer when nothing is random (an oracle).

    No shocks, no crashes, no incidents, inert GRC, no bias, and every
    strategy with the same BTC yield y >= 0. BTC's price then rises by r_E a
    period for sure. Investors' BTC return is R = y - f_m - f_p y, and the
    vault's dollar size settles where growth and flows balance:

        A* = theta cap / (theta + 1 - (1 + R)(1 + r_E) - eta (R - y_out)).

    The firm holds no BTC of its own (it would earn r_E less its security
    cost, below the r_E shareholders require), no hedge (hedging gives up
    r_E - r for nothing), and exactly its floor K* = co A* + runway opex,
    paying out the rest. It closes each period at

        w_ss = (K* - co A*) - opex + (f_m + f_p y + co (1 + R)) A* (1 + r_E),

    so V(w, A*) = w - K* + beta (w_ss - K*) / (1 - beta) for w >= K*.
    Assumes the runway constraint, not the own-funds requirement, sets the
    floor (co A* + runway opex >= k_A A*).
    """

    A: float
    K: float
    w_ss: float
    beta: float

    def value(self, w: torch.Tensor) -> torch.Tensor:
        return w - self.K + self.beta * (self.w_ss - self.K) / (1.0 - self.beta)


def riskless_btc_vault(p: Params) -> RisklessBtcVault:
    p = p.per_period()
    net = [m - l for m, l in zip(p.strategy_yield, p.strategy_loss, strict=True)]
    if (
        any(v != 0.0 for v in p.strategy_vol) or p.s_BTC != 0.0 or p.btc_crash_rate != 0.0
        or p.jump_rate != 0.0 or p.a_A != 0.0 or p.btc_bias != 0.0
        or max(net) != min(net) or net[0] < 0.0 or p.tau != 0.0
    ):
        raise ValueError(
            "the riskless vault needs no volatility, crashes or incidents, no bias, "
            "inert GRC, no tau and equal non-negative net strategy yields"
        )
    y = net[0]
    R = y - p.fee_mgmt - p.fee_perf * y
    growth = (1.0 + R) * (1.0 + p.r_E) - 1.0 + p.btc_flow_yield * (R - p.btc_outside_yield)
    if not p.inflow_speed > growth:
        raise ValueError("the vault would grow without bound: inflow_speed must exceed its growth")
    A = p.inflow_speed * p.btc_capacity / (p.inflow_speed - growth)
    co = p.btc_co_invest
    K = co * A + p.runway * p.opex
    if K < p.k_A * A:
        raise ValueError("the oracle assumes the runway, not own funds, sets the floor")
    w_ss = (K - co * A) - p.opex + (p.fee_mgmt + p.fee_perf * y + co * (1.0 + R)) * A * (1.0 + p.r_E)
    return RisklessBtcVault(A=A, K=K, w_ss=w_ss, beta=p.beta)
