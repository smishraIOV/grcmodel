"""The vault form on an infinite horizon: a 3-D grid over (w, S, A)
(design.md sections 4 and 5).

    V(w, S, A) = max{ 0, max_c [ flow(w, K) + beta E V(w', S', A') ] }

over the firm's equity, its GRC stock, and the vault's assets under
management. The grid, interpolation and policy iteration are `quant.grid`'s;
this module supplies the vault's controls and objective. AUM beyond the grid
is clamped to its edge.

Controls, squashed so every constraint holds by construction:

    g     = delta S_hi sigmoid(z_g)                 keeps S' on the grid
    q     = q_max sigmoid(z_q)                      the firm's BTC treasury, bought spot
    K     = floor(A, g, q) + softplus(z_K)          own funds, cash, no negative equity
    alloc = softmax(z_1, z_2, 0)                    the vault's three strategy shares
    H     = 1.2 sigmoid(z_h) (q + A sum_i a_i b_i)  hedge the treasury and the franchise

The hedge covers the firm's own BTC and the BTC sensitivity of the vault it
earns fees on. As in the bank form it may run slightly past a full hedge, so
that a full hedge is an interior point.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from quant.grid import StateGrid, policy_iteration
from quant.model import issuance_cost, next_grc, shareholder_flow
from quant.numerics import NumericsProfile
from quant.params import Params
from quant.quadrature import JumpRule, ShockRule, pareto_jump_rule, student_t_rule
from quant.vault import (
    VaultControls,
    vault_btc_sensitivity,
    vault_capital_floor,
    vault_outcomes,
    vault_shock_correlation,
)

HEDGE_MAX = 1.2


@dataclass(frozen=True)
class VaultSpec:
    w_lo: float = -0.3
    w_hi: float = 0.5
    n_w: int = 33
    s_hi: float = 0.08
    n_s: int = 5
    A_lo: float = 0.1
    A_hi: float = 1.5
    n_A: int = 9
    n_hermite: int = 3
    n_mixing: int = 6
    n_jump: int = 6


@dataclass
class VaultRaw:
    """Unconstrained parameters, one per control per grid point."""

    z_g: torch.Tensor
    z_K: torch.Tensor
    z_q: torch.Tensor
    z_h: torch.Tensor
    z_1: torch.Tensor
    z_2: torch.Tensor

    @classmethod
    def initial(cls, n: int, profile: NumericsProfile) -> "VaultRaw":
        """Little GRC, a thin buffer, little BTC, mostly hedged, equal shares."""
        def full(x):
            return torch.full((n,), x, dtype=profile.dtype, device=profile.device)

        return cls(
            z_g=full(-2.0), z_K=full(-3.0), z_q=full(-4.0), z_h=full(1.3),
            z_1=full(0.0), z_2=full(0.0),
        )

    def tensors(self) -> list[torch.Tensor]:
        return [self.z_g, self.z_K, self.z_q, self.z_h, self.z_1, self.z_2]

    def controls(self, p: Params, spec: VaultSpec, A: torch.Tensor) -> VaultControls:
        g = p.delta * spec.s_hi * torch.sigmoid(self.z_g)
        q = p.q_max * torch.sigmoid(self.z_q)
        K = vault_capital_floor(p, A, g, q) + F.softplus(self.z_K)
        alloc = torch.softmax(
            torch.stack([self.z_1, self.z_2, torch.zeros_like(self.z_1)], dim=-1), dim=-1
        )
        gross = q + vault_btc_sensitivity(p, A, alloc)
        H = HEDGE_MAX * torch.sigmoid(self.z_h) * gross
        return VaultControls(K=K, g=g, q=q, H=H, alloc=alloc)


def vault_continuation(
    p: Params, spec: VaultSpec, grid: StateGrid, rule: ShockRule, jumps: JumpRule | None,
    V: torch.Tensor, w: torch.Tensor, s: torch.Tensor, A: torch.Tensor, c: VaultControls,
) -> torch.Tensor:
    """flow(w, K) + beta E V(w', S', A'), before the walk-away floor."""
    s_next = next_grc(p, s, c.g)
    out = vault_outcomes(p, A, c, s_next, rule, jumps)
    future = out.expect(
        grid.interpolate(V, [out.wealth, s_next.unsqueeze(-1), out.aum], lambda x: issuance_cost(p, x))
    )
    return shareholder_flow(p, w, c.K) + p.beta * future


@dataclass
class VaultSolution:
    params: Params                  # per period
    spec: VaultSpec
    grid: StateGrid
    rule: ShockRule
    jumps: JumpRule | None
    V: torch.Tensor                 # (n_w, n_s, n_A)
    raw: VaultRaw
    residual: float
    iterations: int
    history: list[float] = field(default_factory=list)
    inputs: Params | None = None    # as given, rates per year

    def states(self) -> list[torch.Tensor]:
        return self.grid.states()

    def controls(self) -> VaultControls:
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


def solve_vault(
    p: Params,
    profile: NumericsProfile,
    spec: VaultSpec = VaultSpec(),
    **iteration_args,
) -> VaultSolution:
    inputs = p
    p = p.per_period()
    grid = StateGrid.uniform(
        [(spec.w_lo, spec.w_hi, spec.n_w), (0.0, spec.s_hi, spec.n_s), (spec.A_lo, spec.A_hi, spec.n_A)],
        profile,
    )
    rule = student_t_rule(
        vault_shock_correlation(p), p.nu, profile, n_hermite=spec.n_hermite, n_mixing=spec.n_mixing
    )
    jumps = (
        pareto_jump_rule(p.jump_mean, p.jump_tail, profile, n=spec.n_jump)
        if p.jump_rate > 0 else None
    )
    w, s, A = grid.states()
    # Start risk-neutral: linear in equity, flat in the other axes.
    V0 = (w - spec.w_lo).reshape(grid.shape).clone()

    def objective(V, raw):
        c = VaultRaw(*raw).controls(p, spec, A)
        return vault_continuation(p, spec, grid, rule, jumps, V, w, s, A, c)

    result = policy_iteration(
        objective, VaultRaw.initial(w.shape[0], profile).tensors(), V0, p.beta, **iteration_args
    )
    return VaultSolution(
        params=p, spec=spec, grid=grid, rule=rule, jumps=jumps, V=result.V,
        raw=VaultRaw(*result.raw), residual=result.residual, iterations=result.iterations,
        history=result.history, inputs=inputs,
    )


@dataclass(frozen=True)
class RisklessVault:
    """The exact answer when nothing is random (an oracle).

    With no shocks, inert GRC, no view on BTC and every strategy earning the
    same net excess return R >= 0, AUM settles where inflows balance growth
    net of fees,

        A* = theta capacity / (theta - (r + R - f_m - f_p R)),

    the firm holds exactly its own-funds floor K* = k_A A*, and pays out the
    rest every period. Its equity closes at w_ss = (1 + r - tau) K* + fees - opex
    each period, so

        V(w, A*) = w - K* + beta (w_ss - K*) / (1 - beta)    for w >= K*.
    """

    A: float
    K: float
    w_ss: float
    beta: float

    def value(self, w: torch.Tensor) -> torch.Tensor:
        return w - self.K + self.beta * (self.w_ss - self.K) / (1.0 - self.beta)


def riskless_vault(p: Params) -> RisklessVault:
    p = p.per_period()
    net = [m - l for m, l in zip(p.vault_mu, p.vault_loss)]
    if (
        any(v != 0.0 for v in p.vault_vol) or p.s_BTC != 0.0 or p.mu_BTC != 0.0
        or p.jump_rate != 0.0 or p.a_A != 0.0 or max(net) != min(net) or net[0] < 0.0
    ):
        raise ValueError(
            "the riskless vault needs no volatility, no jumps, no view on BTC, "
            "inert GRC and equal non-negative net strategy returns"
        )
    R = net[0]
    growth = p.r + R - p.fee_mgmt - p.fee_perf * R
    if not p.inflow_speed > growth:
        raise ValueError("AUM would grow without bound: inflow_speed must exceed net growth")
    A = p.inflow_speed * p.capacity / (p.inflow_speed - growth)
    K = p.k_A * A
    fees = p.fee_mgmt * A + p.fee_perf * R * A
    w_ss = (1.0 + p.r - p.tau) * K + fees - p.opex
    return RisklessVault(A=A, K=K, w_ss=w_ss, beta=p.beta)
