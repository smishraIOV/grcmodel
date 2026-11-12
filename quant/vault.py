"""The vault form, one period: two balance sheets and the flows between them
(design.md section 4).

**The vault** belongs to depositors. It is allocated across three strategies
run by hired managers -- BTC-backed lending, market-neutral, tokenized
treasuries -- and its gains and losses pass through to depositors. Its size is
the assets under management, A.

**The firm** belongs to shareholders. It earns a management fee on A and a
performance fee on a period's positive excess return; it pays dollar operating
costs and GRC; it makes good a share of incident losses; and it holds its own
equity in stablecoins and a spot BTC treasury, hedged with short perps.

Stage A of the design (section 4.6): no high-water mark, so a performance fee
is paid on any positive period, including the recovery from an earlier loss.

Shock vector order: (xi_1, xi_2, eps_BTC) -- the two risky strategies' payoff
shocks and BTC's dollar return. All unit variance. Everything is element-wise
and branch-free, so it vectorizes over grid points, quadrature nodes and jump
outcomes at once.
"""

from dataclasses import dataclass

import numpy as np
import torch

from quant.model import jump_probability, mitigation
from quant.params import Params
from quant.quadrature import JumpRule, ShockRule


def vault_shock_correlation(p: Params) -> np.ndarray:
    """The two risky strategies correlate with each other; BTC is separate in
    the normal part and shares the tail through the Student-t's common scale."""
    return np.array(
        [
            [1.0, p.vault_rho, 0.0],
            [p.vault_rho, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@dataclass(frozen=True)
class VaultControls:
    """One period's decision, taken before the shocks.

    K: the firm's operating equity after dividends or issuance. g: GRC spend.
    q: the firm's BTC treasury, bought spot. H: BTC hedged with short perps.
    alloc: the vault's shares in the three strategies, summing to one, shape
    (..., 3).
    """

    K: torch.Tensor
    g: torch.Tensor
    q: torch.Tensor
    H: torch.Tensor
    alloc: torch.Tensor


def vault_capital_floor(
    p: Params, A: torch.Tensor, g: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """The least operating equity the firm may hold.

    Own funds against AUM and the BTC treasury, K >= k_A A + k_BTC q;
    non-negative cash, K >= g + q; and no negative equity.
    """
    zero = torch.zeros_like(A)
    return torch.maximum(torch.maximum(p.k_A * A + p.k_BTC * q, g + q), zero)


def vault_btc_sensitivity(p: Params, A: torch.Tensor, alloc: torch.Tensor) -> torch.Tensor:
    """How many dollars of the vault move one-for-one with BTC: A sum_i a_i b_i.

    The firm's fee franchise inherits this exposure, which is what its perp
    hedge can offset.
    """
    return A * (alloc * alloc.new_tensor(p.vault_btc)).sum(-1)


@dataclass(frozen=True)
class VaultOutcomes:
    """Every next-period outcome of a decision, with its probability.

    `wealth`, `aum` and `weights` are (..., M) with M = N (1 + J): each
    continuous shock node, without a jump and then with each jump severity.
    `shocks` is (M, 3), the continuous shock behind each outcome. `excess` is
    the vault's excess return in each outcome.
    """

    wealth: torch.Tensor
    aum: torch.Tensor
    weights: torch.Tensor
    shocks: torch.Tensor
    excess: torch.Tensor

    def expect(self, values: torch.Tensor) -> torch.Tensor:
        return (values * self.weights).sum(-1)


def vault_outcomes(
    p: Params,
    A: torch.Tensor,
    c: VaultControls,
    S_next: torch.Tensor,
    rule: ShockRule,
    jumps: JumpRule | None,
) -> VaultOutcomes:
    """Closing firm equity and next-period AUM, over every outcome.

    The vault's excess return over the riskless rate:

        R = sum_i a_i (mu_i - phi_A m_i + phi_B s_i xi_i + b_i s_BTC eps_BTC) - J sum_i a_i j_i

    with J the jump severity (zero without a jump). Then

        fees   = f_m A + f_p max(R, 0) A
        A'     = A (1 + r + R) - fees + theta (capacity - A)
                 - kappa max(-R, 0) A - [jump] omega A                floored at 0
        w'     = (1 + r)(K - g) - tau K + x (mu_BTC + s_BTC eps_BTC) - c_H/2 H^2
                 + fees - opex - psi A J sum_i a_i j_i

    where x = q - H is the firm's own net BTC position. Cash and the BTC
    treasury's riskless part are (1 + r)(K - g), as in the bank form.
    """
    n_shock = rule.nodes.shape[0]
    xi = rule.nodes[:, :2]                       # (N, 2) the risky strategies
    eps = rule.nodes[:, 2]                       # (N,)   BTC

    def col(x):
        return x.unsqueeze(-1)

    a = c.alloc                                  # (..., 3)
    S_col = col(S_next)
    phi_A = mitigation(S_col, p.a_A, p.rho_A)
    phi_B = mitigation(S_col, p.a_B, p.rho_B)
    mu = a.new_tensor(p.vault_mu)
    loss = a.new_tensor(p.vault_loss)
    vol = a.new_tensor(p.vault_vol)
    btc = a.new_tensor(p.vault_btc)
    exposure = a.new_tensor(p.vault_jump)

    drift = (a * (mu - phi_A * loss)).sum(-1, keepdim=True)          # (..., 1)
    risky = phi_B * (a[..., :2] * vol[:2]) @ xi.T                    # (..., N)
    beta = (a * btc).sum(-1, keepdim=True) * p.s_BTC * eps           # (..., N)
    R_cont = drift + risky + beta                                    # (..., N)

    if jumps is not None and p.jump_rate > 0.0:
        severity = torch.cat([jumps.severity.new_zeros(1), jumps.severity])    # (1 + J,)
        hit = (a * exposure).sum(-1, keepdim=True).unsqueeze(-1) * severity    # (..., 1, 1 + J)
        prob = col(jump_probability(p, S_next))                                 # (..., 1)
        branch = torch.cat([1.0 - prob, prob * jumps.weights], dim=-1)          # (..., 1 + J)
        occurred = (severity > 0).to(a.dtype)                                   # (1 + J,)
    else:
        hit = torch.zeros_like(R_cont).unsqueeze(-1)[..., :1]                   # (..., 1, 1)
        branch = torch.ones_like(drift)                                         # (..., 1)
        occurred = drift.new_zeros(1)

    R = R_cont.unsqueeze(-1) - hit                                   # (..., N, 1 + J)
    A3 = A[..., None, None]
    fees = p.fee_mgmt * A3 + p.fee_perf * torch.relu(R) * A3
    aum = (
        A3 * (1.0 + p.r + R)
        - fees
        + p.inflow_speed * (p.capacity - A3)
        - p.outflow_perf * torch.relu(-R) * A3
        - occurred * p.outflow_incident * A3
    )
    aum = torch.relu(aum)

    x = c.q - c.H
    own = (
        (1.0 + p.r) * (c.K - c.g)
        - p.tau * c.K
        - 0.5 * p.c_H * c.H * c.H
        - p.opex
    )[..., None, None]
    btc_pnl = (x[..., None] * (p.mu_BTC + p.s_BTC * eps)).unsqueeze(-1)       # (..., N, 1)
    wealth = own + btc_pnl + fees - p.compensation * hit * A3

    weights = rule.weights[:, None] * branch.unsqueeze(-2)                    # (..., N, 1 + J)
    m = n_shock * branch.shape[-1]
    shocks = rule.nodes.repeat_interleave(branch.shape[-1], dim=0)
    flat = lambda t: t.expand(*R.shape).reshape(*R.shape[:-2], m)  # noqa: E731
    return VaultOutcomes(
        wealth=flat(wealth), aum=flat(aum), weights=flat(weights), shocks=shocks,
        excess=flat(R),
    )
