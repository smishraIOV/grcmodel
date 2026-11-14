"""The BTC yield vault, one period: strategies, fees in BTC, flows, and the
firm's books (design.md sections 2.3-2.6 and 3).

**The vault** belongs to its investors. They deposit BTC and want more BTC.
The vault spreads its BTC across strategies run by hired managers -- lending
BTC to institutions, basis trading and market-making, cold custody -- each
with a BTC yield, a volatility, a loss in BTC crashes, and exposure to
incidents. Its size A is its BTC valued in dollars, so it moves with the BTC
price as well as with its yield and flows.

**The firm** belongs to shareholders. It sets the allocation, earns a
management fee on the vault's BTC and a performance fee on positive BTC yield
-- both paid in BTC, so worth more or less in dollars as BTC moves -- pays
operating costs (fixed in dollars) and GRC, repays a share of incident
losses, and keeps a stake of its own BTC in the vault. Its own books are
`quant.firm`'s.

Scenario order: (xi_1, xi_2, eps_BTC, crash) -- the two active strategies'
yield shocks, BTC's smooth move, and the share of BTC's price a crash takes.
"""

from dataclasses import dataclass

import numpy as np
import torch

from quant.core import jump_probability, mitigation
from quant.firm import btc_prices, capital_floor, scenario_rule, treasury_close
from quant.numerics import NumericsProfile
from quant.params import Params
from quant.quadrature import JumpRule, ShockRule


def btc_vault_correlation(p: Params) -> np.ndarray:
    """The two active strategies' yields correlate; BTC's smooth move is
    separate in the normal part and shares the tail through the Student-t."""
    return np.array(
        [
            [1.0, p.strategy_rho, 0.0],
            [p.strategy_rho, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def btc_vault_rule(
    p: Params, profile: NumericsProfile, n_strategy: int = 3, n_btc: int = 7,
    n_mixing: int = 6, n_crash: int = 5,
) -> ShockRule:
    """Scenarios over (xi_1, xi_2, eps_BTC, crash) (`quant.firm.scenario_rule`)."""
    return scenario_rule(
        p, btc_vault_correlation(p), (n_strategy, n_strategy, n_btc), profile,
        n_mixing=n_mixing, n_crash=n_crash,
    )


@dataclass(frozen=True)
class BtcVaultControls:
    """One period's decision, taken before the shocks.

    K: the firm's equity after dividends or issuance. g: GRC spend.
    q: the firm's own BTC, bought spot. H: BTC hedged with short perps, at most
    the firm's BTC including its stake in the vault. alloc: the vault's shares
    in the three strategies, summing to one, shape (..., 3).
    """

    K: torch.Tensor
    g: torch.Tensor
    q: torch.Tensor
    H: torch.Tensor
    alloc: torch.Tensor


def btc_vault_capital_floor(
    p: Params, A: torch.Tensor, g: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """The firm's capital floor (`quant.firm.capital_floor`), with its stake in
    the BTC vault."""
    return capital_floor(p, A, g, q, p.btc_co_invest * A)


def btc_held(p: Params, A: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """BTC the firm owns, and may hedge: its own plus its stake in the vault."""
    return q + p.btc_co_invest * A


@dataclass(frozen=True)
class BtcVaultOutcomes:
    """Every next-period outcome of a decision, with its probability.

    Tensors are (..., M) with M = N (1 + J): each market scenario, without an
    incident and then with each incident severity. `shocks` is (M, 4).

    wealth: the firm's closing equity. aum: the vault's next size, in dollars.
    yield_: the vault's BTC yield before fees. ret: investors' BTC return,
    after fees and before compensation.
    """

    wealth: torch.Tensor
    aum: torch.Tensor
    weights: torch.Tensor
    shocks: torch.Tensor
    yield_: torch.Tensor
    ret: torch.Tensor

    def expect(self, values: torch.Tensor) -> torch.Tensor:
        return (values * self.weights).sum(-1)


def btc_vault_outcomes(
    p: Params,
    A: torch.Tensor,
    c: BtcVaultControls,
    S_next: torch.Tensor,
    rule: ShockRule,
    jumps: JumpRule | None,
    bias: float | None = None,
) -> BtcVaultOutcomes:
    """Closing firm equity and the vault's next size, over every outcome.

    In BTC terms, per unit of the vault's BTC, with a the allocation, C the
    crash size, J the incident severity and co the firm's stake:

        y      = sum_i a_i (mu_i - phi_A m_i - c_i C) + phi_B (a_1 s_1 xi_1 + a_2 s_2 xi_2)
                 - J sum_i a_i j_i                          the vault's BTC yield
        fee    = f_m + f_p max(y, 0)                        paid to the firm, in BTC
        R      = y - fee                                    investors' BTC return
        cover  = psi (1 - co) J sum_i a_i j_i               the firm repays outsiders

    and in dollars, with P the BTC price at the close (it starts at 1):

        A'  = A (1 + R + cover) P + theta (cap - A)
              + eta (R + cover / (1 - co) - y_out) A - [incident] omega A,    floored at 0
        w'  = treasury_close(...)                           the firm's own books
              + (fee + co (1 + R) - cover) A P

    `bias` is the firm's expected BTC return above the market's; the firm
    decides with `p.btc_bias`, and its value is measured with 0.
    """
    if bias is None:
        bias = p.btc_bias
    xi = rule.nodes[:, :2]                                                  # (N, 2)
    eps, crash = rule.nodes[:, 2], rule.nodes[:, 3]                         # (N,)

    def col(x):
        return x.unsqueeze(-1)

    a = c.alloc                                                             # (..., 3)
    S_col = col(S_next)
    phi_A = mitigation(S_col, p.a_A, p.rho_A)
    phi_B = mitigation(S_col, p.a_B, p.rho_B)
    mu, loss = a.new_tensor(p.strategy_yield), a.new_tensor(p.strategy_loss)
    vol, in_crash = a.new_tensor(p.strategy_vol), a.new_tensor(p.strategy_crash)
    exposure = a.new_tensor(p.strategy_incident)

    drift = (a * (mu - phi_A * loss)).sum(-1, keepdim=True)                 # (..., 1)
    risky = phi_B * (a[..., :2] * vol[:2]) @ xi.T                            # (..., N)
    crashed = (a * in_crash).sum(-1, keepdim=True) * crash                   # (..., N)
    y_market = drift + risky - crashed                                       # (..., N)

    price, _ = btc_prices(p, eps, crash, bias)                               # (N,)
    co = p.btc_co_invest
    books = treasury_close(p, c.K, c.g, c.q, c.H, co * A, price - 1.0)       # (..., N)

    if jumps is not None and p.jump_rate > 0.0:
        severity = torch.cat([jumps.severity.new_zeros(1), jumps.severity])    # (1 + J,)
        prob = col(jump_probability(p, S_next))                                 # (..., 1)
        branch = torch.cat([1.0 - prob, prob * jumps.weights], dim=-1)          # (..., 1 + J)
    else:
        severity = A.new_zeros(1)
        branch = torch.ones_like(col(A))
    occurred = (severity > 0).to(A.dtype)

    hit = (a * exposure).sum(-1, keepdim=True).unsqueeze(-1) * severity      # (..., 1, 1 + J)
    y = y_market.unsqueeze(-1) - hit                                         # (..., N, 1 + J)
    fee = p.fee_mgmt + p.fee_perf * torch.relu(y)
    R = y - fee
    cover = p.compensation * (1.0 - co) * hit
    A3, P3 = A[..., None, None], price[:, None]
    aum = torch.relu(
        A3 * (1.0 + R + cover) * P3
        + p.inflow_speed * (p.btc_capacity - A3)
        + p.btc_flow_yield * (R + cover / (1.0 - co) - p.btc_outside_yield) * A3
        - occurred * p.outflow_incident * A3
    )
    wealth = books.unsqueeze(-1) + (fee + co * (1.0 + R) - cover) * A3 * P3

    weights = rule.weights[:, None] * branch.unsqueeze(-2)                   # (..., N, 1 + J)
    m = rule.nodes.shape[0] * branch.shape[-1]
    shocks = rule.nodes.repeat_interleave(branch.shape[-1], dim=0)

    def flat(t):
        return t.expand(*y.shape).reshape(*y.shape[:-2], m)

    return BtcVaultOutcomes(
        wealth=flat(wealth), aum=flat(aum), weights=flat(weights), shocks=shocks,
        yield_=flat(y), ret=flat(R),
    )
