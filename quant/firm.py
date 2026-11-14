"""The firm's own books and BTC's market, shared by every vault
(design.md sections 3.3, 3.4 and 4.3).

**The firm's books.** Equity K is held as stablecoins (earning nothing), the
firm's own BTC q (bought spot, costing `btc_holding_cost` a year to secure),
and its stake in a vault. Part of its BTC is hedged with short perps H. It
pays operating costs, fixed in dollars, and GRC. A runway of operating costs
is kept in stablecoins.

**BTC's market.** BTC's price over a period is a smooth move, then possibly a
sudden crash, both in the logarithm of the price, so it never reaches zero.
The smooth part's drift is set so the expected end price is the market's
expectation, 1 + r_E, plus the firm's bias.

Scenario order, for any vault: the vault's own shocks first, then eps_BTC
(BTC's smooth move), then the crash size (zero without one).
"""

import math

import numpy as np
import torch

from quant.numerics import NumericsProfile
from quant.params import Params
from quant.quadrature import ShockRule, crash_rule, gauss_hermite, normal_grid, student_t_rule


def crash_probability(p: Params) -> float:
    """Chance of a BTC crash in one period: 1 - exp(-rate). Not affected by
    GRC: a crash is a market event."""
    return 1.0 - math.exp(-p.btc_crash_rate)


def btc_prices(
    p: Params, eps: torch.Tensor, crash: torch.Tensor, bias: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """BTC's price at the end of the period, starting from 1, and the price the
    smooth part of the move alone would reach.

    The move is in the **logarithm** of the price (design.md section 3.4):

        smooth = exp(m + s eps),    eps standard normal
        price  = smooth (1 - crash)

    where `crash` is the share of the price a crash takes, below one
    (`quant.quadrature.crash_rule`). The price is therefore always positive:
    a model that adds a move to the price itself reaches zero, and at BTC's
    volatility did so far too often. m is set so the expected end price is
    1 + r_E + bias whatever the crash risk -- the market's expectation, plus
    the firm's bias:

        m = ln((1 + r_E + bias) / (1 - P(crash) mean share)) - s^2 / 2.
    """
    expected_loss = crash_probability(p) * p.btc_crash_mean
    m = math.log((1.0 + p.r_E + bias) / (1.0 - expected_loss)) - 0.5 * p.s_BTC * p.s_BTC
    smooth = torch.exp(m + p.s_BTC * eps)
    return smooth * (1.0 - crash), smooth


def scenario_rule(
    p: Params,
    corr: np.ndarray,
    counts: tuple[int, ...],
    profile: NumericsProfile,
    n_mixing: int = 6,
    n_crash: int = 5,
    even_btc: bool = False,
) -> ShockRule:
    """Scenarios over the vault's shocks, BTC's smooth move and the crash.

    `corr` and `counts` list the vault's shocks first and BTC's smooth move
    last. The vault's shocks are Student-t and share one tail
    (design.md section 4.2). BTC's smooth move is **normal and separate**: it
    moves the logarithm of the price, and a Student-t there has no finite
    expected price. BTC's tail is the crash, whose probability is folded into
    the weights; a crash still reaches the loan book, the strategies and the
    firm's own BTC together.

    BTC's smooth move uses Gauss-Hermite points, or with `even_btc` an even
    grid (`quant.quadrature.normal_grid`): what the firm ends with has kinks
    in BTC's price, which many evenly spaced points resolve and a few
    Gauss-Hermite points do not.
    """
    corr = np.asarray(corr, dtype=np.float64)
    d = corr.shape[0] - 1
    if not (np.allclose(corr[d, :d], 0.0) and np.allclose(corr[:d, d], 0.0)):
        raise ValueError("BTC's smooth move must be uncorrelated with the vault's shocks")
    vault = student_t_rule(corr[:d, :d], p.nu, profile, n_hermite=tuple(counts[:d]), n_mixing=n_mixing)
    z, wz = normal_grid(counts[d]) if even_btc else gauss_hermite(counts[d])
    eps, eps_weights = profile.tensor(z), profile.tensor(wz)
    if p.btc_crash_rate > 0.0:
        crashes = crash_rule(p.btc_crash_mean, p.btc_crash_tail, profile, n=n_crash)
        size = torch.cat([crashes.severity.new_zeros(1), crashes.severity])
        prob = crash_probability(p)
        chance = torch.cat([size.new_ones(1) * (1.0 - prob), prob * crashes.weights])
    else:
        size, chance = vault.weights.new_zeros(1), vault.weights.new_ones(1)
    v, b, k = vault.size, eps.shape[0], size.shape[0]
    nodes = torch.cat(
        [
            vault.nodes.repeat_interleave(b * k, dim=0),
            eps.repeat_interleave(k).repeat(v).unsqueeze(-1),
            size.repeat(v * b).unsqueeze(-1),
        ],
        dim=-1,
    )
    weights = (vault.weights[:, None, None] * eps_weights[None, :, None] * chance[None, None, :]).reshape(-1)
    return ShockRule(nodes=nodes, weights=weights)


def capital_floor(
    p: Params, size: torch.Tensor, g: torch.Tensor, q: torch.Tensor, stake: torch.Tensor
) -> torch.Tensor:
    """The least equity the firm may hold.

    Own funds against the vault's dollar size: K >= k_A size. A runway of operating costs in stablecoins
    after GRC, BTC and the stake in the vault: K >= g + q + stake + runway
    opex (design.md section 3.3). And no negative equity. The firm's BTC needs
    no separate capital charge: it is bought with equity, so the second
    constraint already holds it fully.
    """
    cash = g + q + stake + p.runway * p.opex
    return torch.maximum(torch.maximum(p.k_A * size, cash), torch.zeros_like(size))


def treasury(
    p: Params,
    K: torch.Tensor,
    g: torch.Tensor,
    q: torch.Tensor,
    H: torch.Tensor,
    stake: torch.Tensor,
    R_btc: torch.Tensor,
) -> torch.Tensor:
    """The firm's own holdings at the close, before anything from the vault:

        (K - g - q - stake)                                stablecoins, earning nothing
        + q (1 + R_btc - h_q) - H (R_btc - f) - c_H/2 H^2   BTC, its security cost, short perps
        - tau K - opex

    f = r + perp_premium is the market's perp funding rate: longs pay it and
    shorts receive it (design.md section 3.4). A short perp gains when BTC
    falls and receives f, so hedged BTC earns f less its security cost.
    Element-wise: the arguments broadcast together.
    """
    btc = q * (1.0 + R_btc - p.btc_holding_cost) - H * (R_btc - p.r - p.perp_premium) - 0.5 * p.c_H * H * H
    return ((K - g - q - stake) - p.tau * K - p.opex) + btc


def treasury_close(
    p: Params,
    K: torch.Tensor,
    g: torch.Tensor,
    q: torch.Tensor,
    H: torch.Tensor,
    stake: torch.Tensor,
    R_btc: torch.Tensor,
) -> torch.Tensor:
    """`treasury` over every scenario: decisions are (...,), R_btc (N,); the
    result is (..., N)."""
    def col(x):
        return x.unsqueeze(-1)

    return treasury(p, col(K), col(g), col(q), col(H), col(stake), R_btc)
