"""The firm's own books and BTC's market, shared by every vault
(design.md sections 3.3, 3.4 and 4.3).

**The firm's books.** Equity K is held as stablecoins (earning nothing), the
firm's own BTC q (bought spot, costing `btc_holding_cost` a year to secure),
and its stake in a vault. Part of its BTC is hedged with short perps H. It
pays operating costs, fixed in dollars, and GRC. A runway of operating costs
is kept in stablecoins.

**BTC's market.** BTC's price over a period is a smooth fat-tailed move, then
possibly a sudden crash. The smooth part's drift is set so the expected end
price is the market's expectation, 1 + r_E, plus the firm's bias.

Scenario order, for any vault: the vault's own shocks first, then eps_BTC
(BTC's smooth move), then the crash size (zero without one).
"""

import math

import numpy as np
import torch

from quant.numerics import NumericsProfile
from quant.params import Params
from quant.quadrature import ShockRule, pareto_jump_rule, student_t_rule


def crash_probability(p: Params) -> float:
    """Chance of a BTC crash in one period: 1 - exp(-rate). Not affected by
    GRC: a crash is a market event."""
    return 1.0 - math.exp(-p.btc_crash_rate)


def btc_prices(
    p: Params, eps: torch.Tensor, crash: torch.Tensor, bias: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """BTC's price at the end of the period, starting from 1, and the price the
    smooth part of the move alone would reach.

    The move is a smooth part, 1 + mu + s eps, then possibly a crash that
    takes a share `crash` of the price at once. mu is set so the expected end
    price is 1 + r_E + bias whatever the crash risk: the market's expectation,
    plus the firm's bias (design.md section 3.4). Prices cannot fall below
    zero.
    """
    expected_loss = crash_probability(p) * p.btc_crash_mean
    mu = (1.0 + p.r_E + bias) / (1.0 - expected_loss) - 1.0
    smooth = torch.clamp(1.0 + mu + p.s_BTC * eps, min=0.0)
    return smooth * (1.0 - crash), smooth


def scenario_rule(
    p: Params,
    corr: np.ndarray,
    counts: tuple[int, ...],
    profile: NumericsProfile,
    n_mixing: int = 6,
    n_crash: int = 5,
) -> ShockRule:
    """The Student-t rule over the vault's shocks and BTC's smooth move (the
    last direction of `corr`), crossed with no crash and each crash size. The
    crash probability is folded into the weights."""
    base = student_t_rule(corr, p.nu, profile, n_hermite=counts, n_mixing=n_mixing)
    if p.btc_crash_rate > 0.0:
        crashes = pareto_jump_rule(p.btc_crash_mean, p.btc_crash_tail, profile, n=n_crash)
        size = torch.cat([crashes.severity.new_zeros(1), crashes.severity])
        prob = crash_probability(p)
        chance = torch.cat([size.new_ones(1) * (1.0 - prob), prob * crashes.weights])
    else:
        size, chance = base.weights.new_zeros(1), base.weights.new_ones(1)
    k = size.shape[0]
    nodes = torch.cat(
        [base.nodes.repeat_interleave(k, dim=0), size.repeat(base.size).unsqueeze(-1)], dim=-1
    )
    weights = (base.weights[:, None] * chance[None, :]).reshape(-1)
    return ShockRule(nodes=nodes, weights=weights)


def capital_floor(
    p: Params, size: torch.Tensor, g: torch.Tensor, q: torch.Tensor, stake: torch.Tensor
) -> torch.Tensor:
    """The least equity the firm may hold.

    Own funds against the vault's dollar size and the firm's BTC, hedged or
    not: K >= k_A size + k_BTC q. A runway of operating costs in stablecoins
    after GRC, BTC and the stake in the vault: K >= g + q + stake + runway
    opex (design.md section 3.3). And no negative equity.
    """
    cash = g + q + stake + p.runway * p.opex
    return torch.maximum(torch.maximum(p.k_A * size + p.k_BTC * q, cash), torch.zeros_like(size))


def treasury_close(
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
        + q (1 + R_btc - h_q) - H (R_btc - r) - c_H/2 H^2   BTC, its security cost, short perps
        - tau K - opex

    A short perp gains when BTC falls and pays funding near r, so hedged BTC
    earns about r less its security cost. Decisions are (...,), R_btc (N,);
    the result is (..., N).
    """
    def col(x):
        return x.unsqueeze(-1)

    btc = (col(q) * (1.0 + R_btc - p.btc_holding_cost) - col(H) * (R_btc - p.r)
           - 0.5 * p.c_H * col(H) * col(H))
    return col((K - g - q - stake) - p.tau * K - p.opex) + btc
