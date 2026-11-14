"""The v0 bank, one period: primitives and the transition (design.md section 4; the bank form itself is set aside, section 11).

Everything here is element-wise and branch-free, so it vectorizes over grid
points, controls and quadrature nodes at once, and stays differentiable for
the later learner. Shapes broadcast: controls carry a leading batch shape,
shocks a trailing node axis.

Shock vector order, everywhere: (xi_P, xi_N, eps_BTC) -- the book's payoff
shock, the new product's, and BTC's dollar return. All unit variance.
"""

from dataclasses import dataclass

import numpy as np
import torch

from quant.params import Params
from quant.quadrature import JumpRule, ShockRule

SHOCKS = ("xi_P", "xi_N", "eps_BTC")


def shock_correlation(p: Params) -> np.ndarray:
    """Correlation of the normal part of the shocks.

    The factor is uncorrelated with the non-tradeable shocks by construction:
    that is what "tradeable" means in Froot-Stein (cov(eps^N, M) = 0). Under the
    Student-t the shared mixing scale still makes the three dependent in the
    tail, which is the point of using it.
    """
    return np.array(
        [
            [1.0, p.rho_NP, 0.0],
            [p.rho_NP, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


# -- primitives -----------------------------------------------------------


def production(p: Params, I: torch.Tensor) -> torch.Tensor:
    """Gross return on a book of I, principal included: F(I) = A I - b/2 I^2."""
    return p.A * I - 0.5 * p.b * I * I


def issuance_cost(p: Params, e: torch.Tensor) -> torch.Tensor:
    """Deadweight cost of raising e of new equity: C(e) = lam e + kappa/2 e^2."""
    return p.lam * e + 0.5 * p.kappa * e * e


def mitigation(G: torch.Tensor, a: float, rho: float) -> torch.Tensor:
    """Share of an exposure left after a GRC stock G: rho + (1 - rho) exp(-a G).

    `rho` is the irreducible floor no program removes (design.md section 4.2).
    """
    return rho + (1.0 - rho) * torch.exp(-a * G)


def shareholder_flow(p: Params, w: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """What shareholders receive this period for moving equity from w to K.

    K < w is a dividend of w - K. K > w is an issue of e = K - w, which costs
    shareholders e + C(e). One expression covers both: w - K - C((K - w)^+).
    """
    return w - K - issuance_cost(p, torch.relu(K - w))


def capital_floor(
    p: Params, I: torch.Tensor, g: torch.Tensor, q: torch.Tensor | None = None
) -> torch.Tensor:
    """The least operating equity K the controls allow.

    Three constraints at once: the capital ratio K >= k I + k_BTC q;
    non-negative cash, K - g + D - I - q >= 0; and no negative equity, K >= 0.
    The BTC treasury q is bought spot, so it uses cash and carries its own
    capital charge.
    """
    if q is None:
        q = torch.zeros_like(I)
    zero = torch.zeros_like(I)
    ratio = p.k * I + p.k_BTC * q
    cash = I + g + q - p.D
    return torch.maximum(torch.maximum(ratio, cash), zero)


@dataclass(frozen=True)
class Controls:
    """One period's decision, taken before the shocks.

    K: operating equity after dividends or issuance. g: GRC spend.
    I: the lending book. alpha: size of the new product.
    q: the BTC treasury, held spot. H: BTC hedged away with short perps.
    """

    K: torch.Tensor
    g: torch.Tensor
    I: torch.Tensor
    alpha: torch.Tensor
    H: torch.Tensor
    q: torch.Tensor | None = None

    def treasury(self) -> torch.Tensor:
        return self.q if self.q is not None else torch.zeros_like(self.I)


def net_btc(p: Params, c: Controls) -> torch.Tensor:
    """The bank's net BTC exposure in dollars: book loading + treasury - hedge.

    What equity gains or loses when BTC moves. Gross holdings do not matter on
    their own; only what is left after the hedge.
    """
    return p.b_BTC * c.I + c.treasury() - c.H


def next_grc(p: Params, G: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """G' = (1 - delta) G + g. Mitigation uses this post-spend stock."""
    return (1.0 - p.delta) * G + g


def next_wealth(
    p: Params, c: Controls, G_next: torch.Tensor, shocks: torch.Tensor
) -> torch.Tensor:
    """Equity at the close of the period: assets less what depositors are owed.

        w' = F(I) - I phi_A m + I phi_B s_P xi_P       the book
           + (1 + r) R + (1 + r) q                     cash and the BTC treasury's
                                                       riskless part, R = K - g + D - I - q
           + alpha (pi + s_N xi_N)                     the new product
           + x (mu_BTC + s_BTC eps_BTC) - c_H/2 H^2    BTC: net exposure x = b_BTC I + q - H
           - tau K                                     carrying equity
           - (1 + r_D) D                               depositors, principal and interest

    BTC's dollar return is r + mu_BTC + s_BTC eps_BTC, where mu_BTC is the
    excess return the firm *believes* in. The treasury's riskless part cancels
    against the cash it used, so q enters only through the net exposure x --
    and through the cash and capital it ties up (`capital_floor`). Hedging with
    short perps gives up the believed premium on what is hedged.

    The deposit principal is subtracted in full and comes back next period as
    part of R when the base is rolled over. Adding it back here as well would
    count it twice.

    Controls have shape (...,); `shocks` has shape (N, 3); the result is
    (..., N).
    """
    xi_P, xi_N, eps_BTC = shocks[:, 0], shocks[:, 1], shocks[:, 2]

    def col(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)

    I, K, g, alpha, H = col(c.I), col(c.K), col(c.g), col(c.alpha), col(c.H)
    x = col(net_btc(p, c))
    G_next = col(G_next)
    phi_A = mitigation(G_next, p.a_A, p.rho_A)
    phi_B = mitigation(G_next, p.a_B, p.rho_B)

    book = production(p, I) - I * phi_A * p.m + I * phi_B * p.s_P * xi_P
    cash = (1.0 + p.r) * (K - g + p.D - I)   # R + q: the treasury's riskless part
    product = alpha * (p.pi + p.s_N * xi_N)
    btc = x * (p.mu_BTC + p.s_BTC * eps_BTC) - 0.5 * p.c_H * H * H
    return book + cash + product + btc - p.tau * K - (1.0 + p.r_D) * p.D


def jump_probability(p: Params, G_next: torch.Tensor) -> torch.Tensor:
    """Chance of a jump this period: 1 - exp(-rate phi_J(G')).

    GRC acts on frequency -- prevention -- with the same irreducible floor as
    the other mechanisms. The probability enters the quadrature *weights*, so
    it is exactly differentiable in the spend; no sampling device is needed.
    """
    return 1.0 - torch.exp(-p.jump_rate * mitigation(G_next, p.a_J, p.rho_J))


@dataclass(frozen=True)
class Outcomes:
    """Every next-period outcome of a decision, with its probability.

    `wealth` and `weights` are (..., M) with M = N (1 + J): each continuous
    shock node, without a jump and then with each jump severity. `shocks` is
    (M, 3), the continuous shock behind each outcome, for measuring the price
    of a particular risk.
    """

    wealth: torch.Tensor
    weights: torch.Tensor
    shocks: torch.Tensor

    def expect(self, values: torch.Tensor) -> torch.Tensor:
        return (values * self.weights).sum(-1)


def outcomes(
    p: Params, c: Controls, G_next: torch.Tensor, rule: ShockRule, jumps: JumpRule | None
) -> Outcomes:
    """w' over the continuous shocks and, if any, the jump losses (design.md 4.3).

    A jump takes a share J of the book, on top of everything else, so with a
    jump w' falls by I J. It impairs an asset rather than paying cash out --
    the book is what a drained bridge or a defaulting protocol takes.
    """
    base = next_wealth(p, c, G_next, rule.nodes)                    # (..., N)
    if jumps is None or p.jump_rate == 0.0:
        weights = rule.weights.expand_as(base)
        return Outcomes(wealth=base, weights=weights, shocks=rule.nodes)
    loss = c.I[..., None, None] * jumps.severity                     # (..., 1, J)
    hit = base.unsqueeze(-1) - loss                                 # (..., N, J)
    wealth = torch.cat([base.unsqueeze(-1), hit], dim=-1)           # (..., N, 1 + J)
    q = jump_probability(p, G_next).unsqueeze(-1)                   # (..., 1)
    branch = torch.cat([1.0 - q, q * jumps.weights], dim=-1)        # (..., 1 + J)
    weights = rule.weights[:, None] * branch.unsqueeze(-2)          # (..., N, 1 + J)
    n_out = wealth.shape[-2] * wealth.shape[-1]
    shocks = rule.nodes.repeat_interleave(1 + jumps.severity.shape[0], dim=0)
    return Outcomes(
        wealth=wealth.reshape(*wealth.shape[:-2], n_out),
        weights=weights.reshape(*weights.shape[:-2], n_out),
        shocks=shocks,
    )
