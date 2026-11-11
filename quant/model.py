"""The v0 bank, one period: primitives and the transition (design.md section 3).

Everything here is element-wise and branch-free, so it vectorises over grid
points, controls and quadrature nodes at once, and stays differentiable for
the later learner. Shapes broadcast: controls carry a leading batch shape,
shocks a trailing node axis.

Shock vector order, everywhere: (xi_P, xi_N, eps_M) -- the book's payoff
shock, the new product's, and the tradeable factor's. All unit variance.
"""

from dataclasses import dataclass

import numpy as np
import torch

from quant.params import Params

SHOCKS = ("xi_P", "xi_N", "eps_M")


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

    `rho` is the irreducible floor no programme removes (design.md section 3.4).
    """
    return rho + (1.0 - rho) * torch.exp(-a * G)


def shareholder_flow(p: Params, w: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """What shareholders receive this period for moving equity from w to K.

    K < w is a dividend of w - K. K > w is an issue of e = K - w, which costs
    shareholders e + C(e). One expression covers both: w - K - C((K - w)^+).
    """
    return w - K - issuance_cost(p, torch.relu(K - w))


def capital_floor(p: Params, I: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """The least operating equity K the controls allow.

    Three constraints at once: the capital ratio K >= k I; non-negative cash,
    K - g + D - I >= 0; and no negative equity, K >= 0.
    """
    zero = torch.zeros_like(I)
    return torch.maximum(torch.maximum(p.k * I, I + g - p.D), zero)


@dataclass(frozen=True)
class Controls:
    """One period's decision, taken before the shocks.

    K: operating equity after dividends or issuance. g: GRC spend.
    I: the lending book. alpha: size of the new product. H: hedge sold on the
    tradeable factor.
    """

    K: torch.Tensor
    g: torch.Tensor
    I: torch.Tensor
    alpha: torch.Tensor
    H: torch.Tensor


def next_grc(p: Params, G: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """G' = (1 - delta) G + g. Mitigation uses this post-spend stock."""
    return (1.0 - p.delta) * G + g


def next_wealth(
    p: Params, c: Controls, G_next: torch.Tensor, shocks: torch.Tensor
) -> torch.Tensor:
    """Equity at the close of the period: assets less what depositors are owed.

        w' = F(I) - I phi_A m + I phi_B s_P xi_P       the book
           + (1 + r) R                                 cash, R = K - g + D - I
           + alpha (pi + s_N xi_N)                     the new product
           + (b_M I - H) s_M eps_M - c_H/2 H^2         the factor, net of the hedge
           - tau K                                     carrying equity
           - (1 + r_D) D                               depositors, principal and interest

    The deposit principal is subtracted in full and comes back next period as
    part of R when the base is rolled over. Adding it back here as well would
    count it twice.

    Controls have shape (...,); `shocks` has shape (N, 3); the result is
    (..., N).
    """
    xi_P, xi_N, eps_M = shocks[:, 0], shocks[:, 1], shocks[:, 2]

    def col(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)

    I, K, g, alpha, H = col(c.I), col(c.K), col(c.g), col(c.alpha), col(c.H)
    G_next = col(G_next)
    phi_A = mitigation(G_next, p.a_A, p.rho_A)
    phi_B = mitigation(G_next, p.a_B, p.rho_B)

    book = production(p, I) - I * phi_A * p.m + I * phi_B * p.s_P * xi_P
    cash = (1.0 + p.r) * (K - g + p.D - I)
    product = alpha * (p.pi + p.s_N * xi_N)
    factor = (p.b_M * I - H) * p.s_M * eps_M - 0.5 * p.c_H * H * H
    return book + cash + product + factor - p.tau * K - (1.0 + p.r_D) * p.D
