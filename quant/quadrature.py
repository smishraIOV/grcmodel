"""Deterministic integration rules for the shocks and jumps (design.md sections 4.2, 5).

The shocks are a multivariate Student-t, written as a scale mixture of normals:

    xi = sqrt(v) * L z,    z ~ N(0, I),    v = (nu / W) / E[nu / W],    W ~ chi2(nu)

with one `v` shared by every component. The shared scale is what produces tail
dependence -- when it is large, every risk is large at once.

Two nested rules:

- **The mixing scale `v`** uses Gauss-Legendre in log-probability space: nodes
  are placed by the CDF of W, with the grid uniform in log u, so points
  crowd into the small-W tail where the large shocks live. The plain
  generalized Gauss-Laguerre rule was measured first and badly understates the
  tail: at nu = 5 it reaches a kurtosis of 7.3 against a true 9 with 64 points.
  This rule reaches 8.8 with 12. The remainder is the probability mass beyond
  `tail` (1e-10), dropped and renormalized.
- **The normal part** is a tensor-product Gauss-Hermite rule, correlated by a
  Cholesky factor.

`v` is renormalized so that the rule's variance is exactly one. Second moments
are therefore exact whatever the node count; what the node count buys is the
tail.

Built in numpy float64, once, and handed to torch through a NumericsProfile.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import gammaincinv

from quant.numerics import NumericsProfile


def gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and probability weights for a standard normal."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


def mixing_rule(nu: float, n: int, tail: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Variance scales `v` and their probabilities, with E[v] = 1 exactly.

    For nu = inf the mixture is degenerate: one node at v = 1.
    """
    if math.isinf(nu):
        return np.ones(1), np.ones(1)
    s, ws = np.polynomial.legendre.leggauss(n)
    lo = math.log(tail)
    s = lo + (s + 1.0) * (-lo) / 2.0
    ws = ws * (-lo) / 2.0
    u = np.exp(s)                   # CDF of W at each node
    p = ws * u                      # du = u ds
    p = p / p.sum()
    W = 2.0 * gammaincinv(nu / 2.0, u)
    v = nu / W
    return v / np.sum(p * v), p


@dataclass(frozen=True)
class ShockRule:
    """Quadrature nodes for a d-dimensional shock vector.

    `nodes` is (N, d) and `weights` (N,), summing to one. Each node is one
    joint scenario; an expectation is `weights @ f(nodes)`.
    """

    nodes: torch.Tensor
    weights: torch.Tensor

    @property
    def size(self) -> int:
        return self.weights.shape[0]

    def expect(self, values: torch.Tensor) -> torch.Tensor:
        """E[f] for values whose *last* axis runs over the nodes."""
        return values @ self.weights


def student_t_rule(
    corr: np.ndarray,
    nu: float,
    profile: NumericsProfile,
    n_hermite: int = 7,
    n_mixing: int = 12,
) -> ShockRule:
    """Rule for a unit-variance multivariate t with correlation matrix `corr`."""
    corr = np.asarray(corr, dtype=np.float64)
    d = corr.shape[0]
    chol = np.linalg.cholesky(corr)
    x, w = gauss_hermite(n_hermite)
    z = np.array(list(itertools.product(x, repeat=d)))             # (H, d)
    wz = np.prod(np.array(list(itertools.product(w, repeat=d))), axis=1)
    correlated = z @ chol.T
    v, pv = mixing_rule(nu, n_mixing)
    nodes = (np.sqrt(v)[:, None, None] * correlated[None, :, :]).reshape(-1, d)
    weights = (pv[:, None] * wz[None, :]).reshape(-1)
    return ShockRule(
        nodes=profile.tensor(nodes),
        weights=profile.tensor(weights / weights.sum()),
    )


@dataclass(frozen=True)
class JumpRule:
    """Severity nodes for a jump loss, as shares of the book, with probabilities.

    The severity is Pareto with tail index `tail`, scaled to mean `mean`, and
    capped at 1: a jump cannot take more than the whole book. Nodes are placed
    like the mixing rule's -- Gauss-Legendre in log tail-probability -- so they
    crowd into the far tail where a heavy-tailed loss does its damage.
    """

    severity: torch.Tensor   # (J,)
    weights: torch.Tensor    # (J,), summing to one

    @property
    def mean(self) -> float:
        return float(self.weights @ self.severity)


def pareto_jump_rule(
    mean: float, tail: float, profile: NumericsProfile, n: int = 8, tail_prob: float = 1e-8
) -> JumpRule:
    s, ws = np.polynomial.legendre.leggauss(n)
    lo = math.log(tail_prob)
    s = lo + (s + 1.0) * (-lo) / 2.0
    ws = ws * (-lo) / 2.0
    survival = np.exp(s)               # P(severity > x) at each node
    p = ws * survival
    p = p / p.sum()
    scale = mean * (tail - 1.0) / tail     # Pareto minimum, so that E = mean before the cap
    severity = np.minimum(scale * survival ** (-1.0 / tail), 1.0)
    return JumpRule(severity=profile.tensor(severity), weights=profile.tensor(p))
