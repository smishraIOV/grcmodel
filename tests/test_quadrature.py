"""The shock rules: exact second moments, and tails that are actually fat."""

import math

import numpy as np
import pytest
import torch

from quant.numerics import REFERENCE
from quant.quadrature import student_t_rule

P = REFERENCE
CORR = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])


def moments(rule):
    x, w = rule.nodes, rule.weights
    mean = w @ x
    cov = (x * w[:, None]).T @ x
    kurt = (w @ x**4) / torch.diagonal(cov) ** 2
    return mean, cov, kurt


@pytest.mark.parametrize("nu", [math.inf, 30.0, 8.0, 5.0])
def test_mean_zero_and_covariance_exact(nu):
    rule = student_t_rule(CORR, nu, P)
    mean, cov, _ = moments(rule)
    assert float(rule.weights.sum()) == pytest.approx(1.0, abs=1e-14)
    assert torch.allclose(mean, torch.zeros(3, dtype=P.dtype), atol=1e-13)
    assert torch.allclose(cov, torch.tensor(CORR, dtype=P.dtype), atol=1e-12)


def test_normal_kurtosis_is_three():
    _, _, kurt = moments(student_t_rule(CORR, math.inf, P))
    assert torch.allclose(kurt, torch.full((3,), 3.0, dtype=P.dtype), atol=1e-10)


@pytest.mark.parametrize("nu, tol", [(8.0, 0.01), (5.0, 0.03)])
def test_t_kurtosis_reaches_the_true_value(nu, tol):
    """3 + 6 / (nu - 4). The plain Gauss-Laguerre rule misses this badly."""
    _, _, kurt = moments(student_t_rule(CORR, nu, P))
    target = 3.0 + 6.0 / (nu - 4.0)
    assert torch.allclose(kurt, torch.full((3,), target, dtype=P.dtype), rtol=tol)


def test_shared_scale_gives_tail_dependence():
    """Uncorrelated components still move together in the tail under the t.

    xi_P and eps_BTC have zero correlation. Under the normal they are independent
    and E[xi_P^2 eps_BTC^2] = 1; under the t the shared scale pushes it above 1.
    """
    def co_moment(nu):
        x, w = student_t_rule(CORR, nu, P).nodes, student_t_rule(CORR, nu, P).weights
        return float(w @ (x[:, 0] ** 2 * x[:, 2] ** 2))

    assert co_moment(math.inf) == pytest.approx(1.0, abs=1e-12)
    assert co_moment(5.0) > 1.5
