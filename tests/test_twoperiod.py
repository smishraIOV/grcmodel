"""The two-period truncation reproduces Froot & Stein (1998) under normal shocks."""

import pytest
import torch

from quant.numerics import REFERENCE
from quant.quadrature import student_t_rule
from quant.twoperiod import (
    ORACLE_FIRST_PERIOD,
    final_marginals,
    first_period_wealth,
    froot_stein,
    issuance_threshold,
    optimal_positions,
    oracle_params,
    shock_correlation,
)

P = REFERENCE


@pytest.fixture(scope="module")
def oracle():
    p = oracle_params()
    rule = student_t_rule(shock_correlation(p), p.nu, P)
    alpha, H = optimal_positions(p, rule, P, ORACLE_FIRST_PERIOD)
    return p, rule, alpha, H


def test_oracle_stays_in_the_quadratic_region(oracle):
    """Every node below the issuance threshold, or the oracle is not exact."""
    p, rule, alpha, H = oracle
    w = first_period_wealth(p, rule, P, ORACLE_FIRST_PERIOD, alpha, H)
    assert float(w.max()) < issuance_threshold(p)


def test_final_value_is_increasing_and_concave():
    p = oracle_params()
    w = torch.linspace(-0.05, 0.3, 200, dtype=P.dtype)
    d1, d2 = final_marginals(p, w)
    assert bool((d1 > 0).all())
    assert bool((d2 < 1e-6).all())
    # Above the payout point an extra unit is simply paid out.
    assert float(d1[-1]) == pytest.approx(1.0, abs=1e-8)


def test_final_value_curvature_matches_closed_form():
    """Below the threshold P is quadratic with P'' = kappa (k^2 kappa / den - 1)."""
    p = oracle_params()
    den = p.beta * p.b + p.k**2 * p.kappa
    expected = p.kappa * (p.k**2 * p.kappa / den - 1.0)
    w = torch.linspace(-0.05, 0.06, 20, dtype=P.dtype)
    _, d2 = final_marginals(p, w)
    assert torch.allclose(d2, torch.full_like(d2, expected), rtol=1e-6)


def test_tradeable_risk_is_fully_hedged(oracle):
    """Prop. 2: H* = b_BTC I, found by an optimizer that does not know it."""
    p, _, _, H = oracle
    assert H == pytest.approx(p.b_BTC * ORACLE_FIRST_PERIOD.I, rel=1e-5)


def test_capital_budgeting_matches_eq_9(oracle):
    """alpha* = (pi - G cov_NP) / (G var_N), with G = -E[P''] / E[P']."""
    p, rule, alpha, H = oracle
    fs = froot_stein(p, rule, P, ORACLE_FIRST_PERIOD, alpha, H)
    assert 0.0 < alpha < p.alpha_max
    assert fs.G > 0.0
    assert alpha == pytest.approx(fs.alpha, rel=1e-4)


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.5])
def test_hurdle_takes_the_sign_of_the_covariance(rho):
    """Eq. (10): a product correlated with the book must clear a higher bar."""
    p = oracle_params(rho_NP=rho)
    rule = student_t_rule(shock_correlation(p), p.nu, P)
    alpha, H = optimal_positions(p, rule, P, ORACLE_FIRST_PERIOD)
    hurdle = froot_stein(p, rule, P, ORACLE_FIRST_PERIOD, alpha, H).hurdle
    assert hurdle == pytest.approx(0.0, abs=1e-12) if rho == 0 else (hurdle > 0) == (rho > 0)


def test_a_view_on_btc_matches_eq_15():
    """With a view the bank keeps x* = mu_BTC / (G s_BTC^2) of net BTC
    exposure -- the paper's eq. (15) case -- found by an optimizer that does
    not know the formula. Exact under normal shocks."""
    p = oracle_params(mu_BTC=0.0001)  # small, so outcomes stay in the exact region
    rule = student_t_rule(shock_correlation(p), p.nu, P)
    fixed = ORACLE_FIRST_PERIOD
    alpha, H = optimal_positions(p, rule, P, fixed)
    w = first_period_wealth(p, rule, P, fixed, alpha, H)
    assert float(w.max()) < issuance_threshold(p), "oracle must stay in its quadratic region"
    exposure = p.b_BTC * fixed.I - H
    expected = froot_stein(p, rule, P, fixed, alpha, H).btc
    assert exposure > 0
    assert exposure == pytest.approx(expected, rel=1e-4)
