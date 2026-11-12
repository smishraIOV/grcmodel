"""One period of the bank: primitives and the accounting."""

from dataclasses import replace

import pytest
import torch

from quant.model import (
    Controls,
    capital_floor,
    mitigation,
    next_grc,
    next_wealth,
    production,
    shareholder_flow,
    shock_correlation,
)
from quant.numerics import REFERENCE
from quant.params import DEFAULTS, Params
from quant.quadrature import student_t_rule

P = REFERENCE


def t(x):
    return P.tensor(x)


def controls(K=0.1, g=0.0, I=1.0, alpha=0.0, H=0.0):
    return Controls(K=t(K), g=t(g), I=t(I), alpha=t(alpha), H=t(H))


def no_shock():
    return P.zeros(1, 3)


def test_units_put_the_first_best_book_at_one():
    """F'(1) = 1 + r, by the definition of b."""
    I = t(1.0).requires_grad_(True)
    (grad,) = torch.autograd.grad(production(DEFAULTS, I), I)
    assert float(grad) == pytest.approx(1.0 + DEFAULTS.r, abs=1e-14)


def test_params_refuse_an_unprofitable_book():
    with pytest.raises(ValueError):
        Params(A=1.01, r=0.02)


def test_idle_bank_earns_the_carry_and_the_deposit_spread():
    """No book, no GRC: w' = K (1 + r - tau) + D (r - r_D)."""
    p = DEFAULTS
    c = controls(K=0.2, I=0.0)
    w = next_wealth(p, c, t(0.0), no_shock())
    expected = 0.2 * (1 + p.r - p.tau) + p.D * (p.r - p.r_D)
    assert float(w) == pytest.approx(expected, abs=1e-14)


def test_shocks_average_out_of_expected_wealth():
    """E[w'] under the rule equals w' at the zero shock, for every nu."""
    for nu in (float("inf"), 5.0):
        p = replace(DEFAULTS, nu=nu)
        rule = student_t_rule(shock_correlation(p), p.nu, P)
        c = controls(K=0.12, g=0.003, I=0.9, alpha=0.3, H=0.2)
        G = next_grc(p, t(0.01), c.g)
        mean = rule.expect(next_wealth(p, c, G, rule.nodes))
        at_zero = next_wealth(p, c, G, no_shock())
        assert float(mean) == pytest.approx(float(at_zero), abs=1e-13)


def test_hedging_removes_the_factor_exposure():
    """With H = b_BTC I the factor drops out of w' entirely."""
    p = DEFAULTS
    shocks = P.tensor([[0.0, 0.0, -3.0], [0.0, 0.0, 3.0]])
    c = controls(I=0.8, H=p.b_BTC * 0.8)
    w = next_wealth(p, c, t(0.0), shocks)
    assert float(w[0]) == pytest.approx(float(w[1]), abs=1e-14)


def test_dividend_and_issuance_flows():
    p = DEFAULTS
    assert float(shareholder_flow(p, t(0.3), t(0.1))) == pytest.approx(0.2)
    e = 0.1
    cost = p.lam * e + 0.5 * p.kappa * e * e
    assert float(shareholder_flow(p, t(0.1), t(0.2))) == pytest.approx(-e - cost)


def test_capital_floor_takes_the_tightest_constraint():
    p = DEFAULTS  # k = 0.08, D = 0.9
    assert float(capital_floor(p, t(0.5), t(0.0))) == pytest.approx(0.04)       # ratio
    assert float(capital_floor(p, t(1.0), t(0.05))) == pytest.approx(0.15)      # cash
    assert float(capital_floor(replace(p, k=0.0), t(0.0), t(0.0))) == 0.0       # no negative equity


def test_mitigation_has_an_irreducible_floor():
    assert float(mitigation(t(0.0), 50.0, 0.3)) == pytest.approx(1.0)
    assert float(mitigation(t(10.0), 50.0, 0.3)) == pytest.approx(0.3)


def test_grc_stock_depreciates():
    p = DEFAULTS
    assert float(next_grc(p, t(0.04), t(0.01))) == pytest.approx(0.04 * 0.75 + 0.01)
