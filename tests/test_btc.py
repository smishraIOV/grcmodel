"""BTC as the firm's home asset: a believed excess return, a spot treasury,
and hedging with perps (design.md section 2.2)."""

from dataclasses import replace

import pytest
import torch

from quant.model import Controls, capital_floor, net_btc, next_wealth, shock_correlation
from quant.numerics import REFERENCE
from quant.params import DEFAULTS
from quant.quadrature import student_t_rule
from quant.solver import solve
from quant.twoperiod import (
    ORACLE_FIRST_PERIOD,
    first_period_wealth,
    froot_stein,
    issuance_threshold,
    optimal_positions,
    oracle_params,
)
from tests.test_solver import BASE, SPEC

P = REFERENCE


def t(x):
    return P.tensor(x)


def test_a_view_on_btc_matches_eq_15():
    """With a believed premium the bank keeps x* = mu_BTC / (G s_BTC^2) of net
    BTC exposure -- the paper's eq. (15) case -- found by an optimizer that
    does not know the formula. Exact under normal shocks."""
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


def test_the_treasury_uses_cash_and_capital():
    """Bought spot: it competes with the book for cash and carries k_BTC."""
    p = replace(DEFAULTS, k=0.08, k_BTC=0.10, D=0.9)
    I, g = t(0.5), t(0.0)
    assert float(capital_floor(p, I, g, t(0.3))) == pytest.approx(0.08 * 0.5 + 0.10 * 0.3)
    # cash binds once book + treasury exceed deposits + equity
    assert float(capital_floor(p, t(0.8), g, t(0.4))) == pytest.approx(0.8 + 0.4 - 0.9)


def test_btc_earns_its_believed_premium_on_net_exposure_only():
    """The treasury's riskless part cancels against the cash it used; what is
    left is net exposure times (mu_BTC + shock), whoever holds it."""
    p = replace(DEFAULTS, mu_BTC=0.10, periods_per_year=1)
    zero = P.zeros(1, 3)
    base = Controls(K=t(0.2), g=t(0.0), I=t(0.5), alpha=t(0.0), H=t(0.25))
    more = Controls(K=t(0.2), g=t(0.0), I=t(0.5), alpha=t(0.0), H=t(0.25), q=t(0.1))
    gain = next_wealth(p, more, t(0.0), zero) - next_wealth(p, base, t(0.0), zero)
    assert float(net_btc(p, more) - net_btc(p, base)) == pytest.approx(0.1)
    assert float(gain) == pytest.approx(0.1 * p.mu_BTC, abs=1e-14)


def test_with_a_view_the_bank_stays_long_btc():
    """A risk-averse bank with a positive view keeps positive net exposure."""
    sol = solve(replace(BASE, mu_BTC=0.05), P, SPEC)
    c = sol.controls()
    w, _ = sol.grid.states()
    safe = w >= 0.0
    assert bool((net_btc(sol.params, c)[safe] > 0).all())
