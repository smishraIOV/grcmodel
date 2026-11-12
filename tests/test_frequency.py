"""Decision frequency: annual statements converted to per-period ones (design.md 2.6)."""

import math
from dataclasses import replace

import pytest

from quant.numerics import REFERENCE
from quant.params import DEFAULTS
from quant.solver import riskless_steady_state, solve
from tests.test_solver import BASE, SPEC

P = REFERENCE


def test_annual_conversion_is_the_identity():
    p = replace(DEFAULTS, periods_per_year=1)
    assert p.per_period() is p


def test_conversion_is_idempotent():
    q = DEFAULTS.per_period()
    assert q.per_period() == q
    assert q.periods_per_year == 1


def test_quarters_add_back_up_to_the_year():
    n = 4
    p = replace(DEFAULTS, periods_per_year=n)
    q = p.per_period()
    assert (1 + q.r) ** n == pytest.approx(1 + p.r, rel=1e-14)
    assert (1 + q.r_E) ** n == pytest.approx(1 + p.r_E, rel=1e-14)
    assert q.A ** n == pytest.approx(p.A, rel=1e-14)
    assert n * q.s_P**2 == pytest.approx(p.s_P**2, rel=1e-14)       # variances add
    assert n * q.m == pytest.approx(p.m, rel=1e-14)
    assert n * q.jump_rate == pytest.approx(p.jump_rate, rel=1e-14)
    assert n * q.mu_BTC == pytest.approx(p.mu_BTC, rel=1e-14)
    assert (1 - q.delta) ** n == pytest.approx(1 - p.delta, rel=1e-14)


def test_sizes_do_not_convert():
    p, q = DEFAULTS, DEFAULTS.per_period()
    for name in ("lam", "kappa", "D", "k", "jump_mean", "jump_tail", "a_A", "a_J", "rho_A", "alpha_max", "q_max", "k_BTC", "nu", "rho_NP"):
        assert getattr(q, name) == getattr(p, name), name


def test_first_best_book_stays_at_one_every_quarter():
    q = DEFAULTS.per_period()
    assert q.b > 0
    assert (q.A - q.b * 1.0) == pytest.approx(1 + q.r, rel=1e-14)    # F'(1) = 1 + r


def test_quarterly_riskless_bank_matches_the_closed_form():
    """The exact no-shock oracle, at quarterly frequency."""
    import torch

    p = replace(
        BASE, s_P=0.0, s_N=0.0, s_BTC=0.0, a_A=0.0, a_B=0.0, nu=math.inf, periods_per_year=4,
    )
    # Fine grid: a quarter's profit is small, so the steady state sits just
    # above the kink where issuance starts, and on 33 points linear
    # interpolation across that kink understated V by 0.006. On 257 the error
    # is 1.6e-6. One scenario node keeps the fine grid cheap.
    sol = solve(p, P, replace(SPEC, n_w=257, n_hermite=1, n_mixing=1), improve_steps=80)
    exact = riskless_steady_state(p, P)
    above = sol.grid.w >= exact.K + 0.05
    got, want = sol.V[above, 0], exact.value(sol.grid.w[above])
    assert sol.inputs.periods_per_year == 4 and sol.params.periods_per_year == 1
    assert torch.allclose(got, want, atol=1e-5), float((got - want).abs().max())


def test_quarterly_solve_converges():
    sol = solve(replace(BASE, periods_per_year=4), P, SPEC)
    assert sol.residual < 1e-6
    assert sol.lower_edge_is_walk_away and sol.upper_edge_pays_out
