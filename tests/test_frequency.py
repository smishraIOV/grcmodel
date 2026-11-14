"""Decision frequency: annual statements converted to per-period ones (design.md 4.4)."""

from dataclasses import replace

import pytest

from quant.params import DEFAULTS


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
    assert n * q.s_BTC**2 == pytest.approx(p.s_BTC**2, rel=1e-14)   # variances add
    assert n * q.jump_rate == pytest.approx(p.jump_rate, rel=1e-14)
    assert n * q.mu_BTC == pytest.approx(p.mu_BTC, rel=1e-14)
    assert n * q.fee_mgmt == pytest.approx(p.fee_mgmt, rel=1e-14)
    assert n * q.opex == pytest.approx(p.opex, rel=1e-14)
    assert [n * x for x in q.vault_mu] == pytest.approx(list(p.vault_mu), rel=1e-14)
    assert [n * x * x for x in q.vault_vol] == pytest.approx([x * x for x in p.vault_vol], rel=1e-14)
    assert (1 - q.inflow_speed) ** n == pytest.approx(1 - p.inflow_speed, rel=1e-14)
    assert (1 - q.delta) ** n == pytest.approx(1 - p.delta, rel=1e-14)
    assert (1 + q.rate_max) ** n == pytest.approx(1 + p.rate_max, rel=1e-14)
    assert (1 + q.outside_rate) ** n == pytest.approx(1 + p.outside_rate, rel=1e-14)
    assert n * replace(p, btc_bias=0.04).per_period().btc_bias == pytest.approx(0.04, rel=1e-14)
    assert n * q.lend_loss == pytest.approx(p.lend_loss, rel=1e-14)
    assert q.runway * q.opex == pytest.approx(p.runway * p.opex, rel=1e-14)
    assert n * q.lend_vol**2 == pytest.approx(p.lend_vol**2, rel=1e-14)


def test_sizes_do_not_convert():
    p, q = DEFAULTS, DEFAULTS.per_period()
    for name in ("lam", "kappa", "jump_mean", "jump_tail", "a_A", "a_J", "rho_A", "q_max", "k_BTC", "nu", "fee_perf", "capacity", "k_A", "compensation", "outflow_incident", "outflow_perf", "co_invest", "loan_demand", "ltv_demand", "ltv_max", "liq_threshold", "liq_penalty", "sale_discount", "interest_share", "flow_yield", "lend_co_invest", "lend_capacity"):
        assert getattr(q, name) == getattr(p, name), name
