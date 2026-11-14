"""The BTC yield vault, one period (design.md sections 2.3-2.6, 3).

Each check is against the design's rules written out by hand with plain
floats, or against a property the design states in words.
"""

import math
from dataclasses import replace

import pytest
import torch

from quant.btc_vault import (
    BtcVaultControls,
    btc_held,
    btc_vault_capital_floor,
    btc_vault_outcomes,
    btc_vault_rule,
)
from quant.firm import crash_probability
from quant.numerics import REFERENCE
from quant.params import DEFAULTS
from quant.quadrature import ShockRule, pareto_jump_rule

P = REFERENCE
BASE = replace(DEFAULTS, periods_per_year=1, jump_rate=0.0)


def t(x):
    return P.tensor(x)


def one(shock):
    """One scenario with probability one: (xi_1, xi_2, eps_BTC[, crash])."""
    shock = list(shock) + [0.0] * (4 - len(shock))
    return ShockRule(nodes=P.tensor([shock]), weights=P.ones(1))


def controls(K=0.3, g=0.0, q=0.0, H=0.0, alloc=(0.5, 0.3, 0.2)):
    return BtcVaultControls(K=t([K]), g=t([g]), q=t([q]), H=t([H]), alloc=t([list(alloc)]))


def phi(a, rho, S):
    return rho + (1 - rho) * math.exp(-a * S)


def by_hand(p, A, c, shock, jump=0.0, S=0.0, bias=None):
    """Closing equity, next size, BTC yield and return, from the design's rules."""
    bias = p.btc_bias if bias is None else bias
    x1, x2, eps, crash = (list(shock) + [0.0])[:4]
    a = c.alloc[0].tolist()
    K, g, q, H = (float(x) for x in (c.K, c.g, c.q, c.H))
    y = sum(a[i] * (p.strategy_yield[i] - phi(p.a_A, p.rho_A, S) * p.strategy_loss[i]
                    - p.strategy_crash[i] * crash) for i in range(3))
    y += phi(p.a_B, p.rho_B, S) * (a[0] * p.strategy_vol[0] * x1 + a[1] * p.strategy_vol[1] * x2)
    hit = jump * sum(a[i] * p.strategy_incident[i] for i in range(3))
    y -= hit
    fee = p.fee_mgmt + p.fee_perf * max(y, 0.0)
    R = y - fee
    co = p.btc_co_invest
    cover = p.compensation * (1 - co) * hit
    prob = 1 - math.exp(-p.btc_crash_rate)
    mu = (1 + p.r_E + bias) / (1 - prob * p.btc_crash_mean) - 1
    price = max(1 + mu + p.s_BTC * eps, 0.0) * (1 - crash)
    R_btc = price - 1
    aum = max(
        A * (1 + R + cover) * price + p.inflow_speed * (p.btc_capacity - A)
        + p.btc_flow_yield * (R + cover / (1 - co) - p.btc_outside_yield) * A
        - (p.outflow_incident * A if jump > 0 else 0.0),
        0.0,
    )
    books = ((K - g - q - co * A) + q * (1 + R_btc - p.btc_holding_cost) - H * (R_btc - p.r - p.perp_premium)
             - 0.5 * p.c_H * H * H - p.tau * K - p.opex)
    wealth = books + (fee + co * (1 + R) - cover) * A * price
    return wealth, aum, y, R


# Scenarios (xi_1, xi_2, eps, crash): calm; strategies up, BTC down; strategies
# down, BTC up; a crash.
SHOCKS = [(0.0, 0.0, 0.0, 0.0), (1.5, 1.0, -1.0, 0.0), (-2.0, -1.5, 1.0, 0.0), (0.5, -0.5, 0.0, 0.4)]


@pytest.mark.parametrize("shock", SHOCKS)
@pytest.mark.parametrize("S", [0.0, 0.02])
def test_accounting_matches_the_formulas(shock, S):
    p = replace(BASE, btc_bias=0.03, c_H=0.1, tau=0.01)
    c = controls(K=0.4, g=0.004, q=0.1, H=0.08, alloc=(0.4, 0.4, 0.2))
    out = btc_vault_outcomes(p, t([0.8]), c, t([S]), one(shock), None)
    w, A, y, R = by_hand(p, 0.8, c, shock, S=S)
    assert float(out.wealth[0, 0]) == pytest.approx(w, abs=1e-14)
    assert float(out.aum[0, 0]) == pytest.approx(A, abs=1e-14)
    assert float(out.yield_[0, 0]) == pytest.approx(y, abs=1e-14)
    assert float(out.ret[0, 0]) == pytest.approx(R, abs=1e-14)


def test_an_incident_costs_the_vault_and_the_firm_repays_a_share():
    p = replace(BASE, jump_rate=0.1)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P, n=4)
    c = controls()
    out = btc_vault_outcomes(p, t([1.0]), c, t([0.0]), one((0.0, 0.0, 0.0)), jumps)
    for j, share in enumerate([0.0] + jumps.severity.tolist()):
        w, A, _, _ = by_hand(p, 1.0, c, (0.0, 0.0, 0.0), jump=share)
        assert float(out.wealth[0, j]) == pytest.approx(w, abs=1e-14)
        assert float(out.aum[0, j]) == pytest.approx(A, abs=1e-14)
    assert float(out.wealth[0, -1]) < float(out.wealth[0, 0])


def test_fees_are_in_btc_so_their_dollar_value_moves_with_the_price():
    """Same BTC yield, different BTC price: the fee's dollar value scales with
    the price, while the firm's dollar costs do not."""
    p = replace(BASE, btc_co_invest=0.0, fee_perf=0.0, btc_crash_rate=0.0)
    c = controls(K=0.3, alloc=(0.0, 0.0, 1.0))
    up = btc_vault_outcomes(p, t([1.0]), c, t([0.0]), one((0.0, 0.0, 1.0)), None)
    down = btc_vault_outcomes(p, t([1.0]), c, t([0.0]), one((0.0, 0.0, -1.0)), None)
    fee = p.fee_mgmt * 1.0
    gap = float(up.wealth - down.wealth)
    assert gap == pytest.approx(fee * 2 * p.s_BTC, abs=1e-14)


def test_cold_custody_carries_only_the_price_and_custody_risk():
    """All in cold custody, no incident: no yield, and investors' BTC falls
    only by the management fee; a crash costs them nothing in BTC."""
    p = BASE
    c = controls(alloc=(0.0, 0.0, 1.0))
    for shock in SHOCKS:
        out = btc_vault_outcomes(p, t([1.0]), c, t([0.0]), one(shock), None)
        assert float(out.yield_) == pytest.approx(0.0, abs=1e-15)
        assert float(out.ret) == pytest.approx(-p.fee_mgmt, abs=1e-15)


def test_strategies_lose_in_a_crash():
    p = BASE
    c = controls(alloc=(0.5, 0.5, 0.0))
    calm = btc_vault_outcomes(p, t([1.0]), c, t([0.0]), one((0.0, 0.0, 0.0, 0.0)), None)
    crash = btc_vault_outcomes(p, t([1.0]), c, t([0.0]), one((0.0, 0.0, 0.0, 0.5)), None)
    expected = 0.5 * (0.5 * p.strategy_crash[0] + 0.5 * p.strategy_crash[1])
    assert float(calm.yield_ - crash.yield_) == pytest.approx(expected, abs=1e-14)


def test_no_performance_fee_on_a_loss():
    p = BASE
    c = controls(alloc=(0.0, 1.0, 0.0))
    out = btc_vault_outcomes(p, t([1.0]), c, t([0.0]), one((0.0, -5.0, 0.0)), None)
    assert float(out.yield_) < 0
    assert float(out.ret) == pytest.approx(float(out.yield_) - p.fee_mgmt, abs=1e-15)


def test_investors_follow_their_btc_yield():
    p = replace(BASE, inflow_speed=0.0, btc_crash_rate=0.0)
    calm = one((0.0, 0.0, 0.0))
    safe = btc_vault_outcomes(p, t([1.0]), controls(alloc=(0.0, 0.0, 1.0)), t([0.0]), calm, None)
    active = btc_vault_outcomes(p, t([1.0]), controls(alloc=(0.0, 1.0, 0.0)), t([0.0]), calm, None)
    assert float(active.aum) > float(safe.aum)


def test_the_bias_moves_only_the_unhedged_btc_and_the_btc_fees():
    """With no crash risk, deciding with the bias and valuing at the market
    differ by b times everything BTC-denominated the firm ends with, less the
    hedge."""
    p = replace(BASE, btc_bias=0.04, btc_crash_rate=0.0)
    c = controls(K=0.4, q=0.2, H=0.05, alloc=(0.0, 0.0, 1.0))
    rule = one((0.0, 0.0, 0.0))
    firm = btc_vault_outcomes(p, t([1.0]), c, t([0.0]), rule, None)
    market = btc_vault_outcomes(p, t([1.0]), c, t([0.0]), rule, None, bias=0.0)
    in_btc = 0.2 - 0.05 + (p.fee_mgmt + p.btc_co_invest * (1 - p.fee_mgmt)) * 1.0
    assert float(firm.wealth - market.wealth) == pytest.approx(0.04 * in_btc, abs=1e-14)


def test_the_hedge_may_cover_the_firm_s_stake():
    p = BASE
    assert float(btc_held(p, t(1.0), t(0.1))) == pytest.approx(0.1 + p.btc_co_invest)


def test_capital_floor_includes_the_stake_and_the_runway():
    p = replace(BASE, k_A=0.02)
    floor = float(btc_vault_capital_floor(p, t(1.0), t(0.01), t(0.2)))
    assert floor == pytest.approx(0.01 + 0.2 + p.btc_co_invest + p.runway * p.opex)


def test_weights_sum_to_one_and_respond_to_grc():
    p = replace(BASE, jump_rate=0.1)
    rule = btc_vault_rule(p, P, n_strategy=3, n_btc=5, n_mixing=3, n_crash=3)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P, n=4)
    c = BtcVaultControls(K=t([0.3, 0.3]), g=t([0.0, 0.0]), q=t([0.0, 0.0]), H=t([0.0, 0.0]),
                         alloc=t([[0.4, 0.4, 0.2], [0.4, 0.4, 0.2]]))
    out = btc_vault_outcomes(p, t([1.0, 1.0]), c, t([0.0, 0.05]), rule, jumps)
    assert torch.allclose(out.weights.sum(-1), P.ones(2), atol=1e-13)
    no_incident = out.weights.reshape(2, -1, 5)[..., 0].sum(-1)
    assert float(no_incident[1]) > float(no_incident[0])
    crashed = float(rule.weights[rule.nodes[:, 3] > 0].sum())
    assert crashed == pytest.approx(crash_probability(p), rel=1e-12)
