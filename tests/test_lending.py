"""The stablecoin lending vault, one period (design.md sections 2.2-2.6, 3).

Each check is against the design's rules written out by hand with plain
floats, or against a property the design states in words.
"""

import math
from dataclasses import replace

import pytest
import torch

from quant.lending import (
    LendingControls,
    btc_prices,
    crash_probability,
    lending_capital_floor,
    lending_outcomes,
    lending_rule,
    lending_shock_correlation,
    loan_book,
    loan_payoff,
)
from quant.numerics import REFERENCE
from quant.params import DEFAULTS
from quant.quadrature import ShockRule, pareto_jump_rule, student_t_rule

P = REFERENCE
BASE = replace(DEFAULTS, periods_per_year=1, jump_rate=0.0)


def t(x):
    return P.tensor(x)


def one(shock):
    """One scenario with probability one: (xi, eps_BTC[, crash])."""
    shock = list(shock) + [0.0] * (3 - len(shock))
    return ShockRule(nodes=P.tensor([shock]), weights=P.ones(1))


def sig(x):
    return 1 / (1 + math.exp(-x))


def controls(K=0.2, g=0.0, q=0.0, H=0.0, rate=0.08, ltv=0.5):
    return LendingControls(K=t([K]), g=t([g]), q=t([q]), H=t([H]), rate=t([rate]), ltv=t([ltv]))


def phi(a, rho, S):
    return rho + (1 - rho) * math.exp(-a * S)


def by_hand(p, A, c, shock, jump=0.0, S=0.0, bias=None):
    """Closing equity, next size, return and bad debt, from the design's rules."""
    bias = p.btc_bias if bias is None else bias
    xi, eps, crash = (list(shock) + [0.0])[:3]
    K, g, q, H, rate, ltv = (float(x) for x in (c.K, c.g, c.q, c.H, c.rate, c.ltv))
    demand = p.loan_demand * max(1 - rate / p.rate_max, 0) * max(1 + p.ltv_demand * (ltv - p.ltv_ref), 0)
    L = min(demand, A)
    prob = 1 - math.exp(-p.btc_crash_rate)
    mu = (1 + p.r_E + bias) / (1 - prob * p.btc_crash_mean) - 1
    smooth = max(1 + mu + p.s_BTC * eps, 0.0)
    price = smooth * (1 - crash)
    R_btc = price - 1
    owed = 1 + rate
    T = owed * ltv / p.liq_threshold
    d = p.sale_discount * phi(p.a_L, p.rho_L, S)
    liq = sig((T - price) / p.liq_band)
    first = sig((T - smooth) / p.liq_band)
    proceeds = (first * T + (1 - first) * min(T, price)) * (1 - d) / ltv
    payoff = liq * min(owed * (1 + p.liq_penalty), proceeds) + (1 - liq) * owed
    bad = liq * max(owed - proceeds, 0.0)
    ops = A * (phi(p.a_A, p.rho_A, S) * p.lend_loss + phi(p.a_B, p.rho_B, S) * p.lend_vol * xi)
    vault = (A - L) + L * payoff - ops - jump * A
    fee = p.interest_share * L * min(max(payoff - 1, 0.0), rate)
    R = (vault - fee) / A - 1
    co = p.lend_co_invest
    cover = p.compensation * (1 - co) * (L * bad + jump * A)
    received = R + cover / ((1 - co) * A)
    aum = max(
        A * (1 + R) + cover + p.inflow_speed * (p.lend_capacity - A)
        + p.flow_yield * (received - p.outside_rate) * A
        - (p.outflow_incident * A if jump > 0 else 0.0),
        0.0,
    )
    wealth = ((K - g - q - co * A) + q * (1 + R_btc - p.btc_holding_cost) - H * (R_btc - p.r) - 0.5 * p.c_H * H * H
              - p.tau * K - p.opex + fee + co * A * (1 + R) - cover)
    return wealth, aum, R, L * bad


# Scenarios (xi, eps, crash): calm; a mild fall; a smooth fall past the
# threshold; a crash that jumps past it; a smooth fall past it, then a crash.
SHOCKS = [(0.0, 0.0, 0.0), (1.0, -0.3, 0.0), (-0.5, -0.9, 0.0), (0.5, 0.0, 0.6), (0.0, -1.0, 0.5)]


@pytest.mark.parametrize("shock", SHOCKS)
@pytest.mark.parametrize("S", [0.0, 0.02])
def test_accounting_matches_the_formulas(shock, S):
    p = replace(BASE, btc_bias=0.03, c_H=0.1, tau=0.01)
    c = controls(K=0.3, g=0.005, q=0.1, H=0.04, rate=0.09, ltv=0.6)
    out = lending_outcomes(p, t([0.8]), c, t([S]), one(shock), None)
    w, A, R, bad = by_hand(p, 0.8, c, shock, S=S)
    assert float(out.wealth[0, 0]) == pytest.approx(w, abs=1e-14)
    assert float(out.aum[0, 0]) == pytest.approx(A, abs=1e-14)
    assert float(out.ret[0, 0]) == pytest.approx(R, abs=1e-14)
    assert float(out.bad_debt[0, 0]) == pytest.approx(bad, abs=1e-14)


SHARP = replace(BASE, liq_band=1e-9)   # the unsmoothed rule, for checks of its cases


def prices(p, eps, crash=0.0):
    return btc_prices(p, t(eps), t(crash), 0.0)


def test_the_test_shocks_cover_every_case():
    """Repaid; liquidated at the threshold; a crash past it, with bad debt."""
    p = SHARP
    rate, ltv = t(0.09), t(0.6)
    cases = []
    for _, eps, crash in SHOCKS:
        payoff, bad = loan_payoff(p, rate, ltv, *prices(p, eps, crash), t(p.sale_discount))
        cases.append((float(payoff) == pytest.approx(1.09), float(bad) > 0))
    assert (True, False) in cases                  # repaid
    assert (False, False) in cases                 # liquidated, collateral covers
    assert (False, True) in cases                  # crash past the threshold: bad debt


def test_without_a_crash_liquidation_never_leaves_bad_debt():
    """Prices are watched continuously, so a smooth fall is caught at the
    threshold, however far it goes; the vault gets what is owed plus the
    penalty, and the rest goes back to the borrower."""
    p = SHARP
    rate, ltv = t(0.05), t(0.7)
    for eps in (-0.5, -1.0, -1.5, -5.0):
        payoff, bad = loan_payoff(p, rate, ltv, *prices(p, eps), t(0.1))
        assert float(payoff) == pytest.approx(1.05 * (1 + p.liq_penalty))
        assert float(bad) == 0.0


def test_a_crash_sells_at_the_price_after_it():
    """A crash that jumps past the threshold leaves the collateral to be sold
    at the lower price: bad debt is what even all of it cannot cover."""
    p = SHARP
    rate, ltv, d = t(0.05), t(0.5), t(0.1)
    for crash in torch.linspace(0.0, 0.9, 46, dtype=P.dtype):
        price, smooth = prices(p, 0.0, float(crash))
        _, bad = loan_payoff(p, rate, ltv, price, smooth, d)
        T = 1.05 * 0.5 / p.liq_threshold
        short = 1.05 - min(T, float(price)) * 0.9 / 0.5 if float(price) < T else 0.0
        assert float(bad) == pytest.approx(max(short, 0.0), abs=1e-14)


def test_a_fall_caught_before_a_crash_sells_at_the_threshold():
    p = SHARP
    price, smooth = prices(p, -1.0, 0.5)
    T = 1.05 * 0.5 / p.liq_threshold
    assert float(smooth) < T
    payoff, bad = loan_payoff(p, t(0.05), t(0.5), price, smooth, t(0.1))
    assert float(bad) == 0.0 and float(payoff) == pytest.approx(1.05 * (1 + p.liq_penalty))


def test_btc_is_expected_to_earn_the_market_return_plus_the_bias():
    """Crash risk is priced into the smooth part, so the expected end price is
    1 + r_E + bias (up to the crash rule's small error in its mean). Quarterly:
    at a year's volatility the floor at a zero price starts to lift the mean."""
    for bias in (0.0, 0.03):
        p = replace(DEFAULTS, nu=math.inf, btc_bias=bias).per_period()
        bias = p.btc_bias
        rule = lending_rule(p, P, n_ops=1, n_btc=9, n_mixing=1, n_crash=8)
        price, _ = btc_prices(p, rule.nodes[:, 1], rule.nodes[:, 2], bias)
        assert float(rule.expect(price)) == pytest.approx(1 + p.r_E + bias, rel=1e-3)


def test_the_scenario_rule_folds_in_the_crash_chance():
    p = BASE
    rule = lending_rule(p, P, n_ops=3, n_btc=7, n_mixing=4, n_crash=5)
    assert rule.size == 3 * 7 * 4 * 6
    assert float(rule.weights.sum()) == pytest.approx(1.0, abs=1e-13)
    crashed = float(rule.weights[rule.nodes[:, 2] > 0].sum())
    assert crashed == pytest.approx(crash_probability(p), rel=1e-12)


def test_the_smoothed_trigger_is_continuous_and_has_a_slope_in_ltv():
    """No jump at the threshold, and the payoff responds to the LTV there."""
    p = BASE
    rate = t(0.05)
    trigger = 1.05 * 0.6 / p.liq_threshold
    grid = torch.linspace(trigger - 0.05, trigger + 0.05, 201, dtype=P.dtype)
    payoff, _ = loan_payoff(p, rate, t(0.6), grid, grid, t(0.1))
    assert float(payoff.diff().abs().max()) < 0.05 * 1.05 * p.liq_penalty   # the sharp jump, 20x over
    ltv = t(0.6).requires_grad_(True)
    near, _ = loan_payoff(p, rate, ltv, t(trigger), t(trigger), t(0.1))
    (slope,) = torch.autograd.grad(near, ltv)
    assert float(slope) > 0.0     # a higher LTV moves the threshold up: more liquidated, more penalty


def test_grc_narrows_the_sale_discount():
    """More GRC stock, smaller discount, less bad debt in the same crash."""
    p = BASE
    c = controls(rate=0.08, ltv=0.7)
    crash = one((0.0, 0.0, 0.6))
    none = lending_outcomes(p, t([1.0]), c, t([0.0]), crash, None)
    some = lending_outcomes(p, t([1.0]), c, t([0.05]), crash, None)
    assert float(none.bad_debt) > float(some.bad_debt) > 0.0


def test_the_loan_book_is_demand_or_the_vault_whichever_is_smaller():
    p = BASE
    assert float(loan_book(p, t(1.0), controls(rate=0.0, ltv=0.5))) == pytest.approx(1.0)
    assert float(loan_book(p, t(2.0), controls(rate=0.0, ltv=0.5))) == pytest.approx(1.5)
    half = float(loan_book(p, t(2.0), controls(rate=0.10, ltv=0.5)))
    assert half == pytest.approx(0.75)                       # half of 1.5 at half of rate_max
    more = float(loan_book(p, t(2.0), controls(rate=0.10, ltv=0.6)))
    assert more == pytest.approx(0.75 * 1.1)                 # +10% per 10 points of LTV


def test_idle_stablecoins_earn_nothing():
    """No demand above rate_max: the vault only loses its operating costs, and
    the firm's stablecoins come back unchanged."""
    p = replace(BASE, lend_loss=0.0, lend_vol=0.0, opex=0.0, lend_co_invest=0.0)
    out = lending_outcomes(p, t([1.0]), controls(K=0.2, rate=0.25), t([0.0]), one((0.0, 0.0)), None)
    assert float(out.loans) == 0.0
    assert float(out.ret) == pytest.approx(0.0, abs=1e-15)
    assert float(out.wealth) == pytest.approx(0.2, abs=1e-15)


def test_fully_hedged_btc_earns_the_riskless_rate_less_its_security_cost():
    p = replace(BASE, opex=0.0, lend_co_invest=0.0, loan_demand=0.0)
    c = controls(K=0.3, q=0.2, H=0.2)
    for eps in (-2.0, 0.0, 1.5):
        out = lending_outcomes(p, t([1.0]), c, t([0.0]), one((0.0, eps)), None)
        assert float(out.wealth) == pytest.approx(0.1 + 0.2 * (1 + p.r - p.btc_holding_cost), abs=1e-14)


def test_the_bias_moves_only_the_unhedged_btc():
    """Deciding with the bias and valuing at the market differ by (q - H) b."""
    p = replace(BASE, btc_bias=0.04, loan_demand=0.0, btc_crash_rate=0.0)
    c = controls(K=0.3, q=0.2, H=0.05)
    rule = one((0.0, 0.0))
    firm = lending_outcomes(p, t([1.0]), c, t([0.0]), rule, None)
    market = lending_outcomes(p, t([1.0]), c, t([0.0]), rule, None, bias=0.0)
    assert float(firm.wealth - market.wealth) == pytest.approx(0.15 * 0.04, abs=1e-14)


def test_btc_cannot_fall_below_zero():
    price, smooth = prices(BASE, -10.0, 0.5)
    assert float(price) == 0.0 and float(smooth) == 0.0


def test_an_incident_costs_the_vault_and_the_firm_repays_a_share():
    p = replace(BASE, jump_rate=0.1)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P, n=4)
    c = controls()
    out = lending_outcomes(p, t([1.0]), c, t([0.0]), one((0.0, 0.0)), jumps)
    for j, share in enumerate([0.0] + jumps.severity.tolist()):
        w, A, _, _ = by_hand(p, 1.0, c, (0.0, 0.0), jump=share)
        assert float(out.wealth[0, j]) == pytest.approx(w, abs=1e-14)
        assert float(out.aum[0, j]) == pytest.approx(A, abs=1e-14)
    assert float(out.wealth[0, -1]) < float(out.wealth[0, 0])


def test_providers_follow_their_yield():
    """A higher rate on a full book means a better yield and a larger vault,
    all else equal; flows respond to the gap with the outside rate."""
    p = replace(BASE, loan_demand=10.0, inflow_speed=0.0)
    calm = one((0.0, 0.0))
    low = lending_outcomes(p, t([1.0]), controls(rate=0.02), t([0.0]), calm, None)
    high = lending_outcomes(p, t([1.0]), controls(rate=0.10), t([0.0]), calm, None)
    assert float(high.aum) > float(low.aum)


def test_weights_sum_to_one_and_respond_to_grc():
    p = replace(BASE, jump_rate=0.1)
    rule = lending_rule(p, P, n_ops=3, n_btc=7, n_mixing=4, n_crash=3)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P, n=4)
    c = LendingControls(K=t([0.2, 0.2]), g=t([0.0, 0.0]), q=t([0.0, 0.0]), H=t([0.0, 0.0]),
                        rate=t([0.08, 0.08]), ltv=t([0.5, 0.5]))
    out = lending_outcomes(p, t([1.0, 1.0]), c, t([0.0, 0.05]), rule, jumps)
    assert torch.allclose(out.weights.sum(-1), P.ones(2), atol=1e-13)
    no_jump = out.weights.reshape(2, -1, 5)[..., 0].sum(-1)
    assert float(no_jump[1]) > float(no_jump[0])


def test_capital_floor_takes_the_tightest_constraint():
    p = replace(BASE, k_A=0.02, k_BTC=0.1, lend_co_invest=0.05, runway=0.0)
    assert float(lending_capital_floor(p, t(1.0), t(0.0), t(0.0))) == pytest.approx(0.05)    # stake
    assert float(lending_capital_floor(p, t(1.0), t(0.01), t(0.2))) == pytest.approx(0.26)   # cash
    p = replace(p, lend_co_invest=0.0)
    assert float(lending_capital_floor(p, t(1.0), t(0.0), t(0.01))) == pytest.approx(0.021)   # own funds


def test_the_runway_keeps_a_year_of_operating_costs_in_stablecoins():
    """Whatever the frequency, the reserve is the same: a year of costs."""
    for n in (1, 4):
        p = replace(DEFAULTS, periods_per_year=n, k_A=0.0, lend_co_invest=0.0).per_period()
        floor = float(lending_capital_floor(p, t(1.0), t(0.0), t(0.1)))
        assert floor == pytest.approx(0.1 + DEFAULTS.runway * DEFAULTS.opex, rel=1e-12)


def test_per_direction_quadrature_keeps_the_moments():
    rule = student_t_rule(lending_shock_correlation(BASE), math.inf, P, n_hermite=(3, 15), n_mixing=1)
    assert rule.size == 45
    second = (rule.weights[:, None] * rule.nodes**2).sum(0)
    assert torch.allclose(second, P.ones(2), atol=1e-13)
