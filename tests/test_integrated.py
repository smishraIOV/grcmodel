"""The integrated firm, one period (design.md sections 2.1-2.6;
docs/plan-integrated.md section 2).

Checked against the rules written out by hand with plain floats, against each
vault's own module where the integrated firm must reduce to it, and against
the properties the design states in words.
"""

import math
from dataclasses import replace

import pytest
import torch

from quant.btc_vault import BtcVaultControls, btc_vault_outcomes
from quant.integrated import (
    IntegratedControls,
    integrated_btc_held,
    integrated_capital_floor,
    integrated_loan_book,
    integrated_outcomes,
    integrated_rule,
)
from quant.lending import LendingControls, lending_outcomes
from quant.numerics import REFERENCE
from quant.params import DEFAULTS, INTEGRATED_DEFAULTS
from quant.quadrature import ShockRule, pareto_jump_rule

P = REFERENCE
BASE = replace(INTEGRATED_DEFAULTS, periods_per_year=1, jump_rate=0.0)
ALLOC = (0.3, 0.3, 0.1, 0.3)


def t(x):
    return P.tensor(x)


def one(shock):
    """One scenario with probability one: (xi_L, xi_1, xi_2, eps_BTC, crash)."""
    return ShockRule(nodes=P.tensor([list(shock)]), weights=P.ones(1))


def controls(K=0.5, g=0.0, q=0.0, H=0.0, rate=0.08, ltv=0.5, alloc=ALLOC):
    return IntegratedControls(
        K=t([K]), g=t([g]), q=t([q]), H=t([H]), rate=t([rate]), ltv=t([ltv]), alloc=t([list(alloc)])
    )


def sig(x):
    return 1 / (1 + math.exp(-x))


def phi(a, rho, S):
    return rho + (1 - rho) * math.exp(-a * S)


def by_hand(p, A_L, A_B, c, shock, J_L=0.0, J_B=0.0, S=0.0):
    """Closing equity and both next sizes, from the design's rules."""
    xi_l, xi_1, xi_2, eps, crash = shock
    K, g, q, H, rate, ltv = (float(x) for x in (c.K, c.g, c.q, c.H, c.rate, c.ltv))
    a = [float(x) for x in c.alloc[0]]
    # BTC's price
    prob = 1 - math.exp(-p.btc_crash_rate)
    m = math.log((1 + p.r_E + p.btc_bias) / (1 - prob * p.btc_crash_mean)) - p.s_BTC**2 / 2
    smooth = math.exp(m + p.s_BTC * eps)
    X = smooth * (1 - crash)
    # the loan book
    supply = A_L + a[3] * A_B
    demand = p.loan_demand * max(1 - rate / p.rate_max, 0) * max(1 + p.ltv_demand * (ltv - p.ltv_ref), 0)
    L = min(demand, supply)
    u = L / supply
    owed = 1 + rate
    T = owed * ltv / p.liq_threshold
    liquidated, first = sig((T - X) / p.liq_band), sig((T - smooth) / p.liq_band)
    caught = T - p.liq_slippage * phi(p.a_L, p.rho_L, S) * max(T - smooth, 0.0)
    sale = first * caught + (1 - first) * min(T, X)
    proceeds = sale * (1 - p.sale_discount * phi(p.a_L, p.rho_L, S)) / ltv
    payoff = liquidated * min(owed * (1 + p.liq_penalty), proceeds) + (1 - liquidated) * owed
    B = max(owed - proceeds, 0) * liquidated
    earned = min(max(payoff - 1, 0), rate)
    ops = phi(p.a_A, p.rho_A, S) * p.lend_loss + phi(p.a_B, p.rho_B, S) * p.lend_vol * xi_l
    gain = u * (payoff - 1) - ops - p.interest_share * u * earned - J_L
    # the dollar vault
    co_l, co_b, psi = p.lend_co_invest, p.btc_co_invest, p.compensation
    cover_l = psi * (1 - co_l) * (u * B + J_L) * A_L
    A_L_next = max(
        A_L * (1 + gain) + cover_l + p.inflow_speed * (p.lend_capacity - A_L)
        + p.flow_yield * (gain + cover_l / ((1 - co_l) * A_L) - p.outside_rate) * A_L
        - (J_L > 0) * p.outflow_incident * A_L
        - (J_B > 0) * p.outflow_contagion * A_B / (A_L + A_B) * A_L, 0.0)
    # the BTC vault, in BTC terms
    y4 = (gain - p.r - p.perp_premium) / X
    xis = (xi_1, xi_2, 0.0)
    y = sum(
        a[i] * (p.strategy_yield[i] - phi(p.a_A, p.rho_A, S) * p.strategy_loss[i] - p.strategy_crash[i] * crash
                + phi(p.a_B, p.rho_B, S) * p.strategy_vol[i] * xis[i])
        for i in range(3)
    ) + a[3] * y4
    exposed = J_B * sum(a[i] * p.strategy_incident[i] for i in range(3))
    y -= exposed
    fee = p.fee_mgmt + p.fee_perf * max(y, 0)
    R_b = y - fee
    cover_b = psi * (1 - co_b) * (exposed + a[3] * (u * B + J_L) / X)
    A_B_next = max(
        A_B * (1 + R_b + cover_b) * X + p.inflow_speed * (p.btc_capacity - A_B)
        + p.btc_flow_yield * (R_b + cover_b / (1 - co_b) - p.btc_outside_yield) * A_B
        - (J_B > 0) * p.outflow_incident * A_B
        - (J_L > 0) * p.outflow_contagion * A_L / (A_L + A_B) * A_B, 0.0)
    # the firm
    stake = co_l * A_L + co_b * A_B
    wealth = (
        (K - g - q - stake) + q * (X - p.btc_holding_cost) - H * (X - 1 - p.r - p.perp_premium)
        - 0.5 * p.c_H * H * H - p.tau * K - p.opex
        + p.interest_share * u * earned * supply
        + co_l * A_L * (1 + gain) - cover_l
        + (fee + co_b * (1 + R_b) - cover_b) * A_B * X
    )
    return wealth, A_L_next, A_B_next


SHOCKS = [
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, -0.5, 0.8, 0.5, 0.0),
    (-1.0, 1.2, -0.7, -0.6, 0.0),     # a smooth fall, liquidated at the threshold
    (0.5, 0.3, 0.3, 0.2, 0.55),       # a crash past the threshold: bad debt
    (0.0, 0.0, 0.0, -0.9, 0.3),       # caught at the threshold, then a crash
]


@pytest.mark.parametrize("shock", SHOCKS)
@pytest.mark.parametrize("S", [0.0, 0.03])
@pytest.mark.parametrize("rate", [0.08, 0.02])   # supply short of demand, and demand short of supply
def test_accounting_matches_the_formulas(shock, S, rate):
    p = replace(BASE, btc_bias=0.03, c_H=0.1, tau=0.01)
    c = controls(K=0.6, g=0.01, q=0.2, H=0.1, rate=rate, ltv=0.6)
    out = integrated_outcomes(p, t([0.8]), t([1.7]), c, t([S]), one(shock), None)
    wealth, A_L, A_B = by_hand(p, 0.8, 1.7, c, shock, S=S)
    assert float(out.wealth) == pytest.approx(wealth, abs=1e-14)
    assert float(out.aum_l) == pytest.approx(A_L, abs=1e-14)
    assert float(out.aum_b) == pytest.approx(A_B, abs=1e-14)


def test_the_test_shocks_cover_bad_debt_and_idle_stablecoins():
    p = BASE
    out = integrated_outcomes(p, t([0.8]), t([1.7]), controls(ltv=0.6), t([0.0]), one(SHOCKS[3]), None)
    assert float(out.bad_debt) > 0.01
    assert float(integrated_loan_book(p, t([0.8]), t([1.7]), controls(rate=0.02))[1]) == 1.0
    assert float(integrated_loan_book(p, t([0.8]), t([1.7]), controls(rate=0.15))[1]) < 0.5


def test_incident_branches_match_the_formulas_and_their_weights_sum_to_one():
    p = replace(BASE, jump_rate=0.1)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P, n=3)
    c = controls(K=0.6, q=0.1, ltv=0.6)
    shock = SHOCKS[3]
    out = integrated_outcomes(p, t([0.8]), t([1.7]), c, t([0.02]), one(shock), jumps)
    assert out.wealth.shape == (1, 7)
    assert float(out.weights.sum()) == pytest.approx(1.0, abs=1e-15)
    cases = [(0.0, 0.0)] + [(float(j), 0.0) for j in jumps.severity] + [(0.0, float(j)) for j in jumps.severity]
    for k, (J_L, J_B) in enumerate(cases):
        wealth, A_L, A_B = by_hand(p, 0.8, 1.7, c, shock, J_L=J_L, J_B=J_B, S=0.02)
        assert float(out.wealth[0, k]) == pytest.approx(wealth, abs=1e-14)
        assert float(out.aum_l[0, k]) == pytest.approx(A_L, abs=1e-14)
        assert float(out.aum_b[0, k]) == pytest.approx(A_B, abs=1e-14)


def test_the_incident_chance_keeps_a_gradient_in_grc_spend():
    p = replace(BASE, jump_rate=0.1)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P, n=3)
    S = P.tensor([0.01], requires_grad=True)
    out = integrated_outcomes(p, t([0.8]), t([1.7]), controls(), S, one(SHOCKS[0]), jumps)
    (no_incident,) = torch.autograd.grad(out.weights[0, 0], S)
    assert float(no_incident) > 0.0


def _rules(p, n_jump=3):
    rule = integrated_rule(p, P, n_ops=2, n_strategy=2, n_btc=5, n_mixing=3, n_crash=3)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P, n=n_jump)
    return rule, jumps


def test_with_no_btc_vault_it_is_the_lending_vault_alone():
    """Scenario by scenario, and with the lending vault's probabilities."""
    p = replace(BASE, jump_rate=0.1, btc_bias=0.02)
    rule, jumps = _rules(p)
    A = t([0.3, 0.9, 1.4])
    S = t([0.0, 0.02, 0.05])
    args = dict(K=t([0.3, 0.4, 0.5]), g=t([0.0, 0.01, 0.0]), q=t([0.1, 0.0, 0.2]), H=t([0.05, 0.0, 0.2]),
                rate=t([0.04, 0.08, 0.12]), ltv=t([0.4, 0.6, 0.7]))
    c = IntegratedControls(**args, alloc=t([list(ALLOC)] * 3))
    out = integrated_outcomes(p, A, torch.zeros_like(A), c, S, rule, jumps)
    alone_rule = ShockRule(nodes=rule.nodes[:, [0, 3, 4]], weights=rule.weights)
    alone = lending_outcomes(p, A, LendingControls(**args), S, alone_rule, jumps)

    n, j = rule.size, jumps.severity.shape[0]
    def split(x):
        return x.reshape(3, n, 1 + 2 * j)
    lend = slice(0, 1 + j)       # no incident, then lending-system incidents
    assert torch.allclose(split(out.wealth)[..., lend], alone.wealth.reshape(3, n, 1 + j), atol=1e-13, rtol=0)
    assert torch.allclose(split(out.aum_l)[..., lend], alone.aum.reshape(3, n, 1 + j), atol=1e-13, rtol=0)
    # an incident in the (empty) BTC vault changes nothing
    for k in range(j):
        assert torch.equal(split(out.wealth)[..., 1 + j + k], split(out.wealth)[..., 0])
        assert torch.equal(split(out.aum_l)[..., 1 + j + k], split(out.aum_l)[..., 0])
    w = split(out.weights)
    merged = torch.cat([w[..., :1] + w[..., 1 + j:].sum(-1, keepdim=True), w[..., 1:1 + j]], dim=-1)
    assert torch.allclose(merged, alone.weights.reshape(3, n, 1 + j), atol=1e-15, rtol=0)
    assert float((out.expect(out.wealth) - alone.expect(alone.wealth)).abs().max()) < 1e-13


def test_with_no_dollar_vault_and_no_sleeve_it_is_the_btc_vault_alone():
    p = replace(BASE, jump_rate=0.1, btc_bias=0.02, lend_capacity=0.0)
    rule, jumps = _rules(p)
    A = t([0.4, 1.2, 2.1])
    S = t([0.0, 0.02, 0.05])
    args = dict(K=t([0.3, 0.4, 0.5]), g=t([0.0, 0.01, 0.0]), q=t([0.1, 0.0, 0.2]), H=t([0.05, 0.0, 0.2]))
    alloc = t([[0.5, 0.3, 0.2], [0.1, 0.9, 0.0], [0.0, 0.2, 0.8]])
    c = IntegratedControls(**args, rate=t([0.04, 0.08, 0.12]), ltv=t([0.4, 0.6, 0.7]),
                           alloc=torch.cat([alloc, torch.zeros(3, 1, dtype=alloc.dtype)], dim=-1))
    out = integrated_outcomes(p, torch.zeros_like(A), A, c, S, rule, jumps)
    alone_rule = ShockRule(nodes=rule.nodes[:, 1:], weights=rule.weights)
    alone = btc_vault_outcomes(p, A, BtcVaultControls(**args, alloc=alloc), S, alone_rule, jumps)

    n, j = rule.size, jumps.severity.shape[0]
    def split(x):
        return x.reshape(3, n, 1 + 2 * j)
    own = [0] + list(range(1 + j, 1 + 2 * j))    # no incident, then BTC-vault incidents
    assert torch.allclose(split(out.wealth)[..., own], alone.wealth.reshape(3, n, 1 + j), atol=1e-13, rtol=0)
    assert torch.allclose(split(out.aum_b)[..., own], alone.aum.reshape(3, n, 1 + j), atol=1e-13, rtol=0)
    for k in range(j):
        assert torch.equal(split(out.wealth)[..., 1 + k], split(out.wealth)[..., 0])
    assert float(out.aum_l.abs().max()) == 0.0
    w = split(out.weights)
    merged = torch.cat([w[..., :1] + w[..., 1:1 + j].sum(-1, keepdim=True), w[..., 1 + j:]], dim=-1)
    assert torch.allclose(merged, alone.weights.reshape(3, n, 1 + j), atol=1e-15, rtol=0)


@pytest.mark.parametrize("shock", SHOCKS)
def test_both_sources_earn_the_same_on_every_dollar(shock):
    """Pro-rata sharing: with no BTC-vault fees, no funding and BTC's price
    aside, a sleeve dollar earns exactly what a dollar-vault dollar earns."""
    p = replace(BASE, fee_mgmt=0.0, fee_perf=0.0, r=0.0, perp_premium=0.0)
    c = controls(ltv=0.6, alloc=(0.0, 0.0, 0.0, 1.0))
    out = integrated_outcomes(p, t([0.8]), t([1.7]), c, t([0.0]), one(shock), None)
    X = _price(p, shock)
    assert float(out.ret_b) * X == pytest.approx(float(out.ret_l), abs=1e-14)
    loans, use = integrated_loan_book(p, t([0.8]), t([1.7]), c)
    assert float(loans) == pytest.approx(float(use) * 0.8 + float(use) * 1.7, abs=1e-15)


def _price(p, shock):
    prob = 1 - math.exp(-p.btc_crash_rate)
    m = math.log((1 + p.r_E + p.btc_bias) / (1 - prob * p.btc_crash_mean)) - p.s_BTC**2 / 2
    return math.exp(m + p.s_BTC * shock[3]) * (1 - shock[4])


@pytest.mark.parametrize("eps", [-1.0, 0.0, 0.7, 2.0])
def test_the_sleeve_keeps_its_btc_whatever_btc_does(eps):
    """All of the BTC vault in the sleeve, nothing lent, no fees, no funding:
    the vault's BTC is unchanged, so its dollar size moves one for one with
    the price."""
    p = replace(BASE, fee_mgmt=0.0, fee_perf=0.0, r=0.0, perp_premium=0.0, loan_demand=0.0,
                lend_loss=0.0, lend_vol=0.0, inflow_speed=0.0, btc_flow_yield=0.0, btc_crash_rate=0.0)
    shock = (0.0, 0.0, 0.0, eps, 0.0)
    c = controls(alloc=(0.0, 0.0, 0.0, 1.0))
    out = integrated_outcomes(p, t([0.8]), t([1.7]), c, t([0.0]), one(shock), None)
    assert float(out.yield_b) == pytest.approx(0.0, abs=1e-15)
    assert float(out.aum_b) == pytest.approx(1.7 * _price(p, shock), abs=1e-14)


def test_the_sleeve_pays_funding_on_all_of_itself_lent_or_not():
    p = replace(BASE, fee_mgmt=0.0, fee_perf=0.0, loan_demand=0.0, lend_loss=0.0, lend_vol=0.0,
                btc_crash_rate=0.0)
    shock = (0.0, 0.0, 0.0, 0.3, 0.0)
    out = integrated_outcomes(p, t([0.8]), t([1.7]), controls(alloc=(0.0, 0.0, 0.5, 0.5)), t([0.0]), one(shock), None)
    assert float(out.yield_b) == pytest.approx(-0.5 * (p.r + p.perp_premium) / _price(p, shock), abs=1e-15)


def test_the_firm_earns_its_interest_share_on_the_whole_loan_book():
    p = replace(BASE, lend_loss=0.0, lend_vol=0.0)
    c = controls(rate=0.08)
    out = integrated_outcomes(p, t([0.8]), t([1.7]), c, t([0.0]), one(SHOCKS[0]), None)
    loans, _ = integrated_loan_book(p, t([0.8]), t([1.7]), c)
    assert float(out.interest) == pytest.approx(p.interest_share * float(loans) * 0.08, abs=1e-15)


def test_the_sleeve_adds_to_the_supply_of_loans():
    p = BASE
    without = integrated_loan_book(p, t([0.8]), t([1.7]), controls(rate=0.02, alloc=(0.5, 0.5, 0.0, 0.0)))[0]
    with_sleeve = integrated_loan_book(p, t([0.8]), t([1.7]), controls(rate=0.02, alloc=(0.4, 0.4, 0.0, 0.2)))[0]
    assert float(without) == 0.8
    assert float(with_sleeve) == pytest.approx(0.8 + 0.2 * 1.7, abs=1e-15)


def test_capital_floor_covers_both_vaults_and_the_hedge_may_cover_the_btc_stake():
    p = replace(BASE, k_A=0.2, runway=0.0)
    A_L, A_B, zero = t([0.8]), t([1.7]), t([0.0])
    assert float(integrated_capital_floor(p, A_L, A_B, zero, zero)) == pytest.approx(0.2 * 2.5, abs=1e-15)
    p = replace(BASE, k_A=0.0)
    floor = float(integrated_capital_floor(p, A_L, A_B, t([0.01]), t([0.1])))
    assert floor == pytest.approx(0.01 + 0.1 + p.lend_co_invest * 0.8 + p.btc_co_invest * 1.7 + p.runway * p.opex, abs=1e-15)
    assert float(integrated_btc_held(p, A_B, t([0.1]))) == pytest.approx(0.1 + p.btc_co_invest * 1.7, abs=1e-15)


def test_the_integrated_defaults_make_the_btc_vault_twice_the_dollar_vault():
    assert INTEGRATED_DEFAULTS.btc_capacity == 2.0 * INTEGRATED_DEFAULTS.lend_capacity
    assert replace(INTEGRATED_DEFAULTS, btc_capacity=DEFAULTS.btc_capacity) == DEFAULTS


def test_an_incident_in_one_vault_costs_the_other_some_of_its_investors():
    """Both vaults carry the firm's name. The outflow from the vault not hit
    is the contagion share, scaled by the hit vault's share of the two."""
    p = replace(BASE, jump_rate=0.1)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P, n=1)
    quiet = replace(p, outflow_contagion=0.0)
    c = controls(alloc=(0.5, 0.3, 0.2, 0.0))          # no sleeve: the lending incident does not touch the BTC vault
    A_L, A_B = 0.8, 1.7
    out = integrated_outcomes(p, t([A_L]), t([A_B]), c, t([0.0]), one(SHOCKS[0]), jumps)
    ref = integrated_outcomes(quiet, t([A_L]), t([A_B]), c, t([0.0]), one(SHOCKS[0]), jumps)
    # outcomes: none, lending-system incident, BTC-vault incident
    assert float(ref.aum_b[0, 1] - out.aum_b[0, 1]) == pytest.approx(p.outflow_contagion * A_L / (A_L + A_B) * A_B, abs=1e-14)
    assert float(ref.aum_l[0, 2] - out.aum_l[0, 2]) == pytest.approx(p.outflow_contagion * A_B / (A_L + A_B) * A_L, abs=1e-14)
    assert torch.equal(out.aum_l[0, :2], ref.aum_l[0, :2]) and torch.equal(out.aum_b[0, [0, 2]], ref.aum_b[0, [0, 2]])
    assert torch.equal(out.wealth, ref.wealth)        # the firm feels it next period, through a smaller vault
