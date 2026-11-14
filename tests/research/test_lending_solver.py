"""The lending vault on an infinite horizon (design.md sections 2.2, 3 and 5).

Annual and coarse, so the suite stays fast; realistic runs are quarterly.
"""

from dataclasses import replace

import pytest

from quant.research.lending_solver import LendingSpec, market_value, solve_lending
from quant.numerics import REFERENCE
from quant.params import DEFAULTS

pytestmark = pytest.mark.slow   # run with `uv run pytest -m slow`

P = REFERENCE
BASE = replace(DEFAULTS, periods_per_year=1)
SPEC = LendingSpec(n_w=17, n_s=3, n_A=5, n_ops=1, n_btc=5, n_mixing=3, n_crash=3, n_jump=3)


@pytest.fixture(scope="module")
def base():
    return solve_lending(BASE, P, SPEC)


@pytest.fixture(scope="module")
def biased():
    return solve_lending(replace(BASE, btc_bias=0.04), P, SPEC)


def test_converges_and_reaches_both_edges(base):
    # The solver settles to about 2e-3 and no further (docs/plan-integrated.md section 7).
    assert base.residual < 1e-3
    assert base.lower_edge_is_walk_away
    assert base.upper_edge_pays_out


def test_value_is_nonnegative_and_increasing_in_equity(base):
    V = base.V
    assert bool((V >= 0).all())
    assert bool((V.diff(dim=0) >= -1e-9).all())


def test_decisions_respect_their_bounds(base):
    c, p = base.controls(), base.params
    assert bool((c.H <= c.q + 1e-15).all())
    assert bool(((c.rate > 0) & (c.rate < p.rate_max)).all())
    assert bool(((c.ltv >= p.ltv_min) & (c.ltv <= p.ltv_max)).all())


@pytest.mark.xfail(
    reason="open: the check's premise is degenerate. With no cost of holding BTC and a fair hedge, "
           "capital held as BTC is free, the firm keeps a large buffer, its value is linear in equity "
           "there, and it is indifferent to hedging (docs/plan-integrated.md section 7)",
    strict=False,
)
def test_with_no_bias_and_fair_hedging_the_firm_holds_no_net_btc():
    """Prop. 2, in the form it takes here. With no bias, BTC held unhedged
    earns only what shareholders require and adds risk, so the firm keeps
    essentially no net BTC: under half a percent of the vault, anywhere. (The
    little BTC it holds as a store of value is hedged 90-96%: at high equity
    the firm is close to risk-neutral, so the last few percent of the hedge
    are worth almost nothing to it.) The hedge may run to 1.2 times the BTC
    held in this test only, so that a full hedge is an interior point. Fair
    hedging means a short perp is paid what the market expects of BTC, r_E:
    a funding premium of r_E - r, so that hedging neither costs nor earns on
    average. (With less, hedging costs the difference, and since BTC's
    expected price became exact a firm with equity to spare is indifferent to
    holding BTC unhedged, and does.)

    Checked at ordinary vault sizes. At the smallest vault on the grid the
    business is worth little, a bad period can bring the firm near walk-away,
    and it keeps about 2% of the vault in unhedged BTC: limited liability
    rewards a small gamble there (design.md section 3.4). A result, not an
    error, and outside what Prop. 2 covers -- the paper has no walk-away."""
    p = replace(BASE, btc_holding_cost=0.0, c_H=0.0, perp_premium=BASE.r_E - BASE.r)
    sol = solve_lending(p, P, replace(SPEC, hedge_max=1.2))
    c = sol.controls()
    _, _, A = sol.states()
    ordinary = A >= 0.4
    assert float((c.q - c.H)[ordinary].abs().max()) < 0.005


def test_a_bias_above_the_security_cost_keeps_the_firm_long_but_bounded(biased):
    """The firm holds net BTC on its view, and less than its cap: risk, not the
    bound, limits the position."""
    c, p = biased.controls(), biased.params
    w, _, _ = biased.states()
    safe = (w >= 0.0) & (w <= 0.3)
    net = (c.q - c.H)[safe]
    assert bool((net > 0).all())
    assert float(c.q[safe].max()) < 0.9 * p.q_max


def test_the_bias_cannot_beat_the_unbiased_firm_at_market_prices(base, biased):
    """Valued at the market's expectation, the policy chosen with a bias is
    worth no more than the policy chosen without one."""
    V_market = market_value(biased)
    assert bool((V_market <= base.V + 1e-6).all())


def test_answer_does_not_depend_on_the_step_budget(base):
    more = solve_lending(BASE, P, SPEC, improve_steps=80, polish_steps=600)
    assert float((more.V - base.V).abs().max()) < 1e-4    # values are of order 0.1-1


@pytest.mark.xfail(
    reason="open: under BTC's price in its logarithm the lending vault alone has operating costs at "
           "60.4% of revenue, just outside 50-60%; costs are to be recalibrated, not nudged "
           "(docs/plan-integrated.md section 7)",
    strict=False,
)
def test_operating_costs_sit_within_the_industry_bounds(base):
    """At a full vault and the terms the firm chooses: payroll at least 30% of
    revenue, all operating costs 50-60% (design.md section 2.7)."""
    from quant.research.lending_results import cost_ratios

    ratios = cost_ratios(base, at_w=0.1, at_A=1.0)
    assert ratios.revenue > 0
    assert ratios.within_bounds, ratios
