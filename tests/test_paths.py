"""The firm over several years on random paths (docs/plan-tool.md step 5).

The draws are checked against the distributions they are meant to come from,
one simulated period against the read-out's weighted scenarios, and the rule's
mechanics against a firm with nothing random, where every path is the same
and can be followed by hand.
"""

import math
from dataclasses import replace

import pytest
import torch

from quant.core import issuance_cost
from quant.firm import btc_prices, crash_probability
from quant.integrated import IntegratedControls, integrated_loan_book
from quant.numerics import REFERENCE
from quant.paths import Rule, capital_needed, draw, problems, report, rule_from, simulate
from quant.policy import Policy, starting_policy
from quant.profile import example_facts, illustrative, to_model
from quant.quadrature import pareto_jump_rule
from quant.readout import read_out
from scripts.whatif import main

P = REFERENCE
FACTS = example_facts()
MODEL = to_model(illustrative(FACTS))
POLICY = Policy(
    equity_kept=19e6, grc_spend=0.4e6, own_btc=2e6, hedged=3e6, borrower_rate=0.09, ltv=0.6,
    allocation=(0.2, 0.4, 0.1, 0.3),
)


def within(sample: torch.Tensor, target: float, errors: float = 4.5) -> bool:
    """Whether a sample's mean is within `errors` standard errors of a target."""
    return abs(float(sample.mean()) - target) <= errors * float(sample.std()) / math.sqrt(sample.numel()) + 1e-12


def test_the_draws_come_from_the_model_s_distributions():
    p = MODEL.params.per_period()
    d = draw(p, periods=4, paths=250_000, seed=7)
    for x in (d.xi_l, d.xi_1, d.xi_2, d.eps):
        assert x.shape == (4, 250_000) and x.dtype == torch.float64
        assert within(x, 0.0)
    assert within(d.eps**2, 1.0) and within(d.xi_1 * d.xi_2, p.strategy_rho) and within(d.xi_l * d.xi_1, 0.0)
    # the vault's shocks are fat-tailed and share their tail; BTC's smooth move is normal and apart
    assert float((d.xi_l**2).mean()) == pytest.approx(1.0, abs=0.05)
    assert float((d.xi_l**2 * d.xi_1**2).mean()) > 1.5 and float((d.eps**4).mean()) == pytest.approx(3.0, abs=0.1)
    assert within(d.xi_l**2 * d.eps**2, 1.0)
    # crashes: how often, and the mean share of the price they take
    crashed = (d.crash > 0).to(torch.float64)
    assert within(crashed, crash_probability(p)) and float(d.crash.max()) < 1.0
    assert within(d.crash[d.crash > 0], p.btc_crash_mean)
    # so BTC is expected to earn what the market requires
    price, _ = btc_prices(p, d.eps, d.crash, 0.0)
    assert within(price, 1 + p.r_E) and float(price.min()) > 0.0
    # incident sizes, against the rule the read-out uses
    assert within(d.severity, pareto_jump_rule(p.jump_mean, p.jump_tail, P, n=200, tail_prob=1e-12).mean)
    assert within(d.incident, 0.5)


def test_the_same_seed_gives_the_same_paths_and_another_seed_agrees_within_error():
    rule = rule_from(POLICY, MODEL)
    a, b = simulate(MODEL, rule, years=2, paths=4000, seed=3), simulate(MODEL, rule, years=2, paths=4000, seed=3)
    assert torch.equal(a.equity, b.equity) and torch.equal(a.payout, b.payout)
    c = simulate(MODEL, rule, years=2, paths=4000, seed=4)
    assert not torch.equal(a.equity, c.equity)
    (va, ea), (vc, ec) = a.value(), c.value()
    assert abs(va - vc) < 4.5 * math.hypot(ea, ec)


def test_one_simulated_period_agrees_with_the_read_out():
    """The same accounting, averaged two ways: over weighted scenarios in the
    read-out, over random draws here."""
    rule = rule_from(POLICY, MODEL)
    assert rule.equity_buffer == pytest.approx(19e6 - capital_needed(rule, MODEL))
    r = simulate(MODEL, rule, years=0.25, paths=400_000, seed=11)
    ro = read_out(MODEL, POLICY)
    assert r.periods == 1 and ro.payout == 1e6 and float(r.payout[0, 0]) == pytest.approx(1e6)
    end = r.equity[1]
    assert within(end, ro.expected_equity)
    assert within(r.dollar_vault[1], ro.dollar_vault_next) and within(r.btc_vault[1], ro.btc_vault_next)
    assert within(4 * r.bad_debt[0], ro.expected_bad_debt)
    assert within((end < ro.capital_needed).to(torch.float64), ro.chance_below_needed)
    assert within((end < 19e6).to(torch.float64), ro.chance_of_loss)
    # the read-out compounds the period's mean return to a year
    assert within(r.dollar_return[0], (1 + ro.dollar_yield) ** 0.25 - 1)
    assert within(r.btc_return[0], (1 + ro.btc_yield) ** 0.25 - 1)
    short = torch.relu(ro.capital_needed - end)
    assert within(short, ro.expected_shortfall)


def quiet_model(n=1, **facts):
    """Nothing random: no volatility, crashes, incidents or operating losses."""
    profile = illustrative(replace(FACTS, decisions_per_year=n, **facts))
    profile = replace(
        profile,
        market=replace(profile.market, btc_volatility=0.0, btc_crashes_per_year=0.0, strategy_volatility=(0.0, 0.0, 0.0)),
        risk=replace(profile.risk, incidents_per_year=0.0, lending_loss=0.0, lending_loss_volatility=0.0,
                     strategy_loss=(0.0, 0.0, 0.0)),
    )
    return to_model(profile)


def test_with_nothing_random_every_path_is_the_same_and_follows_the_rule():
    """Annual. The firm pays out down to its target, earns the read-out's
    expected equity for sure, and next year's payout or raise follows from
    where that leaves it against what it then needs."""
    model = quiet_model(operating_costs=2e6)
    policy = replace(starting_policy(model), borrower_rate=0.04)
    rule = replace(rule_from(policy, model), equity_buffer=1e6)
    r = simulate(model, rule, years=3, paths=5, seed=0)
    for series in (r.equity, r.dollar_vault, r.btc_vault, r.payout, r.raised):
        assert float((series - series[:, :1]).abs().max()) == 0.0
    needed = capital_needed(rule, model)
    target = needed + 1e6
    assert float(r.payout[0, 0]) == pytest.approx(20e6 - target) and float(r.raised[0, 0]) == 0.0
    first = read_out(model, replace(policy, equity_kept=target))
    assert float(r.equity[1, 0]) == pytest.approx(first.expected_equity, rel=1e-9)
    assert float(r.dollar_vault[1, 0]) == pytest.approx(first.dollar_vault_next, rel=1e-9)
    assert float(r.btc_price[1, 0]) == pytest.approx(1.06) and float(r.btc_price[3, 0]) == pytest.approx(1.06**3)
    # the second year, from the state the first left
    state = replace(model.state, w=model.units(float(r.equity[1, 0])), A_L=model.units(float(r.dollar_vault[1, 0])),
                    A_B=model.units(float(r.btc_vault[1, 0])))
    later = replace(model, state=state)
    needed_later = capital_needed(rule, later)
    assert float(r.payout[1, 0]) == pytest.approx(max(float(r.equity[1, 0]) - needed_later - 1e6, 0.0), abs=1e-3)
    assert float(r.raised[1, 0]) == pytest.approx(max(needed_later - float(r.equity[1, 0]), 0.0), abs=1e-3)
    assert not bool(r.failed.any()) and float(r.chance_failed()[-1]) == 0.0
    # what shareholders take: the flows discounted, and equity at the end at book
    beta = 1 / 1.06
    flows = sum(beta**t * float(r.payout[t, 0] - r.raised[t, 0] - r.raise_cost[t, 0]) for t in range(3))
    assert r.value()[0] == pytest.approx(flows + beta**3 * float(r.equity[3, 0]), rel=1e-12)


def test_a_firm_short_of_capital_raises_it_at_the_plain_cost_today():
    model = quiet_model(equity=10e6)
    rule = rule_from(starting_policy(model), model, largest_raise=50e6)
    needed = capital_needed(rule, model)
    r = simulate(model, rule, years=1, paths=3, seed=0)
    gap = needed - 10e6
    assert gap > 0 and float(r.raised[0, 0]) == pytest.approx(gap)
    plain = float(issuance_cost(model.params, P.tensor(model.units(gap)))) * model.scale
    assert float(r.raise_cost[0, 0]) == pytest.approx(plain, rel=1e-12)
    assert float(r.chance_raised()[0]) == 1.0


def test_a_raise_larger_than_shareholders_will_make_is_the_end():
    """Needing more than the largest raise, the firm fails: from then it pays
    nothing, raises nothing and counts for nothing; failure does not undo."""
    model = quiet_model(equity=10e6)
    rule = rule_from(starting_policy(model), model, largest_raise=1e6)
    r = simulate(model, rule, years=2, paths=3, seed=0)
    assert bool(r.failed[1:].all()) and not bool(r.failed[0].any())      # decided in the first period
    assert float(r.payout.abs().max()) == 0.0 and float(r.raised.abs().max()) == 0.0
    assert float(r.equity[1:].abs().max()) == 0.0 and r.value()[0] == 0.0

    risky = simulate(MODEL, replace(rule_from(POLICY, MODEL), ltv=0.75, largest_raise=2e6), years=5, paths=20_000, seed=5)
    failed = risky.chance_failed()
    assert 0.0 < float(failed[-1]) < 1.0 and bool((failed[1:] >= failed[:-1]).all())
    gone = risky.failed[-1]
    assert float(risky.equity[-1][gone].abs().max()) == 0.0
    # raising after a bad period is dearer than the plain cost, never cheaper
    p = MODEL.params
    raised = risky.raised[1:]
    plain = p.lam * raised + 0.5 * p.kappa * raised**2 / MODEL.scale
    assert bool((risky.raise_cost[1:] >= plain - 1e-6).all()) and bool((risky.raise_cost[1:] > 1.5 * plain)[raised > 0].any())


def test_more_buffer_means_fewer_raises():
    lean = simulate(MODEL, replace(rule_from(POLICY, MODEL), equity_buffer=0.0), years=3, paths=20_000, seed=2)
    thick = simulate(MODEL, replace(rule_from(POLICY, MODEL), equity_buffer=10e6), years=3, paths=20_000, seed=2)
    assert float(thick.chance_raised()[-1]) < float(lean.chance_raised()[-1])
    assert float(thick.raise_cost.mean()) < float(lean.raise_cost.mean())


def test_loan_demand_can_follow_btc_s_price():
    """The accounting takes a scale on demand; a path passes BTC's price to
    the power `demand_btc`."""
    p = MODEL.params.per_period()
    c = IntegratedControls(K=P.tensor([0.2]), g=P.tensor([0.0]), q=P.tensor([0.0]), H=P.tensor([0.0]),
                           rate=P.tensor([0.03]), ltv=P.tensor([0.5]), alloc=P.tensor([[0.3, 0.3, 0.4, 0.0]]))
    A_L, A_B = P.tensor([1.0]), P.tensor([1.5])
    full, _ = integrated_loan_book(p, A_L, A_B, c)
    half, _ = integrated_loan_book(p, A_L, A_B, c, demand_scale=P.tensor([0.25]) ** p.demand_btc)
    assert p.demand_btc == 0.5 and float(full) > float(half)
    lent = float(integrated_loan_book(p, A_L * 10, A_B, c)[0])            # demand, with stablecoins to spare
    assert float(integrated_loan_book(p, A_L * 10, A_B, c, demand_scale=0.5)[0]) == pytest.approx(0.5 * lent)


def test_a_rule_from_a_policy_starts_as_the_policy_does():
    rule = rule_from(POLICY, MODEL)
    assert rule.hedge_ratio == pytest.approx(3e6 / (2e6 + 0.05 * 150e6))
    assert rule.largest_raise == pytest.approx(capital_needed(rule, MODEL))
    assert (rule.grc_spend, rule.own_btc, rule.borrower_rate, rule.ltv) == (0.4e6, 2e6, 0.09, 0.6)
    assert rule.allocation == POLICY.allocation


@pytest.mark.parametrize("change, message", [
    (dict(hedge_ratio=1.2), "hedge ratio must be between 0 and 1"),
    (dict(equity_buffer=-1.0), "equity buffer cannot be negative"),
    (dict(ltv=0.9), "outside the range the firm may offer"),
    (dict(allocation=(0.5, 0.5, 0.5, 0.0)), "must sum to 100%"),
])
def test_an_unsound_rule_is_refused(change, message):
    rule = replace(rule_from(POLICY, MODEL), **change)
    assert problems(rule, MODEL)
    with pytest.raises(ValueError, match=message):
        simulate(MODEL, rule, years=1, paths=10)
    assert isinstance(rule, Rule)


def test_the_report_and_the_script(capsys):
    r = simulate(MODEL, rule_from(POLICY, MODEL), years=2.5, paths=2000, seed=1)
    text = report(r)
    assert "2,000 random paths over 2.5 years" in text and "What shareholders take" in text
    assert r.quantiles(r.equity).shape == (11, 5) and r.quantiles(r.btc_vault, alive_only=True).shape == (11, 5)
    main(["--years", "2", "--paths", "500"])
    assert "500 random paths over 2 years" in capsys.readouterr().out
