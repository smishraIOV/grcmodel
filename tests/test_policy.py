"""The firm's policy as an input (design.md section 3.2; docs/plan-tool.md
step 2): what it means for the books, what is refused and why, and the
mapping onto the accounting's decision and back.
"""

from dataclasses import replace

import pytest
import torch

from quant.core import next_grc
from quant.integrated import (
    integrated_btc_held,
    integrated_capital_floor,
    integrated_outcomes,
    integrated_rule,
)
from quant.numerics import REFERENCE
from quant.policy import Policy, check, position, problems, starting_policy, to_controls, to_policy
from quant.profile import example_facts, illustrative, to_model

P = REFERENCE
FACTS = example_facts()          # $100m dollar vault at $80m, $200m BTC vault at $150m, $20m of equity
MODEL = to_model(illustrative(FACTS))
# Stakes are 5% of 80m + 5% of 150m = 11.5m; the runway is 12 months of 4m.
SOUND = Policy(
    equity_kept=18e6, grc_spend=0.4e6, own_btc=2e6, hedged=3e6, borrower_rate=0.09, ltv=0.6,
    allocation=(0.2, 0.5, 0.1, 0.2),
)


def t(x):
    return P.tensor([x])


def test_a_sound_policy_has_no_problems():
    assert problems(SOUND, MODEL) == []
    check(SOUND, MODEL)


def test_what_a_policy_means_for_the_books():
    pos = position(SOUND, MODEL)
    assert (pos.payout, pos.raised) == (2e6, 0.0)
    assert pos.stakes == pytest.approx(11.5e6)
    assert pos.stablecoins == pytest.approx(18e6 - 0.4e6 / 4 - 2e6 - 11.5e6)
    assert pos.runway_needed == pytest.approx(4e6)
    assert pos.own_funds_needed == pytest.approx(0.02 * 230e6)
    assert pos.btc_held == pytest.approx(2e6 + 0.05 * 150e6)
    assert position(replace(SOUND, equity_kept=23e6), MODEL).raised == 3e6


def test_the_policy_becomes_one_period_s_decision_in_model_units():
    c = to_controls(SOUND, MODEL)
    assert float(c.K) == pytest.approx(0.18) and float(c.q) == pytest.approx(0.02)
    assert float(c.H) == pytest.approx(0.03) and float(c.ltv) == 0.6
    assert float(c.g) == pytest.approx(0.004 / 4)                     # a quarter of the year's spend
    assert (1 + float(c.rate)) ** 4 - 1 == pytest.approx(0.09)        # the quarter's rate compounds to 9%
    assert c.alloc.shape == (1, 4) and c.K.dtype == torch.float64


def test_a_policy_round_trips():
    back = to_policy(to_controls(SOUND, MODEL), MODEL)
    for name in ("equity_kept", "grc_spend", "own_btc", "hedged", "borrower_rate", "ltv"):
        assert getattr(back, name) == pytest.approx(getattr(SOUND, name), rel=1e-13), name
    assert back.allocation == pytest.approx(SOUND.allocation, rel=1e-13)


def test_the_constraints_are_the_accounting_s_own():
    """A policy is sound exactly when its decision meets the accounting's
    capital floor and hedge bound, on either side of each."""
    p = MODEL.params.per_period()
    A_L, A_B = t(MODEL.state.A_L), t(MODEL.state.A_B)
    for policy in (SOUND, replace(SOUND, equity_kept=17.6e6), replace(SOUND, equity_kept=17.59e6),
                   replace(SOUND, hedged=9.5e6), replace(SOUND, hedged=9.51e6),
                   replace(SOUND, own_btc=0.0, hedged=0.0, grc_spend=0.0, equity_kept=15.5e6),
                   replace(SOUND, own_btc=0.0, hedged=0.0, grc_spend=0.0, equity_kept=15.49e6)):
        g, q = t(MODEL.units(policy.grc_spend) / 4), t(MODEL.units(policy.own_btc))
        floor = float(integrated_capital_floor(p, A_L, A_B, g, q))
        held = float(integrated_btc_held(p, A_B, q))
        meets = MODEL.units(policy.equity_kept) >= floor - 1e-12 and MODEL.units(policy.hedged) <= held + 1e-12
        assert (problems(policy, MODEL) == []) == meets, policy


def test_the_decision_runs_through_the_accounting():
    p = MODEL.params.per_period()
    c = to_controls(SOUND, MODEL)
    rule = integrated_rule(p, P, n_ops=1, n_strategy=1, n_btc=5, n_mixing=3, n_crash=3)
    out = integrated_outcomes(p, t(MODEL.state.A_L), t(MODEL.state.A_B), c, next_grc(p, t(MODEL.state.S), c.g), rule, None)
    assert out.wealth.shape == (1, rule.size)
    assert float(out.weights.sum()) == pytest.approx(1.0)
    assert bool(torch.isfinite(out.wealth).all())


@pytest.mark.parametrize("change, message", [
    (dict(equity_kept=4e6), r"below the own funds required, \$4,600,000 \(2\.0% of the two vaults\)"),
    (dict(equity_kept=17e6), r"leaves \$3,400,000 in stablecoins .* the runway needs \$4,000,000 \(12 months"),
    (dict(own_btc=5e6), r"leaves \$1,400,000 in stablecoins"),
    (dict(hedged=10e6), r"hedge of \$10,000,000 is more than the BTC the firm owns, \$9,500,000"),
    (dict(ltv=0.8), r"loan-to-value of 80% is outside the range the firm may offer, 30% to 75%"),
    (dict(allocation=(0.5, 0.5, 0.2, 0.0)), r"must sum to 100% \(it sums to 120\.0%\)"),
    (dict(allocation=(0.7, 0.5, -0.2, 0.0)), "cannot have a negative share"),
    (dict(allocation=(0.5, 0.5)), "needs 4 shares"),
    (dict(grc_spend=-1.0), "GRC spend cannot be negative"),
    (dict(borrower_rate=-0.01), "the borrower rate cannot be negative"),
])
def test_an_unsound_policy_is_refused_with_the_reason(change, message):
    policy = replace(SOUND, **change)
    assert problems(policy, MODEL)
    with pytest.raises(ValueError, match=message):
        to_controls(policy, MODEL)


def test_every_problem_is_reported_at_once():
    found = problems(replace(SOUND, equity_kept=4e6, hedged=10e6, ltv=0.2), MODEL)
    assert len(found) == 4      # own funds, runway, hedge, loan-to-value
    with pytest.raises(ValueError, match="this policy cannot be run: .*; .*; .*; "):
        check(replace(SOUND, equity_kept=4e6, hedged=10e6, ltv=0.2), MODEL)


@pytest.mark.parametrize("facts", [
    FACTS,
    replace(FACTS, equity=3e6),                               # too little equity: it must raise
    replace(FACTS, decisions_per_year=1, grc_budget=0.0),
    replace(FACTS, own_funds=0.2, runway_months=0.0),         # own funds, not the runway, sets the floor
])
def test_the_starting_policy_is_sound_and_keeps_the_grc_stock(facts):
    model = to_model(illustrative(facts))
    policy = starting_policy(model)
    assert problems(policy, model) == []
    assert policy.grc_spend == pytest.approx(facts.grc_budget, rel=1e-12, abs=1e-6)
    pos = position(policy, model)
    assert policy.equity_kept >= facts.equity
    if policy.equity_kept > facts.equity:       # at the least the constraints allow
        tight = min(policy.equity_kept - pos.own_funds_needed, pos.stablecoins - pos.runway_needed)
        assert tight == pytest.approx(0.0, abs=1e-6)
