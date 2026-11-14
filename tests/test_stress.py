"""Named stress scenarios (docs/plan-tool.md step 4): each one is the
accounting at a single outcome, checked against the rules written out by
hand (`by_hand` in tests/test_integrated.py) and against what the design says
a smooth fall, a crash and an incident do.
"""

import math
from dataclasses import replace

import pytest

from quant.core import next_grc
from quant.numerics import REFERENCE
from quant.policy import Policy, to_controls
from quant.profile import example_facts, illustrative, to_model
from quant.stress import Stress, problems, report, run, run_all, standard_stresses
from scripts.whatif import main
from tests.test_integrated import by_hand

P = REFERENCE
FACTS = example_facts()
POLICY = Policy(
    equity_kept=19e6, grc_spend=0.4e6, own_btc=2e6, hedged=3e6, borrower_rate=0.09, ltv=0.6,
    allocation=(0.2, 0.4, 0.1, 0.3),
)


def model_at(n):
    return to_model(illustrative(replace(FACTS, decisions_per_year=n)))


MODEL = model_at(4)


@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("stress", standard_stresses(), ids=lambda s: s.name)
def test_each_stress_is_the_accounting_by_hand(stress, n):
    model = model_at(n)
    p = model.params.per_period()
    c = to_controls(POLICY, model)
    S_next = float(next_grc(p, P.tensor([model.state.S]), c.g))
    # by_hand takes the shock to BTC's smooth move; find the one that gives this stress's move.
    prob = 1 - math.exp(-p.btc_crash_rate)
    m = math.log((1 + p.r_E + p.btc_bias) / (1 - prob * p.btc_crash_mean)) - p.s_BTC**2 / 2
    eps = (math.log(1 + stress.btc_move) - m) / p.s_BTC
    wealth, A_L, A_B = by_hand(
        p, model.state.A_L, model.state.A_B, c, (0.0, 0.0, 0.0, eps, stress.btc_crash),
        J_L=stress.lending_incident, J_B=stress.btc_vault_incident, S=S_next,
    )
    r = run(model, POLICY, stress)
    assert r.btc_price == pytest.approx((1 + stress.btc_move) * (1 - stress.btc_crash), rel=1e-15)
    assert r.equity == pytest.approx(model.dollars(wealth), rel=1e-10, abs=1e-3)
    assert r.dollar_vault == pytest.approx(model.dollars(A_L), rel=1e-10, abs=1e-3)
    assert r.btc_vault == pytest.approx(model.dollars(A_B), rel=1e-10, abs=1e-3)


def test_a_smooth_fall_is_caught_at_the_threshold_and_a_crash_is_not():
    """Design section 2.2. The same closing price, 50% down: a smooth fall
    leaves no bad debt and earns the vault the penalty; a crash sells the
    collateral after it, and what it cannot cover is bad debt."""
    smooth = run(MODEL, POLICY, Stress(name="smooth", btc_move=-0.5))
    crash = run(MODEL, POLICY, Stress(name="crash", btc_crash=0.5))
    calm = run(MODEL, POLICY, Stress(name="calm"))
    assert smooth.btc_price == crash.btc_price == 0.5
    assert smooth.bad_debt == 0.0 and smooth.repaid_to_investors == 0.0
    assert smooth.dollar_return > calm.dollar_return            # the liquidation penalty
    # In the crash every loan is liquidated at 0.5: collateral of 1 / 0.6 BTC a dollar lent,
    # sold 10% below market less what GRC narrows; the rest of what is owed is bad debt.
    p = MODEL.params.per_period()
    c = to_controls(POLICY, MODEL)
    stock = float(next_grc(p, P.tensor([MODEL.state.S]), c.g))
    discount = p.sale_discount * (p.rho_L + (1 - p.rho_L) * math.exp(-p.a_L * stock))
    owed = 1 + float(c.rate)
    demand = p.loan_demand * (1 - float(c.rate) / p.rate_max) * (1 + p.ltv_demand * (0.6 - p.ltv_ref))
    loans = min(demand, MODEL.state.A_L + 0.3 * MODEL.state.A_B)      # the dollar vault and the sleeve
    expected = loans * (owed - 0.5 * (1 - discount) / 0.6)
    assert crash.bad_debt == pytest.approx(MODEL.dollars(expected), rel=1e-9)
    assert crash.repaid_to_investors == pytest.approx(0.5 * 0.95 * crash.bad_debt, rel=1e-9)   # half, to outsiders
    assert crash.equity < smooth.equity < calm.equity


def test_an_incident_costs_investors_its_size_less_what_the_firm_repays():
    calm = run(MODEL, POLICY, Stress(name="calm"))
    hit = run(MODEL, POLICY, Stress(name="incident", lending_incident=0.10))
    assert hit.dollar_return - calm.dollar_return == pytest.approx(-0.10 * (1 - 0.5), abs=1e-12)
    assert hit.repaid_to_investors > 0 and hit.equity < calm.equity
    assert hit.dollar_vault < calm.dollar_vault            # the loss, and the withdrawals that follow
    other = run(MODEL, POLICY, Stress(name="incident", btc_vault_incident=0.10))
    assert other.dollar_return == pytest.approx(calm.dollar_return, abs=1e-15)   # a different vault
    exposure = 0.2 * 1.0 + 0.4 * 0.5 + 0.1 * 0.2            # the allocation times each strategy's exposure
    fee_saved = 0.15 * max(calm.btc_return + 0.0, 0.0)      # bounded: no performance fee on a loss
    assert other.btc_return - calm.btc_return == pytest.approx(-0.10 * exposure * (1 - 0.5), abs=fee_saved + 1e-3)


def test_an_incident_can_be_imposed_on_a_profile_that_expects_none():
    quiet = illustrative(FACTS)
    quiet = to_model(replace(quiet, risk=replace(quiet.risk, incidents_per_year=0.0)))
    calm = run(quiet, POLICY, Stress(name="calm"))
    hit = run(quiet, POLICY, Stress(name="incident", lending_incident=0.10))
    assert hit.dollar_return - calm.dollar_return == pytest.approx(-0.05, abs=1e-12)


def test_the_shortfall_and_the_cost_of_raising_it():
    r = run(MODEL, POLICY, Stress(name="crash", btc_crash=0.5))
    assert r.shortfall == pytest.approx(r.capital_needed - r.equity) and r.shortfall > 0
    p = MODEL.params
    plain = p.lam * r.shortfall + 0.5 * p.kappa * r.shortfall**2 / MODEL.scale
    assert r.cost_of_raising == pytest.approx(3 * plain, rel=1e-12)       # three times as dear after BTC halves
    both = run(MODEL, POLICY, Stress(name="both", btc_crash=0.5, lending_incident=0.1))
    plain = p.lam * both.shortfall + 0.5 * p.kappa * both.shortfall**2 / MODEL.scale
    assert both.cost_of_raising == pytest.approx(5 * plain, rel=1e-12)    # and after an incident as well
    calm = run(MODEL, POLICY, Stress(name="calm"))
    assert calm.shortfall == 0.0 and calm.cost_of_raising == 0.0
    assert calm.equity_change == pytest.approx(calm.equity - 19e6)


def test_a_hedge_takes_btc_s_move_out_of_the_firm_s_own_btc():
    """Fully hedged, the firm's own BTC and its stake in the BTC vault gain
    or lose nothing from a smooth move; unhedged they do."""
    held = 2e6 + 0.05 * 150e6
    hedged, bare = replace(POLICY, hedged=held), replace(POLICY, hedged=0.0)
    down = Stress(name="down", btc_move=-0.3)
    gap = run(MODEL, hedged, down).equity - run(MODEL, bare, down).equity
    p = MODEL.params.per_period()
    assert gap == pytest.approx(held * (0.3 + p.r + p.perp_premium), rel=1e-9)   # the fall, and funding received


@pytest.mark.parametrize("stress, message", [
    (Stress(name="x", lending_incident=0.1, btc_vault_incident=0.1), "one incident a period"),
    (Stress(name="x", btc_move=-1.0), "less than 100%"),
    (Stress(name="x", btc_crash=1.0), "below all of it"),
    (Stress(name="x", lending_incident=-0.1), "between 0 and 1"),
])
def test_a_stress_the_model_cannot_represent_is_refused(stress, message):
    assert problems(stress, MODEL)
    with pytest.raises(ValueError, match=message):
        run(MODEL, POLICY, stress)


def test_an_unsound_policy_is_refused():
    with pytest.raises(ValueError, match="the runway needs"):
        run(MODEL, replace(POLICY, equity_kept=10e6), Stress(name="calm"))


def test_the_report_and_the_script(capsys):
    results = run_all(MODEL, POLICY)
    assert len(results) == len(standard_stresses())
    text = report(results)
    assert "BTC crashes 50%" in text and "capital needed" in text
    main(["--ltv", "0.7", "--stress"])
    assert "BTC crashes 70%" in capsys.readouterr().out
