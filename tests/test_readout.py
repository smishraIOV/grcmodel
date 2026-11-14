"""The one-period read-out for a firm and a policy, with no solver
(docs/plan-tool.md step 3). Checked by hand where nothing is random, and for
the properties a user relies on: it adds up, it does not depend on the
currency unit, and finer scenarios do not move it.
"""

from dataclasses import fields, replace

import pytest

from quant.policy import Policy, starting_policy
from quant.profile import FirmState, ModelFirm, example_facts, illustrative, to_model
from quant.readout import ReadOut, read_out, report, scenario_set
from scripts.whatif import main

FACTS = example_facts()
MODEL = to_model(illustrative(FACTS))
POLICY = Policy(
    equity_kept=19e6, grc_spend=0.4e6, own_btc=2e6, hedged=3e6, borrower_rate=0.09, ltv=0.6,
    allocation=(0.2, 0.4, 0.1, 0.3),
)
MONEY = (
    "equity", "dollar_vault", "btc_vault", "equity_kept", "capital_needed", "payout", "raised", "stakes",
    "own_btc", "hedged", "stablecoins", "grc_spend", "loans", "expected_bad_debt", "interest_income",
    "btc_vault_fees", "operating_costs", "payroll", "compensation", "dollar_vault_next", "btc_vault_next",
    "expected_equity", "expected_shortfall", "cost_of_raising", "expected_cost_of_refilling",
)


@pytest.fixture(scope="module")
def base() -> ReadOut:
    return read_out(MODEL, POLICY)


def test_with_nothing_random_it_is_the_accounting_by_hand():
    """Annual, no volatility, crashes, incidents or operating losses: BTC
    rises by the required return for sure, every loan is repaid, and each
    figure follows from the profile on one line."""
    quiet = illustrative(replace(FACTS, decisions_per_year=1))
    quiet = replace(
        quiet,
        market=replace(quiet.market, btc_volatility=0.0, btc_crashes_per_year=0.0,
                       strategy_volatility=(0.0, 0.0, 0.0)),
        risk=replace(quiet.risk, incidents_per_year=0.0, lending_loss=0.0, lending_loss_volatility=0.0,
                     strategy_loss=(0.0, 0.0, 0.0)),
    )
    model = to_model(quiet)
    policy = replace(POLICY, borrower_rate=0.04, ltv=0.5, allocation=(0.5, 0.25, 0.25, 0.0), own_btc=0.0, hedged=0.0)
    r = read_out(model, policy)

    assert r.loans == pytest.approx(80e6) and r.share_lent == pytest.approx(1.0)   # demand exceeds the vault
    assert r.interest_income == pytest.approx(0.40 * 80e6 * 0.04, rel=1e-12)
    assert r.expected_bad_debt == 0.0 and r.incident_chance == pytest.approx(0.0, abs=1e-15)
    assert r.compensation == pytest.approx(0.0, abs=1e-6)      # dollars; rounding only
    y = 0.5 * 0.03 + 0.25 * 0.06                                   # the BTC vault's yield
    fee = 0.01 + 0.15 * y
    assert r.btc_vault_fees == pytest.approx(fee * 150e6 * 1.06, rel=1e-12)   # paid in BTC, worth 6% more
    assert r.dollar_yield == pytest.approx(0.6 * 0.04, rel=1e-12)             # providers keep 60% of the interest
    assert r.btc_yield == pytest.approx(y - fee, rel=1e-12)
    stakes = 0.05 * 80e6 + 0.05 * 150e6
    expected = (
        19e6 - 0.4e6 - 4e6                                         # GRC and operating costs paid
        + r.interest_income + r.btc_vault_fees
        + 0.05 * 80e6 * r.dollar_yield + 0.05 * 150e6 * ((1 + r.btc_yield) * 1.06 - 1)   # gains on its stakes
    )
    assert r.stakes == pytest.approx(stakes)
    assert r.expected_equity == pytest.approx(expected, rel=1e-12)
    # Costs exceed income here, so the firm ends with less than it kept, for sure,
    # but still above what it needs.
    assert expected < 19e6 and r.chance_of_loss == pytest.approx(1.0, abs=1e-12)
    assert r.chance_below_needed == 0.0 and r.expected_shortfall == 0.0 and r.expected_cost_of_refilling == 0.0
    assert r.cost_of_raising == 0.0


def test_it_adds_up(base):
    assert base.revenue == base.interest_income + base.btc_vault_fees
    assert base.operating_ratio == pytest.approx(4e6 / base.revenue)
    assert base.payroll == pytest.approx(0.6 * 4e6) and base.payroll_ratio == pytest.approx(0.6 * base.operating_ratio)
    assert base.payout == 1e6 and base.raised == 0.0
    assert base.stablecoins == pytest.approx(19e6 - 0.1e6 - 2e6 - base.stakes)
    assert base.capital_needed == pytest.approx(0.1e6 + 2e6 + base.stakes + 4e6)   # the runway binds
    assert 0.0 <= base.chance_below_needed <= base.chance_of_loss <= 1.0
    assert base.expected_shortfall > 0.0 and 0.0 < base.share_lent <= 1.0
    assert base.loans == pytest.approx(base.share_lent * (80e6 + 0.3 * 150e6))


def test_raising_now_and_refilling_later_are_costed(base):
    """Raising $3m now costs the plain issuance cost. The expected cost of
    refilling is at least the plain cost of the expected gap (the cost is
    convex), and more, since gaps come after falls and incidents."""
    p = MODEL.params
    raising = read_out(MODEL, replace(POLICY, equity_kept=23e6))
    assert raising.raised == 3e6
    assert raising.cost_of_raising == pytest.approx(p.lam * 3e6 + 0.5 * p.kappa * 3e6**2 / MODEL.scale, rel=1e-12)
    gap = base.expected_shortfall
    assert base.expected_cost_of_refilling > p.lam * gap + 0.5 * p.kappa * gap**2 / MODEL.scale > 0.0


def test_the_same_firm_in_cents_gives_the_same_read_out(base):
    cents = ModelFirm(params=MODEL.params, state=MODEL.state, scale=100.0 * MODEL.scale)
    policy = replace(POLICY, **{k: 100.0 * getattr(POLICY, k) for k in ("equity_kept", "grc_spend", "own_btc", "hedged")})
    other = read_out(cents, policy)
    for f in fields(ReadOut):
        a, b = getattr(base, f.name), getattr(other, f.name)
        factor = 100.0 if f.name in MONEY else 1.0
        assert b == pytest.approx(a if isinstance(a, tuple) else factor * a, rel=1e-10), f.name


def test_finer_scenarios_do_not_move_it(base):
    """More points for BTC's move and the crash, then more for the strategies'
    shocks and incidents. The policy lends 30% of the BTC vault as
    stablecoins: the BTC vault's yield with a sleeve is the figure that did
    not settle while BTC's price could reach zero."""
    p = MODEL.params.per_period()
    for finer in (dict(n_btc=401, n_crash=96), dict(n_btc=101, n_crash=24, n_strategy=6, n_mixing=16, n_jump=16)):
        _compare(read_out(MODEL, POLICY, scenario_set(p, **finer)), base, loose=finer["n_btc"] < 201)


def _compare(fine, base, loose):
    for name in ("interest_income", "btc_vault_fees", "expected_equity", "dollar_vault_next", "btc_vault_next"):
        assert getattr(fine, name) == pytest.approx(getattr(base, name), rel=2e-3), name
    for name in ("dollar_yield", "btc_yield", "incident_chance"):
        assert getattr(fine, name) == pytest.approx(getattr(base, name), abs=2e-4), name
    for name in ("expected_bad_debt", "expected_shortfall"):
        # the second set has fewer points for BTC than the default, to stay quick
        assert getattr(fine, name) == pytest.approx(getattr(base, name), rel=0.02 if loose else 0.005), name
    for name in ("chance_below_needed", "chance_of_loss"):
        assert getattr(fine, name) == pytest.approx(getattr(base, name), abs=0.005), name


def test_grc_spend_lowers_the_chance_of_an_incident(base):
    more = read_out(MODEL, replace(POLICY, grc_spend=2e6, equity_kept=19.5e6))
    assert more.incident_chance < base.incident_chance
    none = to_model(illustrative(replace(FACTS, grc_budget=0.0)))
    bare = read_out(none, replace(POLICY, grc_spend=0.0))
    assert bare.incident_chance == pytest.approx(1 - (1 - 2 * (1 - 2.718281828459045 ** (-0.1 / 4))) ** 4, rel=1e-12)


def test_the_sleeve_adds_loans_and_moves_value_between_the_two_groups_of_investors():
    """Design section 8, item 5: with demand to spare, lending part of the BTC
    vault raises the loan book; at a rate where demand is short, it dilutes
    the dollar providers' yield."""
    short = replace(POLICY, borrower_rate=0.02)        # more demand than stablecoins
    assert read_out(MODEL, short).loans > read_out(MODEL, replace(short, allocation=(0.3, 0.6, 0.1, 0.0))).loans
    idle = replace(POLICY, borrower_rate=0.15)         # more stablecoins than demand
    with_sleeve, without = read_out(MODEL, idle), read_out(MODEL, replace(idle, allocation=(0.3, 0.6, 0.1, 0.0)))
    assert with_sleeve.loans == pytest.approx(without.loans)
    assert with_sleeve.dollar_yield < without.dollar_yield


def test_an_unsound_policy_is_refused():
    with pytest.raises(ValueError, match="the runway needs"):
        read_out(MODEL, replace(POLICY, equity_kept=10e6))


def test_an_empty_vault_is_handled():
    state = FirmState(w=MODEL.state.w, S=MODEL.state.S, A_L=MODEL.state.A_L, A_B=0.0)
    lonely = ModelFirm(params=MODEL.params, state=state, scale=MODEL.scale)
    r = read_out(lonely, replace(starting_policy(lonely), hedged=0.0))
    assert r.btc_vault == 0.0 and r.btc_vault_fees == 0.0 and r.interest_income > 0.0


def test_the_report_and_the_script(base, capsys):
    text = report(base)
    for line in ("Capital", "The loan book", "stablecoin lending sleeve", "How the next quarter may end"):
        assert line in text
    main(["--periods", "1", "--rate", "0.08", "--sleeve", "0.2"])
    printed = capsys.readouterr().out
    assert "How the next year may end" in printed and "illustrative" in printed
    main(["--equity-kept", "1e7"])
    assert "This policy cannot be run" in capsys.readouterr().out
