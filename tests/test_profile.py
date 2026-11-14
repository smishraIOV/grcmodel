"""A firm's profile in real units (design.md section 4.1; docs/plan-tool.md
step 1): the mapping onto the model, the way back, and the statements a user
answers in place of raw parameters.
"""

import math
from dataclasses import asdict, fields, replace

import pytest

from quant.core import issuance_cost, mitigation, next_grc
from quant.lending import loan_demand
from quant.numerics import REFERENCE
from quant.params import DEFAULTS, INTEGRATED_DEFAULTS, Params
from quant.profile import (
    LEFT_AT_DEFAULT,
    FirmProfile,
    GrcEffect,
    LoanPoint,
    RaiseCost,
    _changes,
    example_facts,
    illustrative,
    profile_fields,
    steady_stock,
    to_model,
    to_profile,
    validate,
)

P = REFERENCE
FACTS = example_facts()
PROFILE = illustrative(FACTS)


def t(x):
    return P.tensor(x)


def flat(obj, prefix=""):
    """A nested dataclass as {dotted name: number}."""
    out = {}
    for key, value in (asdict(obj) if not isinstance(obj, dict) else obj).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flat(value, name + "."))
        elif isinstance(value, (tuple, list)):
            out.update({f"{name}[{i}]": v for i, v in enumerate(value)})
        else:
            out[name] = value
    return out


def assert_close(a, b, rel=1e-12):
    a, b = flat(a), flat(b)
    assert a.keys() == b.keys()
    for name in a:
        assert a[name] == pytest.approx(b[name], rel=rel, abs=1e-15), name


def test_the_scale_is_the_dollar_vault_s_capacity():
    model = to_model(PROFILE)
    p, s = model.params, model.state
    assert model.scale == 100e6
    assert (p.lend_capacity, p.btc_capacity) == (1.0, 2.0)
    assert (s.A_L, s.A_B, s.w) == (0.8, 1.5, 0.2)
    assert p.opex == 0.04 and p.runway == 1.0 and p.periods_per_year == 4
    assert model.dollars(s.w) == 20e6 and model.units(20e6) == s.w


def test_a_profile_round_trips():
    assert_close(to_profile(to_model(PROFILE), like=PROFILE), PROFILE)


def test_a_changed_profile_round_trips():
    """Not only the defaults: every kind of statement moved off them."""
    facts = replace(
        FACTS, grc_budget=1.3e6, equity=31e6, decisions_per_year=1, runway_months=9.0,
        small_raise=RaiseCost(2e6, 0.04), large_raise=RaiseCost(30e6, 0.11),
    )
    base = illustrative(facts)
    profile = FirmProfile(
        facts=facts,
        market=replace(base.market, loan_demand=LoanPoint(loans=140e6, rate=0.07), rate_no_demand=0.25,
                       demand_per_10_ltv_points=0.2, strategy_yield=(0.02, 0.05, 0.001)),
        risk=replace(base.risk, grc_wear_off=0.4, grc_on_incidents=GrcEffect(2e6, 0.5, 0.8),
                     grc_on_sale_discount=GrcEffect(0.3e6, 0.1, 0.6)),
    )
    assert_close(to_profile(to_model(profile), like=profile), profile)


def test_the_model_round_trips():
    model = to_model(PROFILE)
    again = to_model(to_profile(model, like=PROFILE))
    assert_close(asdict(again.params), asdict(model.params))
    assert_close(again.state, model.state)
    assert again.scale == model.scale


@pytest.mark.parametrize("n", [1, 4])
def test_the_illustrative_assumptions_are_the_model_s_defaults(n):
    """One source: `quant.params`. With the default facts too, the profile is
    the integrated defaults."""
    model = to_model(illustrative(replace(FACTS, decisions_per_year=n, operating_costs=2e6)))
    assert_close(asdict(model.params), asdict(replace(INTEGRATED_DEFAULTS, periods_per_year=n)), rel=1e-10)


def test_every_parameter_is_set_by_the_profile_or_listed_as_left_out():
    """A parameter added to the model must be given a place in the profile,
    or be named as one it leaves alone."""
    names = {f.name for f in fields(Params)}
    assert set(_changes(PROFILE)) | LEFT_AT_DEFAULT == names
    assert not set(_changes(PROFILE)) & LEFT_AT_DEFAULT


def test_parameters_left_out_come_from_the_base():
    base = replace(DEFAULTS, nu=7.0, q_max=0.5)
    p = to_model(PROFILE, base=base).params
    assert (p.nu, p.q_max) == (7.0, 0.5)


def test_the_model_does_not_depend_on_the_currency_unit():
    """The same firm in cents: every dollar amount times 100. The model's
    parameters and state are unchanged; only the scale moves."""
    def times(obj, k):
        changed = {}
        for f in fields(obj):
            v = getattr(obj, f.name)
            if isinstance(v, RaiseCost):
                changed[f.name] = RaiseCost(v.amount * k, v.cost)
            elif isinstance(v, LoanPoint):
                changed[f.name] = LoanPoint(v.loans * k, v.rate)
            elif isinstance(v, GrcEffect):
                changed[f.name] = GrcEffect(v.spend * k, v.reduction, v.limit)
        return replace(obj, **changed)

    money = ("dollar_vault", "dollar_vault_capacity", "btc_vault", "btc_vault_capacity", "equity",
             "grc_budget", "operating_costs")
    facts = replace(times(FACTS, 100.0), **{name: getattr(FACTS, name) * 100.0 for name in money})
    cents = FirmProfile(facts=facts, market=times(PROFILE.market, 100.0), risk=times(PROFILE.risk, 100.0))
    a, b = to_model(PROFILE), to_model(cents)
    assert b.scale == 100.0 * a.scale
    assert_close(asdict(b.params), asdict(a.params))
    assert_close(b.state, a.state)


def test_the_cost_of_raising_equity_passes_through_both_statements():
    model = to_model(PROFILE)
    for stated in (FACTS.small_raise, FACTS.large_raise):
        e = model.units(stated.amount)
        assert float(issuance_cost(model.params, t(e))) / e == pytest.approx(stated.cost, rel=1e-13)


@pytest.mark.parametrize("n", [1, 4])
def test_loan_demand_passes_through_its_statement(n):
    profile = illustrative(replace(FACTS, decisions_per_year=n))
    profile = replace(profile, market=replace(profile.market, loan_demand=LoanPoint(loans=90e6, rate=0.08)))
    model = to_model(profile)
    p = model.params.per_period()
    rate = (1 + 0.08) ** (1 / n) - 1
    placed = float(loan_demand(p, t(rate), t(profile.market.reference_ltv)))
    assert model.dollars(placed) == pytest.approx(90e6, rel=1e-13)
    assert float(loan_demand(p, t(p.rate_max), t(profile.market.reference_ltv))) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("n", [1, 4])
def test_the_grc_budget_is_the_spend_that_keeps_the_stock_where_it_is(n):
    model = to_model(illustrative(replace(FACTS, decisions_per_year=n)))
    p, S = model.params.per_period(), model.state.S
    spend = model.units(FACTS.grc_budget) / n
    assert float(next_grc(p, t(S), t(spend))) == pytest.approx(S, rel=1e-13)
    assert S == pytest.approx(steady_stock(model.units(FACTS.grc_budget), model.params.delta, n), rel=1e-15)


@pytest.mark.parametrize("n", [1, 4])
def test_grc_does_what_the_statement_says(n):
    """At the stated spend, kept up, the stated share of the risk is gone;
    with far more, the limit and no more."""
    stated = GrcEffect(spend=1.5e6, reduction=0.4, limit=0.65)
    profile = illustrative(replace(FACTS, decisions_per_year=n))
    profile = replace(profile, risk=replace(profile.risk, grc_on_incidents=stated))
    model = to_model(profile)
    p = model.params
    stock = steady_stock(model.units(stated.spend), p.delta, n)
    assert 1 - float(mitigation(t(stock), p.a_J, p.rho_J)) == pytest.approx(0.4, rel=1e-13)
    assert 1 - float(mitigation(t(1e6 * stock), p.a_J, p.rho_J)) == pytest.approx(0.65, rel=1e-13)
    # the other mechanisms are untouched
    assert (p.a_A, p.rho_A) == pytest.approx((to_model(illustrative(replace(FACTS, decisions_per_year=n))).params.a_A, 0.3))


def test_a_zero_reduction_means_grc_does_nothing():
    risk = replace(PROFILE.risk, grc_on_spread=GrcEffect(spend=1e6, reduction=0.0, limit=0.7))
    assert to_model(replace(PROFILE, risk=risk)).params.a_B == 0.0


@pytest.mark.parametrize("change, message", [
    (dict(facts=replace(FACTS, dollar_vault_capacity=0.0)), "sets the scale"),
    (dict(facts=replace(FACTS, btc_vault=-1.0)), "cannot be negative"),
    (dict(facts=replace(FACTS, interest_share=1.4)), "interest share must be between 0 and 1"),
    (dict(facts=replace(FACTS, decisions_per_year=0)), "whole number"),
    (dict(facts=replace(FACTS, small_raise=RaiseCost(30e6, 0.05))), "the small one first"),
    (dict(facts=replace(FACTS, large_raise=RaiseCost(20e6, 0.01))), "cannot cost less"),
    (dict(facts=replace(FACTS, small_raise=RaiseCost(19e6, 0.01), large_raise=RaiseCost(20e6, 0.5))), "less than nothing"),
    (dict(market=replace(PROFILE.market, loan_demand=LoanPoint(1e6, 0.3))), "below the one at which nobody borrows"),
    (dict(risk=replace(PROFILE.risk, grc_on_incidents=GrcEffect(1e6, 0.8, 0.7))), "below the limit"),
    (dict(risk=replace(PROFILE.risk, grc_on_losses=GrcEffect(0.0, 0.1, 0.7))), "spend must be positive"),
    (dict(facts=replace(FACTS, ltv_max=0.9)), "ltv_max"),   # the model's own check
])
def test_a_profile_the_model_cannot_represent_is_refused_with_a_reason(change, message):
    with pytest.raises(ValueError, match=message):
        to_model(replace(PROFILE, **change))


def test_the_example_is_valid_and_lists_its_inputs():
    validate(PROFILE)
    names = profile_fields()
    assert set(names) == {"facts", "market", "risk"}
    assert all(len(v) > 10 for v in names.values())
    assert math.isclose(PROFILE.market.loan_demand.rate, 0.10)
