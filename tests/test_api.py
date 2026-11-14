"""The tool's interface for other programs (docs/plan-tool.md step 9): what
it returns is what the modules behind it return, as data JSON can carry, and
a mistaken request gets reasons, not an exception.
"""

import io
import json
import re
from dataclasses import replace

import pytest

from quant.api import VERSION, WANTS, example_request, inputs, plain, respond
from quant.paths import rule_from, simulate
from quant.policy import starting_policy
from quant.profile import example_facts, illustrative, to_model
from quant.readout import read_out
from quant.stress import Stress, run
from scripts.api import main

FIRM = plain(example_facts())
MODEL = to_model(illustrative(example_facts()))
START = starting_policy(MODEL)


def carried(answer) -> dict:
    """Through JSON and back, strictly: no NaN, no infinity, nothing JSON cannot hold."""
    return json.loads(json.dumps(answer, allow_nan=False))


@pytest.fixture(scope="module")
def answer():
    request = {**example_request(), "paths": 3000}
    return carried(respond(request))


def test_the_example_request_is_answered_in_full(answer):
    assert answer["ok"] and answer["version"] == VERSION and answer["illustrative"] is True
    assert set(answer) >= {"inputs", "read_out", "stress", "paths", "compare"}
    assert set(answer["compare"]) == {"read_out", "stress", "paths", "shareholders_take_gain", "shareholders_take_gain_error"}


def test_the_answer_is_what_the_modules_return(answer):
    profile = illustrative(example_facts())
    model = to_model(replace(profile, risk=replace(profile.risk, incidents_per_year=0.15)))
    policy = replace(starting_policy(model), borrower_rate=0.09, ltv=0.6)
    r = read_out(model, policy)
    assert answer["read_out"]["expected_equity"] == r.expected_equity
    assert answer["read_out"]["revenue"] == r.revenue                   # computed properties come too
    assert answer["read_out"]["allocation"] == list(r.allocation)
    crash = run(model, policy, Stress(name="BTC crashes 50%", btc_crash=0.50))
    (found,) = [s for s in answer["stress"] if s["stress"]["name"] == "BTC crashes 50%"]
    assert found["equity"] == crash.equity and found["equity_change"] == crash.equity_change
    paths = simulate(model, rule_from(policy, model), years=5, paths=3000, seed=0)
    assert answer["paths"]["shareholders_take"] == paths.value()[0]
    assert answer["paths"]["chance_failed"] == paths.chance_failed().tolist()
    assert answer["paths"]["equity"][-1] == paths.quantiles(paths.equity)[-1].tolist()
    assert len(answer["paths"]["year"]) == len(answer["paths"]["equity"]) == len(answer["paths"]["paid_out"]) == 21
    second = replace(policy, ltv=0.7, allocation=(0.25, 0.25, 0.25, 0.25))
    assert answer["compare"]["read_out"][1]["loans"] == read_out(model, second).loans
    other = simulate(model, rule_from(second, model), years=5, paths=3000, seed=0)
    assert answer["compare"]["shareholders_take_gain"] == pytest.approx(other.value()[0] - paths.value()[0], rel=1e-9)


def test_the_inputs_come_back_with_every_default_filled_in(answer):
    echoed = answer["inputs"]
    assert echoed["firm"] == FIRM
    assert echoed["risk"]["incidents_per_year"] == 0.15                 # as changed
    assert echoed["risk"]["incident_loss"] == illustrative(example_facts()).risk.incident_loss   # as defaulted
    assert echoed["policy"]["ltv"] == 0.6 and echoed["policy"]["grc_spend"] == pytest.approx(START.grc_spend)
    assert echoed["second_policy"]["borrower_rate"] == 0.09             # the second starts from the first
    assert echoed["position"]["stablecoins"] > 0


def test_a_firm_alone_is_enough():
    quick = carried(respond({"firm": FIRM}))
    assert quick["ok"] and set(quick) == {"ok", "version", "inputs", "illustrative", "read_out"}
    assert quick["read_out"]["expected_equity"] == read_out(MODEL, START).expected_equity


def test_statements_are_given_as_named_values():
    request = {"firm": FIRM, "risk": {"grc_on_incidents": {"spend": 1e6, "reduction": 0.2, "limit": 0.5}},
               "market": {"loan_demand": {"loans": 50e6, "rate": 0.08}, "strategy_yield": [0.02, 0.05, 0.0]}}
    got = respond(request)
    assert got["ok"]
    assert got["inputs"]["risk"]["grc_on_incidents"] == {"spend": 1e6, "reduction": 0.2, "limit": 0.5}
    assert got["inputs"]["market"]["strategy_yield"] == [0.02, 0.05, 0.0]
    assert got["read_out"]["loans"] < respond({"firm": FIRM})["read_out"]["loans"]


def test_the_sensitivity_and_a_stress_of_the_user_s_own():
    got = carried(respond({"firm": FIRM, "want": ["stress", "sensitivity"],
                           "stresses": [{"name": "mine", "btc_move": -0.2, "btc_crash": 0.4}]}))
    assert got["ok"] and len(got["stress"]) == 1 and got["stress"][0]["btc_price"] == pytest.approx(0.48)
    s = got["sensitivity"]
    assert len(s["dials"]) == 13 and not s["versus"]
    assert set(s["moved_by"]["chance of an incident in a year"]) == {
        "how well GRC works (0: not at all)", "incidents a year, with no GRC"}
    assert set(s["low"]) == {d["name"] for d in s["dials"]}


@pytest.mark.parametrize("request_, message", [
    ({}, "needs a 'firm'"),
    ("not a request", "must be a set of named values"),
    ({"firm": {k: v for k, v in FIRM.items() if k != "equity"}}, "'firm' needs 'equity'"),
    ({"firm": {**FIRM, "equty": 1}}, "'firm' has no input called 'equty'"),
    ({"firm": FIRM, "polcy": {}}, "no entry called 'polcy'"),
    ({"firm": FIRM, "want": ["everything"]}, "not 'everything'"),
    ({"firm": {**FIRM, "dollar_vault_capacity": 0}}, "cannot be used"),
    ({"firm": {**FIRM, "interest_share": 1.5}}, "interest share must be between 0 and 1"),
    ({"firm": FIRM, "market": {"loan_demand": {"loans": 1}}}, "'market.loan_demand' needs 'rate'"),
    ({"firm": FIRM, "policy": {"equity_kept": 5e6}}, "policy: equity kept, .* the runway needs"),
    ({"firm": FIRM, "second_policy": {"ltv": 0.9}}, "second_policy: the loan-to-value of 90%"),
    ({"firm": FIRM, "want": ["compare"]}, "'compare' needs a 'second_policy'"),
    ({"firm": FIRM, "want": ["paths"]}, "'paths' needs 'years'"),
    ({"firm": FIRM, "want": ["paths"], "years": -1}, "'years' must be a positive number"),
    ({"firm": FIRM, "want": ["stress"], "stresses": [{"name": "x", "btc_crash": 1.0}]}, "below all of it"),
    ({"firm": FIRM, "want": ["stress"], "stresses": [{"btc_crash": 0.5}]}, "'stresses' needs 'name'"),
])
def test_a_mistaken_request_is_answered_with_reasons(request_, message):
    got = respond(request_)
    assert got["ok"] is False and got["version"] == VERSION
    assert any(re.search(message, line) for line in got["problems"]), got["problems"]
    json.dumps(got, allow_nan=False)


def test_every_mistake_found_at_once_is_reported():
    got = respond({"firm": FIRM, "polcy": {}, "want": ["everything"]})
    assert len(got["problems"]) == 2


def test_the_inputs_listing_covers_a_request():
    listing = carried(inputs())
    assert listing["want"] == list(WANTS) and listing["firm"] == FIRM
    assert set(listing["policy"]) == {"equity_kept", "grc_spend", "own_btc", "hedged", "borrower_rate", "ltv", "allocation"}
    assert len(listing["allocation_order"]) == 4 and "btc_crash" in listing["stress"]
    assert respond({"firm": listing["firm"], "market": listing["market"], "risk": listing["risk"],
                    "policy": listing["policy"]})["ok"]


def test_the_command(tmp_path, capsys, monkeypatch):
    assert main(["--example"]) == 0
    request = json.loads(capsys.readouterr().out)
    assert request == example_request()
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"firm": FIRM, "want": ["read_out"]}))
    assert main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["read_out"]["equity"] == 20e6
    monkeypatch.setattr("sys.stdin", io.StringIO('{"firm": {}}'))
    assert main(["-"]) == 1
    assert "'firm' needs" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert main(["-"]) == 1
    assert "could not be read" in capsys.readouterr().out
    assert main(["--inputs"]) == 0 and '"allocation_order"' in capsys.readouterr().out
