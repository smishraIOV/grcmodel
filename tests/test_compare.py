"""Two policies side by side (docs/plan-tool.md step 6): the comparison is
the two policies' own read-outs, stresses and paths, on the same scenarios,
and the difference over several years is measured path by path.
"""

import math
from dataclasses import replace

import pytest
import torch

from quant.compare import compare, report
from quant.paths import rule_from, simulate
from quant.policy import Policy
from quant.profile import example_facts, illustrative, to_model
from quant.readout import read_out
from quant.stress import Stress, run
from scripts.whatif import main

MODEL = to_model(illustrative(example_facts(100e6)))
A = Policy(
    equity_kept=19e6, grc_spend=0.4e6, own_btc=2e6, hedged=3e6, borrower_rate=0.09, ltv=0.5,
    allocation=(0.3, 0.5, 0.2, 0.0),
)
B = replace(A, ltv=0.7, allocation=(0.2, 0.4, 0.1, 0.3))


@pytest.fixture(scope="module")
def both():
    return compare(MODEL, A, B, names=("cautious", "bold"), years=3, paths=5000, seed=9)


def test_each_side_is_that_policy_s_own_result(both):
    assert both.read_outs[0] == read_out(MODEL, A) and both.read_outs[1] == read_out(MODEL, B)
    crash = Stress(name="BTC crashes 50%", btc_crash=0.50)
    for side, policy in zip(both.stresses, (A, B), strict=True):
        (found,) = [r for r in side if r.stress == crash]
        assert found == run(MODEL, policy, crash)
    alone = simulate(MODEL, rule_from(B, MODEL), years=3, paths=5000, seed=9)
    assert torch.equal(both.paths[1].equity, alone.equity) and torch.equal(both.paths[1].present_value, alone.present_value)


def test_both_policies_run_on_the_same_paths(both):
    a, b = both.paths
    assert torch.equal(a.btc_price, b.btc_price)          # the same BTC, path by path
    assert not torch.equal(a.equity, b.equity)
    gain, error = both.value_gain()
    assert gain == pytest.approx(b.value()[0] - a.value()[0], rel=1e-9)
    # measured path by path, the difference is known far better than either level
    assert error < 0.5 * math.hypot(a.value()[1], b.value()[1])


def test_a_policy_against_itself_changes_nothing():
    same = compare(MODEL, A, A, years=1, paths=500, seed=1)
    assert same.read_outs[0] == same.read_outs[1] and same.stresses[0] == same.stresses[1]
    assert same.value_gain() == (0.0, 0.0)
    assert "too small to tell" in report(same)


def test_without_a_horizon_there_are_no_paths():
    quick = compare(MODEL, A, B)
    assert quick.paths is None and "Over" not in report(quick).split("Under stress")[1]
    with pytest.raises(ValueError, match="no horizon"):
        quick.value_gain()
    few = compare(MODEL, A, B, stresses=(Stress(name="only this", btc_move=-0.2),))
    assert len(few.stresses[0]) == len(few.stresses[1]) == 1


def test_an_unsound_policy_is_refused_by_name():
    with pytest.raises(ValueError, match="policy bold cannot be run: .*runway"):
        compare(MODEL, A, replace(B, equity_kept=5e6), names=("cautious", "bold"))


def test_the_report_says_what_changed_and_which_is_ahead(both):
    text = report(both)
    for line in ("The two policies", "One period ahead", "Under stress, after one period",
                 "Over 3 years, on the same 5,000 random paths", "BTC crashes 50%", "+20.00pt"):
        assert line in text
    gain, error = both.value_gain()
    ahead = "bold is ahead" if gain > 2 * error else "cautious is ahead" if gain < -2 * error else "too small to tell"
    assert ahead in text and "does not count risk" in text


def test_the_script_compares_when_given_a_second_policy(capsys):
    main(["--ltv", "0.5", "--b-ltv", "0.7", "--b-sleeve", "0.2", "--years", "1", "--paths", "300"])
    printed = capsys.readouterr().out
    assert "The two policies" in printed and "first" in printed and "second" in printed
    main(["--b-equity-kept", "1e5"])
    assert "The second policy cannot be run" in capsys.readouterr().out
