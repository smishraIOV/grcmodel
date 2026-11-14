"""Assumptions shown as assumptions (docs/plan-tool.md step 7): each dial
moves the assumption it names and nothing else, the outputs it cannot touch
do not move, and the change between two policies can be followed the same way.
"""

from dataclasses import replace

import pytest

from quant.policy import Policy
from quant.profile import example_facts, illustrative, to_model
from quant.readout import read_out, scenario_set
from quant.sensitivity import (
    GRC_EFFECTS,
    Output,
    outlook,
    report,
    sensitivity,
    standard_dials,
    standard_outputs,
)
from scripts.whatif import main

PROFILE = illustrative(example_facts())
MODEL = to_model(PROFILE)
POLICY = Policy(
    equity_kept=19e6, grc_spend=0.4e6, own_btc=2e6, hedged=3e6, borrower_rate=0.09, ltv=0.6,
    allocation=(0.2, 0.4, 0.1, 0.3),
)
COARSE = dict(n_ops=1, n_strategy=2, n_btc=41, n_mixing=3, n_crash=8, n_jump=3)   # quick; the tests compare like with like
DIALS = {d.name: d for d in standard_dials(PROFILE)}


@pytest.fixture(scope="module")
def base():
    return sensitivity(PROFILE, POLICY, counts=COARSE)


def test_every_dial_gives_a_profile_the_model_accepts_at_both_ends():
    for d in DIALS.values():
        assert d.low < d.high and d.low <= d.base <= d.high, d.name
        for value in (d.low, d.base, d.high):
            to_model(d.apply(PROFILE, value))


def test_a_dial_at_its_base_is_the_profile_as_entered():
    for d in DIALS.values():
        assert to_model(d.apply(PROFILE, d.base)).params == MODEL.params, d.name


def test_each_dial_moves_what_it_names():
    grc = DIALS["how well GRC works (0: not at all)"]
    off = grc.apply(PROFILE, 0.0)
    assert all(getattr(off.risk, name).reduction == 0.0 for name in GRC_EFFECTS)
    p = to_model(off).params
    assert (p.a_A, p.a_B, p.a_J, p.a_L) == (0.0, 0.0, 0.0, 0.0)
    more = grc.apply(PROFILE, 1.5)
    for name in GRC_EFFECTS:      # half again as well, short of the limit
        was, now = getattr(PROFILE.risk, name), getattr(more.risk, name)
        assert now.reduction == pytest.approx(min(1.5 * was.reduction, 0.98 * was.limit)) and now.limit == was.limit
    crashes = DIALS["BTC crashes a year"]
    assert crashes.apply(PROFILE, crashes.high).market.btc_crashes_per_year == 2 * PROFILE.market.btc_crashes_per_year
    discount = DIALS["sale discount in a liquidation"]
    assert discount.high < 1 - PROFILE.facts.liquidation_threshold       # or selling at the threshold leaves bad debt
    leaving = DIALS["investors leaving after an incident"].apply(PROFILE, 2.0).risk
    assert (leaving.outflow_after_incident, leaving.outflow_from_other_vault) == (0.4, 0.2)


def test_the_base_is_the_read_out_s_own_figures(base):
    r = read_out(MODEL, POLICY, scenario_set(MODEL.params.per_period(), **COARSE))
    assert base.base["expected equity after one period"] == r.expected_equity
    assert base.base["chance of an incident in a year"] == r.incident_chance
    assert base.base["revenue a year"] == r.revenue
    assert not base.versus
    assert not any(o.needs_paths for o in base.outputs)                  # no horizon was asked for


def test_grc_that_works_better_means_fewer_incidents_and_none_means_the_bare_rate(base):
    name, output = "how well GRC works (0: not at all)", "chance of an incident in a year"
    lo, hi = base.low[name][output], base.high[name][output]
    assert hi < base.base[output] < lo
    bare = 1 - (1 - 2 * (1 - 2.718281828459045 ** (-0.1 / 4))) ** 4      # two vaults, a tenth a year each
    assert lo == pytest.approx(bare, rel=1e-12)


def test_an_output_does_not_move_with_an_assumption_it_cannot_depend_on(base):
    incident = "chance of an incident in a year"
    assert "BTC volatility" in base.unaffected(incident) and "loan demand" in base.unaffected(incident)
    moving = [d.name for d, _, _ in base.swings(incident)]
    assert set(moving) == {"how well GRC works (0: not at all)", "incidents a year, with no GRC"}
    crash = "equity after: BTC crashes 50%"                               # one outcome: no chances enter it
    for name in ("BTC crashes a year", "BTC volatility", "incidents a year, with no GRC"):
        assert name in base.unaffected(crash)
    assert "sale discount in a liquidation" in [d.name for d, _, _ in base.swings(crash)]
    # every dial is either moving an output or listed as not moving it
    for o in base.outputs:
        assert len(base.swings(o.name)) + len(base.unaffected(o.name)) == len(base.dials)


def test_the_dial_that_moves_an_output_most_comes_first(base):
    for o in base.outputs:
        sizes = [abs(hi - lo) for _, lo, hi in base.swings(o.name)]
        assert sizes == sorted(sizes, reverse=True)


def test_the_change_between_two_policies_is_followed_the_same_way():
    """More GRC spend against less, over three years on the same paths. With
    GRC assumed to do nothing, the extra spend is only a cost."""
    dials = (DIALS["how well GRC works (0: not at all)"], DIALS["incidents a year, with no GRC"])
    outputs = tuple(o for o in standard_outputs() if o.name in (
        "chance of an incident in a year", "what shareholders take, over the horizon"))
    more = replace(POLICY, grc_spend=1.2e6, equity_kept=19.2e6)
    kwargs = dict(dials=dials, outputs=outputs, years=3, paths=4000, seed=3, counts=COARSE)
    gain = sensitivity(PROFILE, POLICY, versus=more, **kwargs)
    one, two = sensitivity(PROFILE, POLICY, **kwargs), sensitivity(PROFILE, more, **kwargs)
    assert gain.versus
    for name in gain.base:
        assert gain.base[name] == pytest.approx(two.base[name] - one.base[name], rel=1e-9, abs=1e-9)
        for d in dials:
            assert gain.high[d.name][name] == pytest.approx(two.high[d.name][name] - one.high[d.name][name], rel=1e-9, abs=1e-9)
    grc, incident, take = dials[0].name, "chance of an incident in a year", "what shareholders take, over the horizon"
    assert gain.low[grc][incident] == pytest.approx(0.0, abs=1e-12)        # spend that does nothing prevents nothing
    assert gain.high[grc][incident] < 0.0                                   # spend that works prevents some
    assert gain.low[grc][take] < 0.0                                        # and doing nothing, it only costs
    assert gain.high[grc][take] > gain.low[grc][take]                       # it is worth more the better it works


def test_outputs_can_be_the_user_s_own():
    loans = Output("loans", "money", lambda o: o.read_out.loans)
    s = sensitivity(PROFILE, POLICY, dials=(DIALS["loan demand"], DIALS["BTC volatility"]), outputs=(loans,), counts=COARSE)
    assert [d.name for d, _, _ in s.swings("loans")] == ["loan demand"] and s.unaffected("loans") == ["BTC volatility"]
    look = outlook(PROFILE, POLICY, counts=COARSE)
    assert look.paths is None and len(look.stresses) == 2 and loans.pick(look) == s.base["loans"]


def test_the_report_and_the_script(base, capsys):
    text = report(base, top=3)
    assert "chance of an incident in a year:" in text and "as entered" in text
    assert "do not move it at all" in text and "The ranges are not estimates" in text
    main(["--sensitivity"])
    assert "how well GRC works" in capsys.readouterr().out
