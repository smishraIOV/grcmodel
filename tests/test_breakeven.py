"""The break-evens are the project's actual output, so their shape is tested.

Small configurations throughout: these assert that each break-even means what
it claims, not that any particular number is right. The numbers are
illustrative and every one of them moves when a parameter does.
"""

from dataclasses import replace

import pytest

from quant.env.shocks import CommonRandomNumbers
from quant.numerics import REFERENCE
from quant.params import DEFAULTS
from quant.studies.breakeven import (
    FAMILIES,
    is_material,
    alpha_breakeven,
    capitalization_band,
    franchise_breakeven,
    hazard_breakeven,
    solve,
    value_curvature,
)

PERIODS, PATHS, STEPS = 4, 512, 800


def crn():
    return CommonRandomNumbers(0, PERIODS, PATHS, REFERENCE)


def test_hazard_breakeven_is_cost_over_franchise():
    """The only break-even with no alpha in it: both inputs are things a board
    already has a view on, so the whole claim can be argued with without
    touching an uncalibrated parameter."""
    result = hazard_breakeven(DEFAULTS.firm, crn(), periods=PERIODS, steps=STEPS)
    assert result.annual_cost > 0 and result.franchise > 0
    assert result.basis_points == pytest.approx(
        1e4 * result.annual_cost / result.franchise, rel=1e-12
    )
    assert "basis points" in result.sentence()


def test_spend_peaks_at_intermediate_capitalization():
    """The most decision-useful shape the model produces, and it is
    non-monotone: well capitalized, the hazard is too small to be worth buying
    down; thinly capitalized, there is too little franchise left to protect.
    A model that made GRC monotone in capital would be saying something false
    about both ends."""
    rows = capitalization_band(
        DEFAULTS.firm, crn(), [4.0, 8.0, 16.0, 32.0, 64.0], periods=PERIODS, steps=STEPS
    )
    spend = [r.total_grc for _, r in rows]
    assert max(spend) > spend[0] and max(spend) > spend[-1], f"monotone: {spend}"
    assert not is_material(rows[-1][1], DEFAULTS.firm), (
        "a very safe firm should stop paying for a programme"
    )


def test_a_thin_franchise_does_not_justify_a_programme():
    """GRC is bought by the business it protects, so removing the business has
    to remove the programme. That direction is what makes the survival story a
    claim rather than a slogan.

    Swept through the going-concern value directly. An earlier version swept
    productivity instead, which leaves the terminal franchise pinned, so the
    firm always had something worth protecting and every row reported "worth
    running".
    """
    rows = franchise_breakeven(
        DEFAULTS.firm, crn(), [0.0, 15.0], periods=PERIODS, steps=STEPS
    )
    nothing, going_concern = rows[0][1], rows[1][1]
    assert going_concern.total_grc > nothing.total_grc, (
        "more at stake must buy more protection"
    )
    assert going_concern.annual_death_probability < nothing.annual_death_probability
    assert going_concern.value > nothing.value


def test_effectiveness_breakeven_is_far_below_the_risk_neutral_one():
    """The Froot-Stein premium expressed as a threshold. A unit of spend buys
    franchise protection as well as a smaller loss, so the programme pays at
    effectiveness a pure expected-loss calculation rejects.

    Bisected on value *added*, not on budget size. Optimal spend is
    non-monotone in effectiveness -- ln(alpha X)/alpha rises then falls,
    because a very effective programme needs little spending on -- so a
    materiality test on the budget reports "no alpha works" for a programme
    that is enormously worthwhile.
    """
    alpha = alpha_breakeven(
        DEFAULTS.firm, crn(), "operational", low=0.0005, iterations=5,
        periods=PERIODS, steps=STEPS,
    )
    assert alpha is not None, "no effectiveness justified the programme"
    assert alpha < 1.0 / 3.5, "should be below the risk-neutral break-even 1/exposure"


def test_value_curvature_is_reported_with_a_sign():
    """The diagnostic to run before quoting any comparative static. A convex
    region means the firm is risk-loving near failure -- gambling for
    resurrection -- and a learner will find it and recommend cutting GRC in a
    crisis, which is correct inside the model and indefensible outside it.

    Asserts the machinery, not the sign: at the current parameters no convex
    region is detected, and if one appears that is news rather than a failure.
    """
    equities = [4.0, 8.0, 16.0, 32.0]
    values, curvature = value_curvature(
        DEFAULTS.firm, crn(), equities, periods=PERIODS, steps=STEPS
    )
    assert len(values) == len(equities)
    assert len(curvature) == len(equities) - 2
    assert values[-1] > values[0], "more capital should be worth more"
