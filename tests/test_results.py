"""The v0 outputs, and the last of the design's oracles."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from quant.numerics import REFERENCE
from quant.results import grc_value, risk_map, target_stock, thresholds
from quant.solver import solve
from tests.test_solver import BASE, SPEC

P = REFERENCE
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def solved():
    return solve(BASE, P, SPEC)


def test_hurdle_equals_the_product_return_where_it_is_held():
    """First-order condition: at an interior alpha the marginal unit just
    clears its hurdle, so the hurdle equals pi. Exact under any shocks, which
    is what makes the hurdle a measurement rather than a formula.

    Only where the bank operates (V > 0.05): in the walk-away region its
    controls are never used and mean nothing. On a 129-point grid the mean
    error is about 2% of pi and the worst point about 19%, both grid noise
    from slopes of a piecewise-linear value function; on 33 points the worst
    was 30%.
    """
    sol = solve(BASE, P, replace(SPEC, n_w=129))
    rm = risk_map(sol)
    alpha = sol.controls().alpha
    held = (alpha > 0.02) & (alpha < 0.98 * BASE.alpha_max) & (sol.V.reshape(-1) > 0.05)
    error = (rm.hurdle_N[held] - BASE.pi).abs()
    assert int(held.sum()) > 100
    assert float(error.mean()) < 1e-4
    assert float(error.max()) < 5e-4


def test_risk_aversion_is_positive_and_rises_as_capital_thins(solved):
    rm = risk_map(solved)
    w, s = solved.grid.states()
    G = rm.G[s == 0]
    ws = w[s == 0]
    assert bool((G[ws >= -0.2] > 0).all())
    # thinner capital, more risk-averse: compare the payout region to w = 0
    at_zero = float(G[torch.argmin(ws.abs())])
    at_top = float(G[-1])
    assert at_zero > at_top


def test_thresholds_are_ordered(solved):
    th = thresholds(solved)
    assert bool((th.walk_away < th.payout).all())


def test_target_stock_is_interior(solved):
    assert 0.0 < target_stock(solved) < SPEC.s_hi


def test_without_non_tradeable_risk_there_is_no_premium():
    """Oracle 5: with no risk the bank must keep, GRC is worth only its
    expected-loss saving, so the constrained bank and its frictionless twin
    value it identically."""
    p = replace(BASE, s_P=0.0, s_N=0.0, jump_rate=0.0)
    gv = grc_value(p, P, SPEC)
    assert gv.gain > 0
    assert gv.premium == pytest.approx(0.0, abs=1e-5)


def test_every_entry_point_imports():
    """A script nothing imports can be broken for days."""
    for path in sorted((ROOT / "scripts").glob("*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "main"), path.name
