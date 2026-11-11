"""The infinite-horizon solver (design.md section 4).

Coarse grids throughout, so the suite stays fast. Each solve takes a few
seconds on CPU in float64.
"""

from dataclasses import replace

import pytest
import torch

from quant.numerics import REFERENCE
from quant.params import DEFAULTS
from quant.solver import GridSpec, riskless_steady_state, solve

P = REFERENCE
SPEC = GridSpec(n_w=33, n_s=5, s_hi=0.08, w_lo=-0.4, w_hi=0.6, n_hermite=3, n_mixing=4)
# GRC that is worth buying and a product that is not free money, so the
# controls are interior. The defaults are not yet calibrated for this.
BASE = replace(DEFAULTS, m=0.02, s_P=0.03, a_A=100.0, a_B=100.0, pi=0.002)


@pytest.fixture(scope="module")
def base_solution():
    return solve(BASE, P, SPEC)


def test_converges(base_solution):
    sol = base_solution
    assert sol.residual < 1e-6
    assert sol.upper_edge_pays_out, "grid must reach past the payout threshold"
    assert sol.lower_edge_is_walk_away, "grid must reach down into walk-away"


def test_value_is_nonnegative_and_increasing(base_solution):
    V = base_solution.V
    assert bool((V >= 0).all())
    assert bool((V[1:] - V[:-1] >= -1e-9).all())


def test_riskless_bank_matches_the_closed_form():
    """No shocks, inert GRC: V is known exactly above the steady-state floor."""
    p = replace(BASE, s_P=0.0, s_N=0.0, s_M=0.0, a_A=0.0, a_B=0.0, nu=float("inf"))
    # The steady state sits just above the kink where issuance starts. Linear
    # interpolation is exact there on the coarse grid; Catmull-Rom bulged over
    # the kink by 1.7e-4, which is part of why it is no longer the default.
    spec = replace(SPEC, n_hermite=1, n_mixing=1)
    sol = solve(p, P, spec, improve_steps=80)
    exact = riskless_steady_state(p, P)
    above = sol.grid.w >= exact.K + 0.05
    got = sol.V[above, 0]
    want = exact.value(sol.grid.w[above])
    assert torch.allclose(got, want, atol=1e-6), float((got - want).abs().max())


def test_tradeable_risk_is_hedged_away_from_walk_away(base_solution):
    """Prop. 2 in the dynamic model, where the bank is risk-averse.

    Under Catmull-Rom interpolation this failed with matched +/- errors of
    0.03: the interpolant had a convex patch and the solver gambled on it.
    """
    sol = base_solution
    c = sol.controls()
    w, _ = sol.grid.states()
    safe = w >= 0.0
    residual = BASE.b_M * c.I - c.H
    assert float(residual[safe].abs().max()) < 1e-3


def test_free_equity_buys_a_buffer(base_solution):
    """r_E = r, tau = 0: carrying equity costs nothing, so the bank holds a buffer.

    Above some level it is *indifferent* between holding more and paying out,
    so K is not pinned there and the test asks only that the buffer exist. At
    r_E = 6% the same bank holds only its floor (next test).
    """
    p = replace(BASE, r_E=BASE.r, tau=0.0)
    sol = solve(p, P, SPEC)
    c = sol.controls()
    w, _ = sol.grid.states()
    top = w >= 0.4
    assert float((c.K[top] - BASE.k * c.I[top]).min()) > 0.1


def test_costly_equity_holds_only_the_floor(base_solution):
    """At r_E = 6% the bank holds the minimum capital and pays out the rest.

    Not an oracle -- a finding at the current calibration, pinned so a change
    to it is noticed: raising equity after a loss is cheaper than carrying a
    buffer.
    """
    sol = base_solution
    c = sol.controls()
    w, _ = sol.grid.states()
    top = w >= 0.4
    floor = BASE.k * c.I[top]
    assert float((c.K[top] - floor).abs().max()) < 1e-3


def test_answer_does_not_depend_on_the_step_budget(base_solution):
    """Doubling the inner optimizer's steps must not move the value."""
    longer = solve(BASE, P, SPEC, improve_steps=80, first_improve_steps=800)
    assert float((longer.V - base_solution.V).abs().max()) < 1e-3
