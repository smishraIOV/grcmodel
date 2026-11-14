"""The integrated firm on the coarse 4-D grid (docs/plan-integrated.md steps
4 and 4a). Annual and very coarse, so the tests stay quick: they check that
the solve is well-formed and that the read-out adds up, not what it finds.
"""

from dataclasses import replace

import pytest
import torch

from quant.integrated_results import readout, state_index
from quant.integrated_solver import IntegratedSpec, market_value, solve_integrated
from quant.numerics import REFERENCE
from quant.params import INTEGRATED_DEFAULTS
from scripts.integrated import report

P = REFERENCE
BASE = replace(INTEGRATED_DEFAULTS, periods_per_year=1)
SPEC = IntegratedSpec(n_w=13, n_s=2, n_L=3, n_B=3, n_ops=1, n_strategy=1, n_btc=5, n_mixing=3,
                      n_crash=3, n_jump=2)


@pytest.fixture(scope="module")
def base():
    return solve_integrated(BASE, P, SPEC, max_iterations=25)


def test_converges_and_reaches_both_edges(base):
    # The solver settles to about 2e-3 and no further (docs/plan-integrated.md section 7).
    assert base.residual < 1e-3
    assert base.lower_edge_is_walk_away
    assert base.upper_edge_pays_out


def test_value_is_nonnegative_and_increasing_in_equity(base):
    assert float(base.V.min()) >= 0.0
    assert float((base.V[1:] - base.V[:-1]).min()) >= -1e-9


def test_decisions_respect_their_bounds(base):
    p = base.params
    c = base.controls()
    _, _, A_L, A_B = base.states()
    assert float(c.g.min()) >= 0.0 and float(c.q.min()) >= 0.0
    assert bool((c.H <= c.q + p.btc_co_invest * A_B + 1e-15).all())
    assert bool((c.K >= p.k_A * (A_L + A_B) - 1e-15).all())
    assert bool((c.rate <= p.rate_max).all()) and bool((c.ltv <= p.ltv_max).all())
    assert torch.allclose(c.alloc.sum(-1), P.ones(A_L.shape[0]), atol=1e-14)


def test_the_market_value_cannot_exceed_the_firm_s_own_with_no_bias(base):
    """With no bias the two expectations are the same, so re-valuing the
    policy at the market's changes nothing beyond what the solve left
    unconverged."""
    assert float((market_value(base) - base.V).abs().max()) < 1e-3


def test_the_read_out_adds_up(base):
    r = readout(base, at_w=0.3, at_L=1.0, at_B=2.0)
    i = state_index(base, 0.3, 1.0, 2.0)
    w, s, A_L, A_B = (float(x[i]) for x in base.states())
    assert (r.equity, r.grc_stock, r.dollar_vault, r.btc_vault) == (w, s, A_L, A_B)
    assert r.equity_kept == pytest.approx(r.equity - r.dividend + r.issue, abs=1e-12)
    assert r.equity_kept >= r.capital_floor - 1e-12
    assert r.stablecoins >= base.params.runway * base.params.opex - 1e-12
    assert sum(r.allocation) == pytest.approx(1.0, abs=1e-12)
    assert r.revenue == pytest.approx(r.interest_income + r.btc_vault_fees)
    assert r.payroll_ratio == pytest.approx(BASE.payroll_share * r.operating_ratio)
    assert 0.0 <= r.chance_walk_away <= r.chance_below_floor + 1e-12 <= 1.0 + 1e-12
    assert 0.0 <= r.incident_chance < 2 * BASE.jump_rate
    text = report(r)
    assert "Value to shareholders" in text and "stablecoin lending sleeve" in text
