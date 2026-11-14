"""The BTC yield vault on an infinite horizon (design.md sections 2.3, 3 and 5).

Annual and coarse, so the suite stays fast; realistic runs are quarterly.
"""

import math
from dataclasses import replace

import pytest
import torch

from quant.research.btc_vault_solver import BtcVaultSpec, market_value, riskless_btc_vault, solve_btc_vault
from quant.numerics import REFERENCE
from quant.params import DEFAULTS

pytestmark = pytest.mark.slow   # run with `uv run pytest -m slow`

P = REFERENCE
BASE = replace(DEFAULTS, periods_per_year=1)
SPEC = BtcVaultSpec(n_w=17, n_s=3, n_A=5, n_strategy=1, n_btc=5, n_mixing=3, n_crash=3, n_jump=3)


def riskless_params(**overrides):
    base = replace(
        BASE, nu=math.inf, s_BTC=0.0, btc_crash_rate=0.0, jump_rate=0.0,
        strategy_vol=(0.0, 0.0, 0.0), strategy_yield=(0.03, 0.03, 0.03),
        strategy_loss=(0.0, 0.0, 0.0), a_A=0.0, a_B=0.0, a_J=0.0,
    )
    return replace(base, **overrides)


@pytest.mark.parametrize("periods", [1, 4])
def test_riskless_vault_matches_the_closed_form(periods):
    """No randomness: the vault settles at A*, the firm holds its floor, no BTC
    of its own and no hedge, pays out the rest, and its value is exact at A*."""
    p = riskless_params(periods_per_year=periods)
    exact = riskless_btc_vault(p)
    # A* on a grid node, so no interpolation in the vault's size; one scenario.
    # The floor K* and the steady state w_ss on equity nodes too: the value has
    # a kink at K*, and a node between them would blur it.
    h = exact.w_ss - exact.K
    spec = BtcVaultSpec(A_lo=exact.A - 0.4, A_hi=exact.A + 0.4, n_A=9,
                        w_lo=exact.K - 10 * h, w_hi=exact.K + 50 * h, n_w=61, n_s=3,
                        n_strategy=1, n_btc=1, n_mixing=1, n_crash=1, n_jump=1)
    sol = solve_btc_vault(p, P, spec)
    w = sol.grid.axes[0]
    assert float(sol.grid.axes[2][spec.n_A // 2]) == pytest.approx(exact.A, abs=1e-12)
    # The optimizer holds K a hair above the floor (softplus); the small cost of
    # that, every period, adds up over about 1 / (1 - beta) periods: 1e-5 at
    # quarterly frequency.
    above = w >= exact.K - 1e-12
    got = sol.V[above, 0, spec.n_A // 2]
    want = exact.value(w[above])
    assert torch.allclose(got, want, atol=3e-5), float((got - want).abs().max())


@pytest.fixture(scope="module")
def base():
    return solve_btc_vault(BASE, P, SPEC)


@pytest.fixture(scope="module")
def biased():
    return solve_btc_vault(replace(BASE, btc_bias=0.04), P, SPEC)


def test_converges_and_reaches_both_edges(base):
    # The solver settles to about 2e-3 and no further (docs/plan-integrated.md section 7).
    assert base.residual < 1e-3
    assert base.lower_edge_is_walk_away
    assert base.upper_edge_pays_out


def test_value_is_nonnegative_and_increasing_in_equity(base):
    V = base.V
    assert bool((V >= 0).all())
    assert bool((V.diff(dim=0) >= -1e-9).all())


def test_decisions_respect_their_bounds(base):
    c, p = base.controls(), base.params
    _, _, A = base.states()
    assert bool((c.H <= c.q + p.btc_co_invest * A + 1e-15).all())
    assert torch.allclose(c.alloc.sum(-1), P.ones(A.shape[0]), atol=1e-14)


def test_with_no_bias_the_firm_holds_no_btc_of_its_own(base):
    """Unhedged it earns r_E less its security cost, below what shareholders
    require; hedged it earns perp funding, less again. Beyond its stake in the
    vault, the firm holds essentially none."""
    assert float(base.controls().q.max()) < 0.01


@pytest.mark.xfail(
    reason="open: the biased firm's policy, valued at market, beats the unbiased solve by up to "
           "5e-3, so the unbiased solve stops short of its best (docs/plan-integrated.md section 7)",
    strict=False,
)
def test_the_bias_cannot_beat_the_unbiased_firm_at_market_prices(base, biased):
    V_market = market_value(biased)
    assert bool((V_market <= base.V + 1e-6).all())


def test_answer_does_not_depend_on_the_step_budget(base):
    more = solve_btc_vault(BASE, P, SPEC, improve_steps=80, polish_steps=600)
    assert float((more.V - base.V).abs().max()) < 1e-4    # values are of order 0.1-1
