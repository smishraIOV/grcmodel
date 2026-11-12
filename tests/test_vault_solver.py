"""The vault form on an infinite horizon (design.md sections 4 and 5)."""

import math
from dataclasses import replace

import pytest
import torch

from quant.numerics import REFERENCE
from quant.params import DEFAULTS
from quant.vault_solver import VaultSpec, riskless_vault, solve_vault

P = REFERENCE
SMALL = VaultSpec(n_w=17, n_s=3, n_A=5, n_hermite=2, n_mixing=3, n_jump=3)
# No view on BTC: with one, the firm treats BTC as a positive-NPV project and
# the solve degenerates into a BTC holding company (design.md section 11).
NO_VIEW = replace(DEFAULTS, periods_per_year=1, mu_BTC=0.0)


def riskless_params(**overrides):
    base = replace(
        DEFAULTS, periods_per_year=1, nu=math.inf, vault_vol=(0.0, 0.0, 0.0), s_BTC=0.0,
        mu_BTC=0.0, jump_rate=0.0, a_A=0.0, a_B=0.0, a_J=0.0,
        vault_mu=(0.03, 0.03, 0.03), vault_loss=(0.0, 0.0, 0.0),
    )
    return replace(base, **overrides)


@pytest.mark.parametrize("periods", [1, 4])
def test_riskless_vault_matches_the_closed_form(periods):
    """No shocks: AUM settles at A*, the firm holds its floor and pays out the
    rest, and its value is known exactly at A*."""
    p = riskless_params(periods_per_year=periods)
    exact = riskless_vault(p)
    # A* on a grid node, so no interpolation in AUM; one scenario node.
    spec = VaultSpec(A_lo=exact.A - 0.4, A_hi=exact.A + 0.4, n_A=9, n_w=65, n_s=3,
                     n_hermite=1, n_mixing=1)
    sol = solve_vault(p, P, spec)
    w = sol.grid.axes[0]
    assert float(sol.grid.axes[2][spec.n_A // 2]) == pytest.approx(exact.A, abs=1e-12)
    above = w >= exact.K + 0.05
    got = sol.V[above, 0, spec.n_A // 2]
    want = exact.value(w[above])
    assert torch.allclose(got, want, atol=1e-5), float((got - want).abs().max())


@pytest.fixture(scope="module")
def no_view():
    return solve_vault(NO_VIEW, P, SMALL)


def test_converges_and_reaches_both_edges(no_view):
    assert no_view.residual < 1e-6
    assert no_view.lower_edge_is_walk_away
    assert no_view.upper_edge_pays_out


def test_value_is_nonnegative_and_increasing_in_equity(no_view):
    V = no_view.V
    assert bool((V >= 0).all())
    assert bool((V[1:] - V[:-1] >= -1e-9).all())


def test_allocation_is_a_split_of_the_vault(no_view):
    alloc = no_view.controls().alloc
    assert torch.allclose(alloc.sum(-1), torch.ones(alloc.shape[0], dtype=P.dtype), atol=1e-12)
    assert bool((alloc >= 0).all())


def test_with_no_view_and_no_btc_in_the_vault_the_firm_holds_no_net_btc():
    """Prop. 2 for the vault: nothing to gain from BTC, so no net position."""
    p = replace(NO_VIEW, vault_btc=(0.0, 0.0, 0.0))
    sol = solve_vault(p, P, SMALL)
    c = sol.controls()
    w, _, _ = sol.states()
    safe = w >= 0.0
    assert float((c.q - c.H)[safe].abs().max()) < 1e-3
