"""The shared grid: multilinear interpolation and equity's off-grid rules, in N-D."""

import pytest
import torch

from quant.research.grid import StateGrid
from quant.numerics import REFERENCE

P = REFERENCE
FREE = lambda x: torch.zeros_like(x)  # noqa: E731 -- issuance at no cost


def grid3():
    return StateGrid.uniform([(-0.5, 1.0, 7), (0.0, 0.1, 5), (0.2, 1.0, 9)], P)


def affine(g, a=(1.0, 2.0, -3.0), c=0.7):
    w, s, A = g.states()
    return (a[0] * w + a[1] * s + a[2] * A + c).reshape(g.shape)


def test_states_are_row_major():
    g = grid3()
    w, s, A = g.states()
    assert w.shape == (7 * 5 * 9,)
    assert float(A[1] - A[0]) == pytest.approx(g.step(2))
    assert float(w[5 * 9] - w[0]) == pytest.approx(g.step(0))


def test_multilinear_is_exact_for_an_affine_function():
    """Inside the grid an affine V must be reproduced exactly, in every axis."""
    g = grid3()
    V = affine(g, c=10.0)  # positive throughout, so the walk-away floor is inert
    pts = [P.tensor([0.13, -0.41, 0.99]), P.tensor([0.07, 0.01, 0.1]), P.tensor([0.33, 0.95, 0.2])]
    got = g.interpolate(V, pts, FREE)
    want = 1.0 * pts[0] + 2.0 * pts[1] - 3.0 * pts[2] + 10.0
    assert torch.allclose(got, want, atol=1e-13)


def test_equity_off_grid_rules():
    """Above the top: slope one. Below the bottom: issue the gap at its cost."""
    g = grid3()
    V = affine(g, a=(1.0, 0.0, 0.0), c=2.0)  # V = w + 2
    s, A = P.tensor([0.05]), P.tensor([0.5])
    above = g.interpolate(V, [P.tensor([1.3]), s, A], FREE)
    assert float(above) == pytest.approx(1.0 + 2.0 + 0.3)
    cost = lambda x: 0.1 * x  # noqa: E731
    below = g.interpolate(V, [P.tensor([-0.8]), s, A], cost)
    assert float(below) == pytest.approx((-0.5 + 2.0) - 0.3 - 0.03)


def test_other_axes_are_clamped_and_value_is_floored_at_zero():
    g = grid3()
    V = affine(g, a=(1.0, 0.0, -3.0), c=0.0)
    w = P.tensor([0.4])
    clamped = g.interpolate(V, [w, P.tensor([0.05]), P.tensor([5.0])], FREE)   # A beyond its top
    at_top = g.interpolate(V, [w, P.tensor([0.05]), P.tensor([1.0])], FREE)
    assert float(clamped) == pytest.approx(float(at_top))
    assert float(at_top) == 0.0  # 0.4 - 3.0 < 0, floored by walk-away
