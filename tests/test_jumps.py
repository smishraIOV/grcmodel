"""v1: heavy-tailed jump losses (design.md section 2.4)."""

from dataclasses import replace

import pytest
import torch

from quant.model import Controls, jump_probability, outcomes, shock_correlation
from quant.numerics import REFERENCE
from quant.quadrature import pareto_jump_rule, student_t_rule
from quant.solver import solve
from tests.test_solver import BASE, SPEC

P = REFERENCE
JUMPY = replace(BASE, jump_rate=0.1)


def test_severity_rule_is_heavy_tailed_and_capped():
    rule = pareto_jump_rule(0.05, 2.5, P, n=8)
    assert float(rule.weights.sum()) == pytest.approx(1.0, abs=1e-14)
    assert float(rule.severity.max()) <= 1.0
    # The cap and the dropped far tail shave the mean a little, never much.
    assert rule.mean == pytest.approx(0.05, rel=0.05)
    # Heavy tail: the worst node is many times the mean.
    assert float(rule.severity.max()) > 10 * rule.mean


def test_grc_lowers_jump_frequency_to_its_floor():
    p = JUMPY
    none = float(jump_probability(p, P.tensor(0.0)))
    lots = float(jump_probability(p, P.tensor(10.0)))
    assert none == pytest.approx(1 - torch.exp(torch.tensor(-p.jump_rate)).item())
    assert lots == pytest.approx(1 - torch.exp(torch.tensor(-p.jump_rate * p.rho_J)).item())


def test_jump_probability_is_differentiable_in_spend():
    """The probability sits in the quadrature weights, so the spend gets a gradient."""
    p = JUMPY
    rule = student_t_rule(shock_correlation(p), p.nu, P, n_hermite=3, n_mixing=4)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P)
    g = P.tensor([0.005]).requires_grad_(True)
    c = Controls(K=P.tensor([0.08]), g=g, I=P.tensor([0.8]), alpha=P.tensor([0.0]), H=P.tensor([0.4]))
    out = outcomes(p, c, g, rule, jumps)
    expected_loss_free_wealth = out.expect(out.wealth).sum()
    (grad,) = torch.autograd.grad(expected_loss_free_wealth, g)
    assert float(grad) != 0.0


def test_outcome_weights_sum_to_one():
    p = JUMPY
    rule = student_t_rule(shock_correlation(p), p.nu, P, n_hermite=3, n_mixing=4)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P)
    c = Controls(K=P.tensor([0.08, 0.1]), g=P.tensor([0.0, 0.01]), I=P.tensor([0.8, 0.5]),
                 alpha=P.tensor([0.1, 0.0]), H=P.tensor([0.4, 0.25]))
    out = outcomes(p, c, c.g, rule, jumps)
    assert torch.allclose(out.weights.sum(-1), torch.ones(2, dtype=P.dtype), atol=1e-13)


def test_jumps_cannot_raise_the_bank_s_value():
    """Adding a source of loss can only make the bank worth less, everywhere."""
    calm = solve(BASE, P, SPEC)
    jumpy = solve(JUMPY, P, SPEC)
    assert jumpy.residual < 1e-6
    assert bool((jumpy.V <= calm.V + 1e-9).all())
