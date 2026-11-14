"""Primitives every business form shares, and the jump rule (design.md
sections 3 and 4)."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from quant.core import jump_probability, mitigation, next_grc, shareholder_flow
from quant.numerics import REFERENCE
from quant.params import DEFAULTS
from quant.quadrature import pareto_jump_rule

P = REFERENCE
ROOT = Path(__file__).resolve().parent.parent


def t(x):
    return P.tensor(x)


def test_dividend_and_issuance_flows():
    p = DEFAULTS
    assert float(shareholder_flow(p, t(0.3), t(0.1))) == pytest.approx(0.2)
    e = 0.1
    cost = p.lam * e + 0.5 * p.kappa * e * e
    assert float(shareholder_flow(p, t(0.1), t(0.2))) == pytest.approx(-e - cost)


def test_mitigation_has_an_irreducible_floor():
    assert float(mitigation(t(0.0), 50.0, 0.3)) == pytest.approx(1.0)
    assert float(mitigation(t(10.0), 50.0, 0.3)) == pytest.approx(0.3)


def test_grc_stock_depreciates():
    p = DEFAULTS
    assert float(next_grc(p, t(0.04), t(0.01))) == pytest.approx(0.04 * (1 - p.delta) + 0.01)


def test_severity_rule_is_heavy_tailed_and_capped():
    rule = pareto_jump_rule(0.05, 2.5, P, n=8)
    assert float(rule.weights.sum()) == pytest.approx(1.0, abs=1e-14)
    assert float(rule.severity.max()) <= 1.0
    # The cap and the dropped far tail shave the mean a little, never much.
    assert rule.mean == pytest.approx(0.05, rel=0.05)
    # Heavy tail: the worst node is many times the mean.
    assert float(rule.severity.max()) > 10 * rule.mean


def test_grc_lowers_jump_frequency_to_its_floor():
    p = DEFAULTS
    none = float(jump_probability(p, t(0.0)))
    lots = float(jump_probability(p, t(10.0)))
    assert none == pytest.approx(1 - math.exp(-p.jump_rate))
    assert lots == pytest.approx(1 - math.exp(-p.jump_rate * p.rho_J))


def test_jump_probability_is_differentiable_in_spend():
    S = t([0.005]).requires_grad_(True)
    (grad,) = torch.autograd.grad(jump_probability(DEFAULTS, S).sum(), S)
    assert float(grad) < 0.0


def test_every_entry_point_imports():
    """A script nothing imports can be broken for days. The research scripts
    have the same check in tests/research/test_layout.py."""
    for path in sorted((ROOT / "scripts").glob("*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "main"), path.name
