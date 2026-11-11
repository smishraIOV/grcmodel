"""Numerics discipline: every tensor is built with an explicit dtype."""

import ast
from pathlib import Path

import pytest
import torch

from quant.numerics import CPU_FAST, REFERENCE, get_profile

QUANT = Path(__file__).resolve().parent.parent / "quant"
CONSTRUCTORS = {"tensor", "zeros", "ones", "full", "empty", "arange", "linspace", "rand", "randn"}


def test_reference_is_float64_on_cpu():
    assert REFERENCE.dtype == torch.float64
    assert REFERENCE.tensor(1.0).dtype == torch.float64
    assert CPU_FAST.tensor(1.0).dtype == torch.float32


def test_unknown_profile_is_refused():
    with pytest.raises(ValueError):
        get_profile("nope")


def test_no_bare_tensor_construction():
    """A `torch.tensor(...)` without `dtype=` silently gives float32.

    Every tensor goes through a NumericsProfile, so outside numerics.py a torch
    constructor must name its dtype. `*_like` constructors inherit one and are
    fine.
    """
    offenders = []
    for path in QUANT.rglob("*.py"):
        if path.name == "numerics.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in CONSTRUCTORS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and not any(k.arg == "dtype" for k in node.keywords)
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders
