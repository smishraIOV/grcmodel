"""Rung 1: the closed forms, which are exact and therefore the ground truth.

Moved from quant/model.py unchanged in substance -- it now reads a Shock
rather than a RiskDraw. This is the only thing in the project that knows the
right answer without solving anything, so every approximate solver is
ultimately measured against it.

The benchmark survives the move to a dynamic model. It is reached today by
switching the financing friction off; in the dynamic model it is reached by
setting the discount factor to zero, which removes the continuation value and
leaves exactly this one-period problem. Same identity, different degeneration.
"""

import math
from dataclasses import dataclass

import torch

from quant.env.shocks import Shock
from quant.params import GrcAlphas

FAMILIES = ("credit", "operational", "compliance")


@dataclass(frozen=True)
class FrictionlessBenchmark:
    """Closed-form optimal GRC budgets when external finance is free."""

    credit: float
    operational: float
    compliance: float

    def total(self) -> float:
        return self.credit + self.operational + self.compliance


def family_exposures(shock: Shock, book: float = 1.0) -> dict[str, float]:
    """Base-probability-weighted expected loss for each risk family.

    `book` converts the credit channel from a rate into money. Where credit loss
    scales with what the firm lends, `shock.credit_loss` is a loss rate per unit
    deployed, and the exposure a budget is traded against is that rate times the
    book. The default of 1.0 leaves the oracle's four-state world untouched,
    where credit loss is an exogenous money amount.

    Getting this wrong is quiet. With the rate read as money, credit's
    risk-neutral break-even 1/X came out at 118.3 instead of 5.0 -- off by
    exactly the size of the book -- and simply looked like a family nothing
    could justify protecting.
    """
    weight = shock.base_weight / shock.base_weight.sum()
    return {
        "credit": book * (weight * shock.credit_loss).sum().item(),
        "operational": (weight * shock.op_occurs * shock.op_severity).sum().item(),
        "compliance": (
            weight * shock.compliance_occurs * shock.compliance_severity
        ).sum().item(),
    }


def frictionless_benchmark(shock: Shock, alphas: GrcAlphas) -> FrictionlessBenchmark:
    """The budgets a risk-neutral firm would choose with no financing friction.

    With the friction off the firm's value is linear in wealth, investment is
    unconstrained, and the objective separates across risk families into
    min_g [ g + exposure * exp(-alpha * g) ]. That has the closed-form solution
    g* = max(0, ln(alpha * exposure) / alpha) -- zero whenever a unit of spend
    buys back less than a unit of expected loss.

    The compliance case works out the same way as the others because the
    likelihood-ratio weights integrate to one, so E[compliance loss](g_k)
    reduces to exposure * exp(-alpha_k * g_k).
    """
    exposures = family_exposures(shock)

    def optimal(family: str) -> float:
        alpha = getattr(alphas, family)
        marginal_return = alpha * exposures[family]
        return math.log(marginal_return) / alpha if marginal_return > 1.0 else 0.0

    return FrictionlessBenchmark(*(optimal(family) for family in FAMILIES))


def unconstrained_investment(production_scale: float, production_curvature: float) -> float:
    """I* = S*ln(A): what the firm invests with no financing friction."""
    return production_curvature * math.log(production_scale)
