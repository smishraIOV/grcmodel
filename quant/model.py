"""Shared firm-value model and optimizer loop.

Implements a two-stage problem, Froot Stein.

    t0   choose GRC budgets g_c, g_o, g_k (credit, operational, compliance)
    t1   observe the risk draw -> internal wealth w = E - sum(g) - L(g) , see wealth()
         choose investment I, funding any excess over w externally at a
         convex premium
    value = E[ w - I + F(I) - P(max(0, I - w)) ]

The investment term F(I) is what makes this Froot-Stein rather than a bare
convex penalty on losses. Without an investment opportunity whose funding
depends on internal wealth, risk management is only loss-avoidance and cannot
add value.

* quant/static.py supplies a small explicitly-weighted state space, 
* quant/simulate.py supplies a large sampled batch. 
* optimize_policy is the one autograd loop both use.
"""

import math
from dataclasses import dataclass

import torch

from quant.frictions import exponential_mitigation, financing_cost, production
from quant.numerics import DEFAULT_PROFILE, NumericsProfile
from quant.params import GrcAlphas

__all__ = [
    "GrcAlphas",
    "GrcBudgets",
    "RiskDraw",
    "FrictionlessBenchmark",
    "PolicyResult",
    "FirmValueModel",
    "family_exposures",
    "frictionless_benchmark",
    "optimize_policy",
]


@dataclass
class GrcBudgets:
    """GRC spend split across the risk families in docs/framework.md section 1."""

    credit: torch.Tensor
    operational: torch.Tensor
    compliance: torch.Tensor

    @classmethod
    def from_vector(cls, vector: torch.Tensor) -> "GrcBudgets":
        return cls(credit=vector[0], operational=vector[1], compliance=vector[2])

    def total(self) -> torch.Tensor:
        return self.credit + self.operational + self.compliance


@dataclass
class RiskDraw:
    """A probability-weighted batch of risk realizations, one entry per path.

    Each family is represented by the moment its GRC budget actually moves
    (docs/framework.md section 1):

    - credit      a continuous loss; GRC reduces its MEAN
    - operational an occurrence indicator times a severity; GRC reduces the
                  SEVERITY (better controls contain an incident, they do not
                  stop it happening)
    - compliance  an occurrence indicator times a severity; GRC reduces the
                  OCCURRENCE PROBABILITY (a compliance programme prevents
                  breaches; it does not soften the penalty once one lands)

    `base_weight` is the probability of each path before GRC shifts anything.
    quant/static.py sets it explicitly; quant/simulate.py leaves it uniform.
    """

    credit_loss: torch.Tensor
    op_occurs: torch.Tensor
    op_severity: torch.Tensor
    compliance_occurs: torch.Tensor
    compliance_severity: torch.Tensor
    base_weight: torch.Tensor
    compliance_base_probability: float

    def to(self, profile: NumericsProfile) -> "RiskDraw":
        def move(tensor: torch.Tensor) -> torch.Tensor:
            return profile.to(tensor)

        return RiskDraw(
            credit_loss=move(self.credit_loss),
            op_occurs=move(self.op_occurs),
            op_severity=move(self.op_severity),
            compliance_occurs=move(self.compliance_occurs),
            compliance_severity=move(self.compliance_severity),
            base_weight=move(self.base_weight),
            compliance_base_probability=self.compliance_base_probability,
        )

    def n_paths(self) -> int:
        return self.credit_loss.shape[0]

    def mitigated_loss(self, budgets: GrcBudgets, alphas: GrcAlphas) -> torch.Tensor:
        credit = exponential_mitigation(self.credit_loss, budgets.credit, alphas.credit)
        operational = self.op_occurs * exponential_mitigation(
            self.op_severity, budgets.operational, alphas.operational
        )
        # Compliance loss is untouched here: its budget acts through
        # path_weights, by making a breach rarer rather than cheaper.
        compliance = self.compliance_occurs * self.compliance_severity
        return credit + operational + compliance

    def path_weights(self, budgets: GrcBudgets, alphas: GrcAlphas) -> torch.Tensor:
        """Path probabilities after the compliance budget has shifted the
        breach probability.

        A breach is a discrete event, so the compliance budget cannot act on it
        by shrinking a sampled indicator -- torch.bernoulli is not
        differentiable in p. Instead the indicators are drawn once at
        `compliance_base_probability` and each path is reweighted by the
        likelihood ratio: p(g)/p0 on breach paths, (1-p(g))/(1-p0) on the rest.
        That is unbiased, differentiable in g, and keeps breaches discrete --
        which matters, because it is the spread they create that the convex
        financing premium acts on.
        """
        p0 = self.compliance_base_probability
        p = exponential_mitigation(
            torch.as_tensor(p0, device=self.credit_loss.device, dtype=self.credit_loss.dtype),
            budgets.compliance,
            alphas.compliance,
        )
        ratio = torch.where(self.compliance_occurs > 0.5, p / p0, (1.0 - p) / (1.0 - p0))
        weights = self.base_weight * ratio
        return weights / weights.sum()


@dataclass
class FrictionlessBenchmark:
    """Closed-form optimal GRC budgets when external finance is free."""

    credit: float
    operational: float
    compliance: float

    def total(self) -> float:
        return self.credit + self.operational + self.compliance


def family_exposures(draw: RiskDraw) -> dict[str, float]:
    """Base-probability-weighted expected loss for each risk family."""
    weight = draw.base_weight / draw.base_weight.sum()
    return {
        "credit": (weight * draw.credit_loss).sum().item(),
        "operational": (weight * draw.op_occurs * draw.op_severity).sum().item(),
        "compliance": (weight * draw.compliance_occurs * draw.compliance_severity).sum().item(),
    }


def frictionless_benchmark(draw: RiskDraw, alphas: GrcAlphas) -> FrictionlessBenchmark:
    """The budgets a risk-neutral firm would choose with no financing friction.

    With financing_scale = 0 the firm's value is linear in wealth, investment
    is unconstrained, and the objective separates across risk families into
    min_g [ g + exposure * exp(-alpha * g) ]. That has the closed-form solution
    g* = max(0, ln(alpha * exposure) / alpha) -- zero whenever a unit of spend
    buys back less than a unit of expected loss.

    Each family's exposure is its base-probability-weighted expected loss. The
    compliance case works out the same way as the others because the
    likelihood-ratio weights integrate to one, so E[compliance loss](g_k)
    reduces to exposure * exp(-alpha_k * g_k).

    This is the reference the zero-friction control test asserts against, and
    the baseline against which the Froot-Stein premium is measured.
    """
    exposures = family_exposures(draw)

    def optimal(family: str) -> float:
        alpha = getattr(alphas, family)
        marginal_return = alpha * exposures[family]
        return math.log(marginal_return) / alpha if marginal_return > 1.0 else 0.0

    return FrictionlessBenchmark(
        credit=optimal("credit"),
        operational=optimal("operational"),
        compliance=optimal("compliance"),
    )


@dataclass
class PolicyResult:
    """Outcome of one optimize_policy run."""

    credit: float
    operational: float
    compliance: float
    value: float
    constrained_fraction: float

    @property
    def total(self) -> float:
        return self.credit + self.operational + self.compliance


@dataclass
class FirmValueModel:
    """Firm value per docs/quant-model.md section 4: expected present value of
    net cash flows, net of financing frictions and GRC spend.
    """

    initial_equity: float
    financing_convexity: float
    distress_reference: float
    production_scale: float
    production_curvature: float
    alphas: GrcAlphas
    financing_scale: float = 1.0

    def wealth(self, budgets: GrcBudgets, draw: RiskDraw) -> torch.Tensor:
        # w = E - sum(g) - L(g)
        return self.initial_equity - budgets.total() - draw.mitigated_loss(budgets, self.alphas)

    def compute_value(
        self, budgets: GrcBudgets, investment: torch.Tensor, draw: RiskDraw
    ) -> torch.Tensor:
        wealth = self.wealth(budgets, draw)
        external = torch.clamp(investment - wealth, min=0.0)
        premium = financing_cost(
            external,
            convexity=self.financing_convexity,
            reference=self.distress_reference,
            scale=self.financing_scale,
        )
        payoff = (
            wealth
            - investment
            + production(investment, self.production_scale, self.production_curvature)
            - premium
        )
        return (payoff * draw.path_weights(budgets, self.alphas)).sum()

    def unconstrained_investment(self) -> float:
        """I* = S*ln(A): what the firm would invest with no financing friction."""
        return self.production_curvature * math.log(self.production_scale)


def optimize_policy(
    model: FirmValueModel,
    draw: RiskDraw,
    n_steps: int = 4000,
    lr: float = 0.05,
    profile: NumericsProfile | None = None,
) -> PolicyResult:
    """Jointly choose the three GRC budgets and the per-path investment level
    that maximize firm value, by gradient ascent with Adam.

    Investment is a vector, one entry per path: it is chosen at t1, after the
    risk draw is observed, so it is state-contingent by construction.

    Both controls are reparameterized through softplus rather than clamped.
    torch.clamp's gradient is zero exactly at the boundary, so a control
    initialized at 0 -- the natural starting point for a spend level -- would
    never receive a signal to move away from it.
    """
    profile = profile or DEFAULT_PROFILE
    draw = draw.to(profile)

    raw_budgets = profile.zeros(3, requires_grad=True)
    raw_investment = profile.zeros(draw.n_paths(), requires_grad=True)
    optimizer = torch.optim.Adam([raw_budgets, raw_investment], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        budgets = GrcBudgets.from_vector(torch.nn.functional.softplus(raw_budgets))
        investment = torch.nn.functional.softplus(raw_investment)
        (-model.compute_value(budgets, investment, draw)).backward()
        optimizer.step()

    with torch.no_grad():
        budgets = GrcBudgets.from_vector(torch.nn.functional.softplus(raw_budgets))
        investment = torch.nn.functional.softplus(raw_investment)
        value = model.compute_value(budgets, investment, draw)
        external = torch.clamp(investment - model.wealth(budgets, draw), min=0.0)
        weights = draw.path_weights(budgets, model.alphas)
        # A money-valued threshold, not a bare 1e-9: `external` is softplus
        # output so it is never exactly zero, and a fixed absolute epsilon is
        # float64-scaled -- under float32 it sits at the noise floor and every
        # path reads as constrained. Scaling the reference level by sqrt(eps)
        # keeps both the units and the meaning across profiles.
        negligible = model.distress_reference * torch.finfo(profile.dtype).eps ** 0.5
        constrained = profile.sum((external > negligible).to(profile.dtype) * weights)

    return PolicyResult(
        credit=budgets.credit.item(),
        operational=budgets.operational.item(),
        compliance=budgets.compliance.item(),
        value=value.item(),
        constrained_fraction=constrained.item(),
    )
