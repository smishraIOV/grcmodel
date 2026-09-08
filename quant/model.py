"""Shared firm-value model and optimizer loop.

docs/quant-model.md section 5 stages a static model (5.1) and a Monte Carlo
simulation (5.2) on top of it. Both share this module: quant/static.py calls
FirmValueModel with a single-shock RiskDraw, quant/simulate.py calls it with
a batched RiskDraw and relies on compute_value's mean-over-batch to turn
that into a scalar objective. optimize_policy is the one autograd loop both
use.
"""

from dataclasses import dataclass

import torch

from quant.device import get_device
from quant.frictions import financing_cost, grc_investment_cost, grc_mitigation


@dataclass
class RiskDraw:
    """A realization (or batch of realizations) of the risk shocks named in
    docs/framework.md section 1. Simplification for this first model: all
    three loss categories are mitigated by GRC investment via the same
    factor (grc_mitigation) — a real model would give each risk family its
    own sensitivity to GRC spend.
    """

    credit_loss: torch.Tensor
    op_incident_loss: torch.Tensor
    compliance_breach_loss: torch.Tensor

    def total_loss(self) -> torch.Tensor:
        return self.credit_loss + self.op_incident_loss + self.compliance_breach_loss


@dataclass
class FirmValueModel:
    """Computes net firm value per docs/quant-model.md section 4:
    expected PV of net cash flows minus expected distress/friction costs,
    here reduced to: initial equity, minus GRC spend, minus the
    GRC-mitigated risk-shock loss, minus convex financing cost on whatever
    shortfall remains.
    """

    initial_equity: float
    financing_convexity: float
    grc_alpha: float
    financing_scale: float = 1.0

    def compute_value(self, g: torch.Tensor, risk_draw: RiskDraw) -> torch.Tensor:
        mitigation = grc_mitigation(g, alpha=self.grc_alpha)
        mitigated_loss = risk_draw.total_loss() * (1.0 - mitigation)
        net_position = self.initial_equity - grc_investment_cost(g) - mitigated_loss
        shortfall = torch.clamp(-net_position, min=0.0)
        cost = financing_cost(shortfall, convexity=torch.tensor(self.financing_convexity), scale=self.financing_scale)
        value = net_position - cost
        return value.mean() if value.dim() > 0 else value


def optimize_policy(
    model: FirmValueModel,
    risk_draw: RiskDraw,
    n_steps: int = 500,
    lr: float = 0.05,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """Find the GRC investment level g that maximizes firm value, via
    gradient ascent (implemented as minimizing -value) with Adam.

    Returns (optimal_g, optimal_value) as plain floats.
    """
    device = device or get_device()
    risk_draw = RiskDraw(
        credit_loss=risk_draw.credit_loss.to(device),
        op_incident_loss=risk_draw.op_incident_loss.to(device),
        compliance_breach_loss=risk_draw.compliance_breach_loss.to(device),
    )
    # g must stay >= 0. A hard torch.clamp(g, min=0.0) has zero gradient
    # exactly at the boundary, which deadlocks gradient ascent when g starts
    # at 0 (the natural starting point). Reparameterize through softplus
    # instead, which is smooth and strictly positive everywhere.
    raw = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        g = torch.nn.functional.softplus(raw)
        value = model.compute_value(g, risk_draw)
        loss = -value
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        g_final = torch.nn.functional.softplus(raw)
        value_final = model.compute_value(g_final, risk_draw)

    return g_final.item(), value_final.item()
