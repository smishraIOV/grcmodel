"""Monte Carlo scaffold (docs/quant-model.md section 5.2).

Samples batched risk shocks for the families named in docs/framework.md
section 1 and feeds them into the same FirmValueModel / optimize_policy
harness quant/static.py uses with a single two-state shock. The
distributions below are placeholder-simple (not calibrated to anything
real) -- the deliverable this pass is the architecture that lets a
Monte Carlo policy search reuse the static model's code unchanged, not
realistic risk parameters.
"""

from dataclasses import dataclass

import torch

from quant.device import get_device
from quant.model import FirmValueModel, RiskDraw, optimize_policy


@dataclass
class RiskSamplerConfig:
    """Placeholder distribution parameters for each risk family. Credit and
    market risk are modeled as a continuous loss; tech-infra-driven
    incidents and compliance breaches are modeled as rare (Bernoulli-gated)
    but severe when they occur, per docs/framework.md section 1's
    traditional-vs-crypto-native split.
    """

    credit_loss_mean: float = 4.0
    op_incident_probability: float = 0.15
    op_incident_severity_mean: float = 20.0
    compliance_breach_probability: float = 0.05
    compliance_breach_severity_mean: float = 15.0


def sample_risk_draw(n_paths: int, config: RiskSamplerConfig, device: torch.device) -> RiskDraw:
    credit_loss = torch.distributions.Exponential(1.0 / config.credit_loss_mean).sample((n_paths,)).to(device)

    op_incident_occurs = torch.bernoulli(torch.full((n_paths,), config.op_incident_probability)).to(device)
    op_incident_severity = torch.distributions.Exponential(1.0 / config.op_incident_severity_mean).sample(
        (n_paths,)
    ).to(device)
    op_incident_loss = op_incident_occurs * op_incident_severity

    compliance_breach_occurs = torch.bernoulli(torch.full((n_paths,), config.compliance_breach_probability)).to(
        device
    )
    compliance_breach_severity = torch.distributions.Exponential(
        1.0 / config.compliance_breach_severity_mean
    ).sample((n_paths,)).to(device)
    compliance_breach_loss = compliance_breach_occurs * compliance_breach_severity

    return RiskDraw(
        credit_loss=credit_loss,
        op_incident_loss=op_incident_loss,
        compliance_breach_loss=compliance_breach_loss,
    )


def run_policy_search(
    model: FirmValueModel,
    n_paths: int = 4096,
    config: RiskSamplerConfig | None = None,
    seed: int | None = 0,
) -> tuple[float, float]:
    """Same optimize_policy harness quant/static.py uses, but with a large
    sampled batch of risk draws instead of a fixed two-state shock."""
    device = get_device()
    if seed is not None:
        torch.manual_seed(seed)
    risk_draw = sample_risk_draw(n_paths, config or RiskSamplerConfig(), device)
    return optimize_policy(model, risk_draw, device=device)


if __name__ == "__main__":
    device = get_device()
    print(f"device: {device}")
    model = FirmValueModel(initial_equity=3.0, financing_convexity=2.0, grc_alpha=0.3)
    g_star, value_star = run_policy_search(model)
    print(f"optimal g: {g_star:.4f}")
    print(f"expected firm value: {value_star:.4f}")
