"""Cost/mitigation functions from docs/quant-model.md section 3.

All functions are torch-differentiable so they can sit inside an
autograd optimization loop (quant/model.py's optimize_policy).
"""

import torch


def financing_cost(shortfall: torch.Tensor, convexity: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Convex cost of raising external financing to cover a capital shortfall.

    The Froot-Stein term: cost rises faster than linearly in the shortfall
    as `convexity` grows past 1.0. `shortfall` should already be clamped to
    >= 0 (no cost when there's no shortfall).
    """
    return scale * shortfall**convexity


def grc_mitigation(g: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Fraction of risk-shock losses avoided by GRC investment level `g`.

    Diminishing returns: mitigation -> 1 as g grows, but each extra dollar
    of `g` buys less additional mitigation than the last (per
    docs/quant-model.md section 3's GRC-investment-cost note).
    """
    return 1.0 - torch.exp(-alpha * g)


def grc_investment_cost(g: torch.Tensor) -> torch.Tensor:
    """Direct cost of GRC spend itself. Linear (dollar-for-dollar spend);
    the diminishing returns live in grc_mitigation's effect, not this cost.
    """
    return g
