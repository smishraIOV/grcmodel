"""Cost, payoff and mitigation functions from docs/quant-model.md section 3.

All functions are torch-differentiable so they can sit inside an autograd
optimization loop (quant/model.py's optimize_policy).
"""

import torch


def financing_cost(
    shortfall: torch.Tensor,
    convexity: float,
    reference: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """Convex cost of raising external finance to cover a shortfall.

    `reference` (K in the docs) carries money units, so `shortfall / reference`
    is dimensionless and the result is money. The earlier form -- a bare
    `shortfall ** convexity` -- subtracted money^convexity from money, which
    made the model's answer depend on whether the firm was denominated in
    dollars or cents (docs/critical-review.md F1). With K present, `convexity`
    is a pure shape parameter.

    `shortfall` must already be clamped to >= 0. `convexity` must be >= 1;
    below 1 the derivative at shortfall = 0 is unbounded.
    """
    return scale * reference * (shortfall / reference) ** convexity


def production(investment: torch.Tensor, scale: float, curvature: float) -> torch.Tensor:
    """Concave payoff from deploying capital: F(I) = A*S*(1 - exp(-I/S)).

    F'(0) = A, so investing is worthwhile at the margin when A > 1, and the
    unconstrained optimum is I* = S*ln(A). This is the channel that makes risk
    management value-adding rather than merely loss-avoiding: when a bad draw
    leaves internal wealth below I*, the firm must either underinvest or pay
    the convex financing premium above.

    Written with `expm1` rather than as `1 - exp(-x)`, and that is not a
    micro-optimization. The whole payoff is the *difference* between F(I) and I,
    which at the calibrated curvature is about 1.7 out of 80 -- so the quantity
    that matters is a two-percent residue of two numbers that nearly cancel.
    `1 - exp(-x)` for small x computes that residue by subtracting two float32
    numbers close to 1 and throws away most of its significant digits;
    `-expm1(-x)` computes it directly.

    Caught when the curvature rose from 600 to 2792 for the levered balance
    sheet: `tests/test_numerics.py::test_profiles_agree` went from a float32
    versus float64 disagreement of 2e-7 to 1.5e-5, straight through a tolerance
    the documentation quotes budgets to. The bug was always there -- a smaller
    curvature simply kept x large enough to hide it.
    """
    return scale * curvature * (-torch.expm1(-investment / curvature))


def exponential_mitigation(quantity: torch.Tensor, g: torch.Tensor, alpha: float) -> torch.Tensor:
    """Shrink `quantity` by exp(-alpha * g) for GRC spend `g`.

    Diminishing returns: each additional unit of spend removes less than the
    last. Which moment of which risk family this is applied to -- a mean, a
    severity, or an occurrence probability -- is quant/model.py's decision, not
    this function's.
    """
    return quantity * torch.exp(-alpha * g)
