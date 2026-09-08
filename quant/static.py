"""Static two-period Froot-Stein model (docs/quant-model.md section 5.1).

Collapses state to initial equity plus one fixed representative risk shock,
and sweeps the financing-cost convexity to reproduce, numerically via
autograd, the qualitative Froot-Stein result: optimal GRC investment rises
as the convexity of external-financing costs rises.
"""

import torch

from quant.device import get_device
from quant.model import FirmValueModel, RiskDraw, optimize_policy

# A two-state representative shock (good state / bad state), split across
# the risk families named in docs/framework.md section 1 (credit /
# operational / compliance). This is still a "static" one-shot decision, but
# needs at least this much genuine uncertainty: the Froot-Stein result is a
# Jensen's-inequality effect (a convex cost function makes reducing the
# *spread* of a random loss valuable, not just its mean) -- with a single
# deterministic shock there is no spread for convexity to act on, and the
# optimal g collapses to the same value regardless of convexity. Averaging
# over these two states reuses FirmValueModel.compute_value's batch-mean
# path, the same one quant/simulate.py's Monte Carlo draws will use.
REPRESENTATIVE_SHOCK = RiskDraw(
    credit_loss=torch.tensor([2.0, 6.0]),
    op_incident_loss=torch.tensor([1.0, 5.0]),
    compliance_breach_loss=torch.tensor([0.0, 2.0]),
)

INITIAL_EQUITY = 3.0
GRC_ALPHA = 0.3
CONVEXITY_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def run_convexity_sweep() -> list[tuple[float, float, float]]:
    device = get_device()
    results = []
    for convexity in CONVEXITY_SWEEP:
        model = FirmValueModel(
            initial_equity=INITIAL_EQUITY,
            financing_convexity=convexity,
            grc_alpha=GRC_ALPHA,
        )
        g_star, value_star = optimize_policy(model, REPRESENTATIVE_SHOCK, device=device)
        results.append((convexity, g_star, value_star))
    return results


if __name__ == "__main__":
    device = get_device()
    print(f"device: {device}")
    print(f"{'convexity':>10} | {'optimal g':>10} | {'firm value':>10}")
    for convexity, g_star, value_star in run_convexity_sweep():
        print(f"{convexity:>10.2f} | {g_star:>10.4f} | {value_star:>10.4f}")
