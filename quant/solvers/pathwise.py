"""Rung 1.5: the perfect-information bound.

This is the static model's `optimize_policy`, correctly renamed. It holds a
free investment parameter for every path, chosen with full knowledge of what
that path realized. At a one-period horizon that coincides with the optimum,
because all uncertainty resolves before the single decision -- which is why it
was the right solver for the static model and produced its published numbers.

At any longer horizon it is not a policy at all. It violates
non-anticipativity and returns a strict upper bound:

    V(any implementable policy)  <=  V*  <=  V_PI

That makes it more useful as a bound than it ever was as a solver. Any learner
reporting a value above V_PI has a bug -- an information leak, a mis-signed
discount, a double-counted reward, or a death that failed to absorb. The gap
V_PI - V(policy) is the expected value of perfect information, and if it is
near zero the problem has no interesting dynamics for a policy to exploit,
which is worth discovering early rather than after building a learner.

The cheat is deliberate and lives in the policy's type: PerPathPolicy holds a
parameter indexed by path, so it can only be constructed by something that
already knows the batch. An honest policy takes a state and returns an action.
"""

import torch

from quant.env.actions import ActionSpec, FirmAction
from quant.env.env import EvalResult, FirmEnv, evaluate
from quant.env.shocks import CommonRandomNumbers
from quant.env.state import N_FAMILIES, FirmState


class PerPathPolicy:
    """Budgets shared across paths, investment free per path and per period.

    Budgets are shared because they are committed before the period's shock is
    seen; investment is per-path because that is precisely the clairvoyance
    being bounded. Both are time-indexed so the bound extends to a horizon.
    """

    def __init__(self, batch: int, horizon: int, profile):
        self.raw_budgets = profile.zeros(horizon, N_FAMILIES, requires_grad=True)
        self.raw_investment = profile.zeros(horizon, batch, requires_grad=True)

    def parameters(self) -> list[torch.Tensor]:
        return [self.raw_budgets, self.raw_investment]

    def __call__(self, state: FirmState) -> FirmAction:
        batch = state.batch()
        raw = torch.cat(
            [
                self.raw_budgets[state.t].expand(batch, N_FAMILIES),
                self.raw_investment[state.t].unsqueeze(-1),
            ],
            dim=-1,
        )
        return ActionSpec.from_raw(raw)


def perfect_information_bound(
    env: FirmEnv,
    crn: CommonRandomNumbers,
    n_steps: int = 4000,
    lr: float = 0.05,
) -> tuple[PerPathPolicy, EvalResult]:
    """Maximize firm value by gradient ascent through the differentiable rollout.

    Pathwise gradients, not a score-function estimator: the simulator is
    differentiable, so the gradient is exact rather than sampled, which is what
    lets the frictionless case land on its closed form to ~1e-15.
    """
    policy = PerPathPolicy(crn.batch, env.config.horizon, env.config.profile)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        (-env.value_of(env.rollout(policy, crn, differentiable=True))).backward()
        optimizer.step()

    return policy, evaluate(policy, env, crn)
