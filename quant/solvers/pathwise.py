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

from quant.env.actions import N_RAW, ActionSpec, FirmAction
from quant.env.env import EvalResult, FirmEnv, evaluate
from quant.env.shocks import CommonRandomNumbers
from quant.env.state import N_FAMILIES, FirmState


class PerPathPolicy:
    """A free control per path and per period. Clairvoyant by construction.

    `shared_budgets` decides which of two different objects this is, and the
    distinction matters more than it looks:

    - **False (default): a genuine upper bound.** Every control is free per
      path, so this is the wait-and-see solution -- decide with the whole
      future known. It contains every implementable policy, including a
      state-feedback one, because whatever a policy computes from the state
      this can simply choose.

    - **True: the static model's decision timing.** GRC budgets are committed
      before the quarter's shock is seen and so cannot vary across paths,
      while investment is chosen after. That is exactly what the two-period
      model did, and it is what reproduces its published numbers.

    Getting this wrong is subtle and was a real bug here: with shared budgets
    at a horizon greater than one, paths have diverged by the second quarter,
    so a state-feedback policy can legitimately set a *different* budget on
    each path and beat the "bound". It is only a bound at one period, where
    every path starts identical and path-dependence buys nothing.
    """

    def __init__(self, batch: int, horizon: int, profile, shared_budgets: bool = False):
        self.shared_budgets = shared_budgets
        self.raw_budgets = profile.zeros(
            *( (horizon, N_FAMILIES) if shared_budgets else (horizon, batch, N_FAMILIES) ),
            requires_grad=True,
        )
        self.raw_investment = profile.zeros(horizon, batch, requires_grad=True)

    def parameters(self) -> list[torch.Tensor]:
        return [self.raw_budgets, self.raw_investment]

    def fill_from_constant(self, raw: torch.Tensor, horizon: int, batch: int) -> None:
        """Place a constant policy's parameters into this parameterization."""
        budgets = raw[:N_FAMILIES]
        shape = (horizon, N_FAMILIES) if self.shared_budgets else (horizon, batch, N_FAMILIES)
        with torch.no_grad():
            self.raw_budgets.copy_(budgets.expand(shape))
            self.raw_investment.copy_(raw[N_FAMILIES].expand(horizon, batch))

    def __call__(self, state: FirmState) -> FirmAction:
        batch = state.batch()
        budgets = self.raw_budgets[state.t]
        if self.shared_budgets:
            budgets = budgets.expand(batch, N_FAMILIES)
        raw = torch.cat([budgets, self.raw_investment[state.t].unsqueeze(-1)], dim=-1)
        return ActionSpec.from_raw(raw)


class RawConstantPolicy:
    """The same action every period, with the action learned.

    The smallest policy class worth optimizing, and the right floor to measure
    a learner against: a network that cannot beat a tuned constant has learned
    nothing about the state, only about the average. It is also the exact
    optimum whenever the observation carries no information -- at a one-period
    horizon every path starts identically, so a state-feedback policy can do no
    better than this and matching it is a convergence check rather than a
    result.
    """

    def __init__(self, profile):
        self.raw = profile.zeros(N_RAW, requires_grad=True)

    def parameters(self) -> list[torch.Tensor]:
        return [self.raw]

    def __call__(self, state: FirmState) -> FirmAction:
        return ActionSpec.from_raw(self.raw.expand(state.batch(), N_RAW))


def optimize_constant(
    env: FirmEnv, crn: CommonRandomNumbers, n_steps: int = 6000, lr: float = 0.05
) -> tuple[RawConstantPolicy, EvalResult]:
    policy = RawConstantPolicy(env.config.profile)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        (-env.value_of(env.rollout(policy, crn, differentiable=True))).backward()
        optimizer.step()
    return policy, evaluate(policy, env, crn)


def perfect_information_bound(
    env: FirmEnv,
    crn: CommonRandomNumbers,
    n_steps: int = 4000,
    lr: float = 0.05,
    warm_start: bool | None = None,
    shared_budgets: bool = False,
) -> tuple[PerPathPolicy, EvalResult]:
    """Maximize firm value by gradient ascent through the differentiable rollout.

    Pathwise gradients, not a score-function estimator: the simulator is
    differentiable, so the gradient is exact rather than sampled, which is what
    lets the frictionless case land on its closed form to ~1e-15.

    **Warm-started from the best constant policy, and it has to be.** Once a
    survival barrier is switched on, a cold start is unreliable in a way that
    is specific and worth understanding: limited liability means a failed
    firm's equity holders get zero, so a dead path's contribution is a
    constant and its gradient with respect to everything is exactly zero. A
    path that dies early therefore carries no signal about how to have avoided
    dying. From a cold start most paths die in the first quarter or two, the
    optimizer sits in that gradient desert, and it converges to something a
    plain constant policy beats -- measured on the eight-quarter model, 14.4
    against 25.4, which is not a bound at all.

    Warm-starting fixes it because the constant policy is a *member* of this
    policy class: PerPathPolicy can represent it exactly (asserted in
    tests/test_env.py), so starting there guarantees the result is at least as
    good and restores the property a bound has to have. Same run warm-started
    reaches 33.0.

    That the desert exists is the more important finding. It is the first
    place in this project where differentiating the simulator stops working,
    and it arrived from a hard barrier rather than from anything exotic.

    A smooth survival hazard (quant/hazard.py) narrows it but does not close
    it, which is worth stating because this docstring previously claimed
    otherwise. The hazard restores the gradient in the region a sensible policy
    operates in; it cannot help where survival has underflowed to exactly zero,
    and a cold start drives paths there within a few quarters. Measured over
    eight quarters, a cold start reaches 10.5 against a constant policy's 25.4
    with a hard barrier and 3.2 against 25.2 with the hazard. Warm-starting is
    required either way.

    Left to itself (`warm_start=None`) this turns on exactly when a barrier is
    active, because that is when the desert exists. With no barrier nothing
    dies, the objective is smooth everywhere, and a cold start converges to
    the closed form to ~1e-15 -- warm-starting there only moves the starting
    point of an optimization that was already working.
    """
    profile = env.config.profile
    horizon = env.config.horizon
    policy = PerPathPolicy(crn.batch, horizon, profile, shared_budgets=shared_budgets)
    if warm_start is None:
        warm_start = env.config.equity_floor > -float("inf")

    if warm_start:
        constant, _ = optimize_constant(env, crn, n_steps=max(1, n_steps // 2), lr=lr)
        policy.fill_from_constant(constant.raw, horizon, crn.batch)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        (-env.value_of(env.rollout(policy, crn, differentiable=True))).backward()
        optimizer.step()

    return policy, evaluate(policy, env, crn)
