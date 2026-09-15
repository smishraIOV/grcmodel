"""Rung 3, first instalment: a state-feedback policy trained through the simulator.

This is the first *implementable* policy in the project. The perfect-information
bound holds a free parameter per path and so cannot be deployed by a firm that
has to decide before seeing the quarter; this one maps the state it can
actually observe to an action, and can.

Trained by backpropagation through the rollout rather than by a score-function
estimator. The simulator is differentiable, so the gradient is exact rather
than sampled -- pathwise gradients carry O(1) variance where score-function
gradients carry O(1/sqrt(N)), which is why the same machinery reaches the
closed form to ~1e-15 elsewhere in this project. Reaching for a model-free
algorithm here would be discarding information the model already has.

Two things this buys beyond a number:

- The sandwich `V(neural) <= V* <= V_PI` brackets the true optimum. A learner
  above the bound has an information leak; one far below a hand-tuned constant
  has an optimization problem. Neither failure is visible from the value alone.
- `V_PI - V(neural)` is the expected value of perfect information. If it is
  near zero the state carries nothing a policy can exploit, and no amount of
  learner sophistication will help -- which is worth discovering before
  building one.
"""

from dataclasses import dataclass

import torch

from quant.env.actions import N_RAW, PAYOUT_INIT, ActionSpec, FirmAction, abandon_init
from quant.env.env import EvalResult, FirmEnv, evaluate
from quant.env.shocks import CommonRandomNumbers
from quant.env.state import FirmState


class NeuralPolicy(torch.nn.Module):
    """Observation -> action, through a small MLP.

    Deliberately small. The state is low-dimensional and the optimal
    investment rule is close to a one-dimensional map of wealth, so capacity
    is not the binding constraint -- and a small network keeps the grid solver
    that will check this one feasible.
    """

    def __init__(self, env: FirmEnv, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.env = env
        profile = env.config.profile
        obs_dim = env.observe(env.reset(1)).shape[-1]

        layers: list[torch.nn.Module] = []
        width = obs_dim
        for _ in range(n_layers):
            layers += [torch.nn.Linear(width, hidden), torch.nn.Tanh()]
            width = hidden
        final = torch.nn.Linear(width, N_RAW)
        # Start near softplus(0): a small positive spend and investment, so the
        # first gradients are informative rather than saturated.
        torch.nn.init.zeros_(final.bias)
        with torch.no_grad():
            # Scaled to the horizon: a per-quarter figure that is negligible over
            # eight quarters is halfway out the door over thirty-two.
            final.bias[-2] = abandon_init(env.config.horizon)
            final.bias[-1] = PAYOUT_INIT   # ...and retaining most of what it earns
        torch.nn.init.normal_(final.weight, std=1e-2)
        layers.append(final)

        self.net = torch.nn.Sequential(*layers).to(
            device=profile.device, dtype=profile.dtype
        )

    def forward(self, state: FirmState) -> FirmAction:
        return ActionSpec.from_raw(self.net(self.env.observe(state)))

    def __call__(self, state: FirmState) -> FirmAction:
        return self.forward(state)


@dataclass
class TrainResult:
    policy: NeuralPolicy
    evaluation: EvalResult
    history: list[float]


def train_pathwise(
    env: FirmEnv,
    crn: CommonRandomNumbers,
    hidden: int = 64,
    n_steps: int = 3000,
    lr: float = 3e-3,
    seed: int = 0,
    record_every: int = 100,
    grad_clip: float = 100.0,
) -> TrainResult:
    """Maximize firm value by gradient ascent through the differentiable rollout.

    The same common random numbers are used every step. That makes the
    objective deterministic, so this is a smooth optimization rather than a
    noisy one -- and it makes the result directly comparable with any other
    solver given the same `crn`, since both see the identical scenario set
    rather than two independent samples of it.
    """
    torch.manual_seed(seed)
    policy = NeuralPolicy(env, hidden=hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # **The gradient is clipped, and the reason is measured rather than
    # precautionary.** At the five-year horizon roughly one run in six used to
    # destroy itself: training converged normally for ~500 steps with the
    # gradient norm *falling* the whole way (value 32.97, norm 0.64 at step
    # 500), then a single step produced a norm of 2.7e5 -- four hundred thousand
    # times the previous one -- and the policy was worthless eight steps later.
    #
    # The amplifier is the run channel's straight-through relaxation. It
    # differentiates `(logit(p) - logit(u)) / 0.1`, and `d logit(p)/dp` is
    # `1/(p(1-p))`, about 147 at a run probability near 0.7%; the temperature
    # multiplies that by another ten. Switching the run channel off drops the
    # peak norm from 2.7e5 to 82 and the failure disappears.
    #
    # What follows the spike is a runaway rather than a stumble, because
    # `run_probability` is exponential in the capital ratio: leverage up, ratio
    # down, runs likelier, fire sales, capital down. Measured over eleven steps,
    # leverage 4.98 -> 13.60 and run probability 0.7% -> 41.7%, ending at zero
    # survival. Death and wind-down are both absorbing, so none of it is
    # recoverable.
    #
    # Adam does not protect against this -- it makes it worse. The update is
    # `lr * g / sqrt(v)`, and after 500 steps of small steady gradients `v` is
    # tiny, so a sudden enormous `g` produces a step far larger than `lr`; the
    # variance estimate only catches up afterwards. Clipping caps `g` *before*
    # that ratio is formed, which is why it is the right tool and a smaller `lr`
    # is not.
    #
    # 100 is chosen where the two distributions separate, not as the first value
    # that worked. Steady-state norms are 0.6-2, the cold start peaks near 82,
    # and the pathology is 2.7e5. Measured at the failing seed it fires on 2
    # steps of 600 and turns 0.77 into 33.25; on a healthy seed it fires on 0 of
    # 600 and the result is unchanged to three decimals. Set `grad_clip=None` to
    # disable.
    #
    # Note this is still the only solver in the repo with a **cold start** --
    # it begins at a firm value near 0.5. `perfect_information_bound`
    # warm-starts from the constant policy and its docstring calls that
    # mandatory once the barrier is on. That remains unaddressed.
    history = []
    for step in range(n_steps):
        optimizer.zero_grad()
        value = env.value_of(env.rollout(policy, crn, differentiable=True))
        (-value).backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()
        if step % record_every == 0:
            history.append(value.item())

    return TrainResult(policy, evaluate(policy, env, crn), history)
