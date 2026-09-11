"""One period of the firm's transition, composed of named sub-steps.

Split this way so each piece can be tested on its own and so the accounting
identity can be asserted elementwise rather than inferred. The static model's
`compute_value` did all of this in one expression and then took the
expectation inside the same function, which is what made it impossible to
extend to a horizon: there was no per-path reward to carry forward.

The identity every step must satisfy, checked in tests/test_env.py to 1e-10:

    equity' = equity - grc - loss - investment + production - premium

`differentiable` is declared per dynamics rather than assumed. As cliff events
and a survival barrier arrive, some channels stop being differentiable; making
each one say so keeps that a deliberate, visible decision instead of something
that quietly drifts and silently degrades the pathwise gradients.
"""

from dataclasses import dataclass
from typing import Protocol

import torch

from quant.env.actions import FirmAction
from quant.env.shocks import Shock
from quant.env.state import FirmState, StepResult
from quant.frictions import exponential_mitigation, financing_cost, production
from quant.numerics import NumericsProfile
from quant.params import FirmParams, GrcAlphas


class FirmDynamics(Protocol):
    differentiable: bool

    def step(self, state: FirmState, action: FirmAction, shock: Shock) -> StepResult: ...


@dataclass(frozen=True)
class StandardDynamics:
    """Convex-financing-cost dynamics: the static model, one period at a time.

    The financing friction here is the smooth convex premium inherited from
    the corporate-finance literature. It is scaffolding: a later stage
    replaces it with a survival hazard, which is a better fit for a business
    whose dominant risks are discrete cliff events. Keeping it now means every
    stage until then has an exact regression target.
    """

    firm: FirmParams
    alphas: GrcAlphas
    profile: NumericsProfile
    financing_scale: float = 1.0
    equity_floor: float = -float("inf")
    differentiable: bool = True

    # -- sub-steps -------------------------------------------------------

    def mitigated_loss(self, grc: torch.Tensor, shock: Shock) -> torch.Tensor:
        """Loss after GRC, each family acting on its own moment.

        Compliance is absent here on purpose: its budget acts through the path
        weight, by making a breach rarer rather than cheaper.
        """
        credit = exponential_mitigation(shock.credit_loss, grc[..., 0], self.alphas.credit)
        operational = shock.op_occurs * exponential_mitigation(
            shock.op_severity, grc[..., 1], self.alphas.operational
        )
        compliance = shock.compliance_occurs * shock.compliance_severity
        return credit + operational + compliance

    def path_weight(self, grc: torch.Tensor, shock: Shock) -> torch.Tensor:
        """Likelihood ratio for the compliance budget's effect on breach odds.

        A breach is discrete, so the budget cannot act by shrinking a sampled
        indicator -- torch.bernoulli is not differentiable in p. Indicators are
        drawn at the base probability and each path is reweighted by p(g)/p0 on
        breach paths and (1-p(g))/(1-p0) elsewhere. Differentiable in g, and it
        keeps breaches discrete, which matters because the spread they create
        is what the convex premium prices
        (docs/static-model-debug-notes.md section 5).

        Returned unnormalized: normalizing per step would destroy the product
        over a trajectory. Trajectory.path_weights normalizes once, at the end.
        """
        p0 = shock.compliance_base_probability
        p = exponential_mitigation(
            self.profile.tensor(p0), grc[..., 2], self.alphas.compliance
        )
        ratio = torch.where(shock.compliance_occurs > 0.5, p / p0, (1.0 - p) / (1.0 - p0))
        return shock.base_weight * ratio

    def financing(self, wealth: torch.Tensor, investment: torch.Tensor):
        external = torch.clamp(investment - wealth, min=0.0)
        premium = financing_cost(
            external,
            convexity=self.firm.financing_convexity,
            reference=self.firm.distress_reference,
            scale=self.financing_scale,
        )
        return external, premium

    # -- the transition --------------------------------------------------

    def step(self, state: FirmState, action: FirmAction, shock: Shock) -> StepResult:
        spend = action.total_grc()
        loss = self.mitigated_loss(action.grc, shock)
        wealth = state.equity - spend - loss
        external, premium = self.financing(wealth, action.investment)
        produced = production(
            action.investment, self.firm.production_scale, self.firm.production_curvature
        )

        equity = wealth - action.investment + produced - premium
        survives = equity >= self.equity_floor
        moved = state.advance(equity=equity, grc_stock=action.grc, alive=state.alive & survives)
        nxt = moved.freeze_dead(state)

        return StepResult(
            state=nxt,
            # Nothing is distributed before the horizon: all value is carried
            # in equity and realized by the terminal value. Stage 2 splits out
            # a dividend once there is a discount rate to trade it off against.
            reward=torch.zeros_like(equity),
            weight=self.path_weight(action.grc, shock),
            terminated=state.alive & ~survives,
            info={
                "loss": loss,
                "wealth": wealth,
                "external": external,
                "premium": premium,
                "production": produced,
                "grc_spend": spend,
            },
        )
