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
from quant.hazard import failure_intensity, intensity_components, log_survival
from quant.numerics import NumericsProfile
from quant.params import FirmParams, GrcAlphas, HazardParams


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
    # Fraction of the GRC stock that decays each period. 1.0 makes the stock
    # equal the flow, which is the static model: spend buys one period of
    # protection and nothing more.
    grc_depreciation: float = 1.0
    # None switches the smooth hazard off, leaving the hard barrier as the only
    # death channel. That is the configuration the closed-form and static
    # regression tests run against, not the intended model.
    hazard: HazardParams | None = None
    # Off by default: the reduction config has no exit option, which is what
    # the static model assumed and what its closed form is derived under.
    allow_abandonment: bool = False
    # Off by default: the reduction config lets the firm fund any investment at
    # a convex price, which is what the static model assumed.
    funding_constrained: bool = False
    # Off by default: the reduction retains everything and is valued at the
    # horizon, which is what every earlier stage assumed.
    allow_payout: bool = False
    differentiable: bool = True

    def orderly_value(self, equity: torch.Tensor) -> torch.Tensor:
        """What winding down deliberately recovers, evaluated on the equity the
        firm still has when it decides -- before it operates for another
        quarter and risks losing more."""
        return self.firm.orderly_recovery * torch.clamp(equity, min=0.0)

    def failure_value(self, equity: torch.Tensor) -> torch.Tensor:
        """What is recovered if the firm fails this period.

        A fraction of whatever positive equity remains, floored at zero by
        limited liability. GRC is absent by construction: it acts on how often
        failure happens, never on what failure costs.

        The floor is also where the gradient stops, which is worth knowing
        rather than discovering. Below zero equity this term is flat, so the
        only thing still pushing a path away from deep insolvency is the
        survival probability -- which is exactly the job the hazard was brought
        forward to do.
        """
        return self.firm.failure_recovery * torch.clamp(equity, min=0.0)

    def log_survival(
        self, state: FirmState, equity: torch.Tensor, stock: torch.Tensor
    ) -> torch.Tensor:
        """log P(survive this period), given where the period left the firm.

        Takes the GRC stock as well as equity, because two of the three hazard
        channels are reduced by it directly. That is what makes GRC buy
        survival rather than only smaller losses -- and it is the difference
        between a programme justified by expected-loss reduction and one
        justified by the franchise it protects.

        Two channels, deliberately layered. The smooth hazard does the economic
        work and supplies the gradient. The hard barrier underneath is a
        numerical backstop: the convex financing premium is unbounded below, so
        without an absorbing floor equity squares each period and overflows
        float64 within eight quarters. With the hazard on, the barrier should
        rarely bind -- approaching it already costs survival continuously.
        """
        intensity = (
            failure_intensity(
                equity,
                stock,
                self.firm.initial_equity,
                self.hazard,
                self.alphas,
                self.firm.periods_per_year,
            )
            if self.hazard is not None
            else torch.zeros_like(equity)
        )
        survived = log_survival(intensity)
        crossed = torch.full_like(equity, -float("inf"))
        alive_now = torch.where(equity >= self.equity_floor, survived, crossed)
        # A path that was already gone contributes no further hazard; its
        # cumulative survival is already zero.
        return torch.where(state.alive, alive_now, torch.zeros_like(equity))

    def grc_stock(self, state: FirmState, action: FirmAction) -> torch.Tensor:
        """Next period's GRC capital: K' = (1 - delta) K + g.

        Mitigation is applied to this *post-spend* stock, so a control bought
        this quarter protects this quarter. The alternative -- mitigating on
        the opening stock, so spend takes a period to bite -- is arguably more
        realistic for a compliance programme and would add an install lag, but
        it costs the exact reduction to the static model at delta = 1 and buys
        an unmeasurable parameter. Persistence is where the option value comes
        from, not delay: at delta < 1 a quarter's spend keeps working in later
        quarters without being respent, which is what makes pre-emptive GRC
        worth more than its one-period loss reduction and what makes this more
        than the static problem repeated.

        alpha applies to the stock, not the flow, which keeps its units
        identical to the static model's.
        """
        return (1.0 - self.grc_depreciation) * state.grc_stock + action.grc

    # -- sub-steps -------------------------------------------------------

    def mitigated_loss(self, stock: torch.Tensor, shock: Shock) -> torch.Tensor:
        """Loss after GRC, each family acting on its own moment.

        Takes the GRC *stock*, not the period's spend. Compliance is absent
        here on purpose: its budget acts through the path weight, by making a
        breach rarer rather than cheaper.
        """
        credit = exponential_mitigation(shock.credit_loss, stock[..., 0], self.alphas.credit)
        operational = shock.op_occurs * exponential_mitigation(
            shock.op_severity, stock[..., 1], self.alphas.operational
        )
        compliance = shock.compliance_occurs * shock.compliance_severity
        return credit + operational + compliance

    def path_weight(self, stock: torch.Tensor, shock: Shock) -> torch.Tensor:
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
            self.profile.tensor(p0), stock[..., 2], self.alphas.compliance
        )
        # torch.where evaluates both branches, so a bare p / p0 produces a NaN
        # gradient at p0 = 0 even though that branch is never selected -- and
        # p0 = 0 is a config someone reaches for to switch the channel off.
        # Flooring the denominator keeps the unselected branch finite; where it
        # bites there are no breach paths for it to apply to.
        safe_p0 = max(p0, torch.finfo(self.profile.dtype).tiny)
        ratio = torch.where(
            shock.compliance_occurs > 0.5, p / safe_p0, (1.0 - p) / (1.0 - p0)
        )
        return shock.base_weight * ratio

    def funding_capacity(self, wealth: torch.Tensor) -> torch.Tensor:
        """The most the firm can deploy this quarter: what it has, plus what it
        can raise against it.

        Both terms collapse together in a bad quarter, which is the point. The
        multiple is applied to internal wealth, and market access is a sigmoid
        in wealth over opening equity -- so a firm that has just taken a large
        loss finds both that it has less of its own money and that less of
        anyone else's is available. Funding withdraws exactly when it is needed.
        """
        if not self.funding_constrained:
            return torch.full_like(wealth, float("inf"))
        ratio = wealth / self.firm.initial_equity
        access = torch.sigmoid(
            (ratio - self.firm.market_access_ratio) / self.firm.market_access_scale
        )
        usable = torch.clamp(wealth, min=0.0)
        return usable + self.firm.external_funding_multiple * usable * access

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
        stock = self.grc_stock(state, action)
        loss = self.mitigated_loss(stock, shock)
        wealth = state.equity - spend - loss

        # The firm may want more than it can fund. What it deploys is the
        # lesser of the two, so a bad quarter forces underinvestment rather
        # than merely making investment expensive.
        capacity = self.funding_capacity(wealth)
        investment = torch.minimum(action.investment, capacity)

        external, premium = self.financing(wealth, investment)
        produced = production(
            investment, self.firm.production_scale, self.firm.production_curvature
        )

        gross = wealth - investment + produced - premium
        # Distributed out of what the quarter actually left, and only if it
        # left something. This is where the model finally has a genuine
        # intertemporal trade-off: a dividend is worth its face value now,
        # while capital retained is worth whatever it buys in survival and
        # funding capacity later. Until now every reward was zero and the
        # discount rate was a scalar multiplier on a terminal value.
        # Out of the quarter's profit, never out of the capital base. That is
        # the ordinary accounting constraint on dividends, it needs no
        # parameter, and without it the control is a way to strip the firm:
        # distributing all equity returns it at face value, which strictly
        # beats the 0.7 an orderly wind-down recovers, so the exit option is
        # dominated and the balance sheet can be emptied in a quarter.
        #
        # With it, a firm that distributes everything it earns holds its
        # capital flat -- which is what keeps funding scarce and the
        # constraint above binding.
        distributable = torch.clamp(gross - state.equity, min=0.0)
        dividend = (
            action.payout * distributable
            if self.allow_payout
            else torch.zeros_like(gross)
        )
        equity = gross - dividend
        survives = equity >= self.equity_floor
        intensity = (
            intensity_components(
                equity,
                stock,
                self.firm.initial_equity,
                self.hazard,
                self.alphas,
                self.firm.periods_per_year,
            )
            if self.hazard is not None
            else None
        )
        moved = state.advance(equity=equity, grc_stock=stock, alive=state.alive & survives)
        nxt = moved.freeze_dead(state)

        return StepResult(
            state=nxt,
            # Nothing is distributed before the horizon: all value is carried
            # in equity and realized by the terminal value. Stage 2 splits out
            # a dividend once there is a discount rate to trade it off against.
            reward=dividend,
            weight=self.path_weight(stock, shock),
            log_survival=self.log_survival(state, equity, stock),
            failure_value=self.failure_value(equity),
            # Decided at the start of the period, on what the firm knows then:
            # you wind down on the basis of the balance sheet you have, not the
            # quarter you are about to have.
            abandon_prob=(
                action.abandon * state.alive
                if self.allow_abandonment
                else torch.zeros_like(equity)
            ),
            orderly_value=self.orderly_value(state.equity),
            terminated=state.alive & ~survives,
            info={
                "loss": loss,
                "wealth": wealth,
                "external": external,
                "premium": premium,
                "production": produced,
                "grc_spend": spend,
                "dividend": dividend,
                "investment": investment,
                "funding_capacity": capacity,
                "funding_binds": (action.investment > capacity),
                "grc_flow": action.grc,
                "grc_stock": stock,
                "hazard": intensity,
            },
        )
