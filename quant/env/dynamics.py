"""One period of the firm's transition, composed of named sub-steps.

Split this way so each piece can be tested on its own and so the accounting
identity can be asserted elementwise rather than inferred. The static model's
`compute_value` did all of this in one expression and then took the
expectation inside the same function, which is what made it impossible to
extend to a horizon: there was no per-path reward to carry forward.

The identity every step must satisfy, checked in tests/test_env.py to 1e-10:

    equity' = equity - grc - loss - investment + production - premium
                     - interest + reserve income

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
from quant.params import CliffParams, FirmParams, FundingParams, GrcAlphas, HazardParams


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
    # None switches the cliff channel off, which is what the reduction and the
    # closed-form benchmark run under.
    cliff: CliffParams | None = None
    # Off for the exactly-solvable oracle, which needs credit loss to be
    # exogenous: the closed form g* = ln(alpha X)/alpha requires the objective
    # to separate across families, and a loss proportional to a chosen book
    # does not separate. The oracle is a deliberately degenerate configuration
    # (four states, no hazard, no funding constraint) and this is one more
    # restriction of the same kind.
    credit_scales_with_book: bool = False
    # None leaves the firm with no liabilities: it funds its book out of equity
    # plus a multiple of equity that costs nothing and is settled inside the
    # period. That is what every result before the liability stage was produced
    # under, and it is what the closed form and the static regression require.
    # Set, the firm carries a deposit stock that persists between periods, pays
    # for it, and finds it shrinking when its capital does.
    funding: FundingParams | None = None
    differentiable: bool = True

    def cliff_loss(self, state: FirmState, stock: torch.Tensor, shock: Shock):
        """A severe, survivable hit whose frequency GRC reduces and whose
        severity it does not.

        The occurrence probability depends on the operational GRC stock, and a
        drawn Bernoulli has no gradient in its probability -- the problem
        docs/static-model-debug-notes.md section 5 met for compliance. The
        escape used there, likelihood-ratio reweighting, is a poor fit here:
        per-step weights multiply along a trajectory so their variance
        compounds, and at a base rate of 2.5% only a handful of paths per
        quarter would carry the entire gradient.

        This uses a straight-through relaxation instead. The forward pass is
        the true hard indicator, so the discrete jump and its tail are exactly
        right; the backward pass differentiates a tempered sigmoid of the same
        threshold. The gradient is biased, but the bias is a temperature knob
        rather than something that grows with the horizon, which is a much
        better failure mode than a variance that compounds.

        Smoothing the event away instead -- charging p(G) x severity every
        quarter -- is not available. The hazard is convex in equity, so
        E[h(E - L)] is not h(E - E[L]): replacing a 2.5% chance of losing a
        third of the firm with a certain loss of 1% deletes precisely the
        state this channel exists to represent.
        """
        probability = exponential_mitigation(
            self.profile.tensor(self.cliff.period_probability),
            stock[..., 1],  # operational GRC: the crypto rails are its pillar
            self.alphas.operational,
        )
        # Floored for the same reason `run_probability` is: `torch.logit` of
        # zero is -inf with an infinite derivative, so a cliff rate of zero --
        # the natural way to switch the channel off -- returns NaN from the
        # *backward* pass while the forward pass stays correct. Nothing looks
        # wrong until a whole solve comes back NaN.
        #
        # The run channel got this guard when the bug was found there; this one
        # did not, and the asymmetry survived until an ablation set
        # `period_probability = 0.0` and NaN'd from the first step. Third
        # instance of the same trap in this file.
        tiny = torch.finfo(self.profile.dtype).tiny
        probability = probability.clamp(min=tiny, max=1.0 - 1e-9)
        uniform = shock.cliff_uniform.clamp(1e-9, 1.0 - 1e-9)
        occurs = (uniform < probability).to(self.profile.dtype)

        relaxed = torch.sigmoid(
            (torch.logit(probability) - torch.logit(uniform))
            / self.cliff.relaxation_temperature
        )
        # Forward value is `occurs` exactly; the gradient is the relaxation's.
        indicator = occurs + relaxed - relaxed.detach()
        return indicator * shock.cliff_fraction * torch.clamp(state.equity, min=0.0)

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
        self,
        state: FirmState,
        equity: torch.Tensor,
        stock: torch.Tensor,
        unmet_share: torch.Tensor | None = None,
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
                unmet_share,
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

    def mitigated_loss(
        self, stock: torch.Tensor, shock: Shock, book: torch.Tensor
    ) -> torch.Tensor:
        """Loss after GRC, each family acting on its own moment.

        Takes the GRC *stock*, not the period's spend. Compliance is absent
        here on purpose: its budget acts through the path weight, by making a
        breach rarer rather than cheaper.

        **Credit loss scales with the book.** `shock.credit_loss` is a loss rate
        per unit deployed when `credit_scales_with_book` is on, so lending twice
        as much costs twice the expected defaults. It was an exogenous money
        amount, which meant the firm could expand its balance sheet at no
        additional credit risk -- measured, expected loss was 0.2712 a quarter
        whether it deployed 5 or 80. That inverts the central banking
        trade-off: there was no cost to taking assets.

        Operational and compliance losses do *not* scale with the book. An
        incident costs what it costs, and a regulator's penalty is not
        proportional to the loan portfolio.
        """
        rate = exponential_mitigation(shock.credit_loss, stock[..., 0], self.alphas.credit)
        credit = rate * book if self.credit_scales_with_book else rate
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

        Returns the **ratio only**, not the ratio times the path's prior
        probability. The prior belongs to the trajectory and is applied once;
        multiplying it in every period made the accumulated weight carry
        (1/batch)**T, which is a constant factor that cancels in the
        normalization and is therefore invisible -- right up until it
        underflows. At 2048 paths over 104 weekly steps that is 1e-344, every
        path's weight became exactly zero, and the normalization turned into
        0/0. Eight quarterly steps hid it completely.

        Returned unnormalized in the other sense too: normalizing per step
        would destroy the product over a trajectory. Trajectory.path_weights
        accumulates in log space and normalizes once, at the end.
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
        return torch.where(
            shock.compliance_occurs > 0.5, p / safe_p0, (1.0 - p) / (1.0 - p0)
        )

    def deposit_capacity(self, equity: torch.Tensor) -> torch.Tensor:
        """The deposit base this much capital can carry.

        A leverage limit times a market-access sigmoid, both in the capital
        ratio, so it is dimensionless in the same way the hazard is. The
        product is what makes the liability side procyclical: a firm that has
        lost capital can carry fewer deposits both because the limit is a
        multiple of a smaller number and because the market is less willing to
        leave money with it. The two collapse together, which is how funding
        actually behaves.

        Deposits do not jump to this level -- `deposit_flow` moves toward it.
        The gap between where the base is and where it can be is the state that
        makes the liability side more than a formula in equity.
        """
        ratio = equity / self.firm.initial_equity
        access = torch.sigmoid(
            (ratio - self.funding.access_ratio) / self.funding.access_scale
        )
        return self.funding.deposit_capacity * torch.clamp(equity, min=0.0) * access

    def deposit_flow(
        self, deposits: torch.Tensor, equity: torch.Tensor
    ) -> torch.Tensor:
        """D' = D + theta (capacity(E') - D): partial adjustment toward capacity.

        Evaluated on *closing* equity, so a quarter's losses start the deposit
        base moving the same quarter they land rather than a period later.

        Partial rather than instant, and that is the whole reason deposits are
        a state variable. At theta = 1 the base is a deterministic function of
        equity, the liability side collapses back into the asset side, and
        there is nothing outstanding that could run. The stock has to be able
        to sit away from where the firm's capital says it belongs.
        """
        theta = self.funding.period_adjustment(self.firm.periods_per_year)
        target = self.deposit_capacity(equity)
        return torch.clamp(deposits + theta * (target - deposits), min=0.0)

    def deposit_interest(self, deposits: torch.Tensor) -> torch.Tensor:
        """What the funding costs for the period, on the base outstanding when
        it opened.

        Charged whether or not the book earns, which is the point: without a
        price, leverage is free and the firm takes all of it, and 'how levered
        should we be' stops being a question. It is also the channel that makes
        a *liquid* balance sheet expensive -- deposits parked as reserves still
        pay interest -- which is what the next stage needs in order for holding
        a buffer to be a real decision rather than a free one.
        """
        rate = self.funding.period_deposit_rate(self.firm.periods_per_year)
        return rate * torch.clamp(deposits, min=0.0)

    def reserves_held(
        self, state: FirmState, fundable: torch.Tensor, investment: torch.Tensor
    ) -> torch.Tensor:
        """Balance-sheet funding the firm chose not to lend.

        The residual of the lending decision, not a control of its own: whatever
        equity and deposits could have deployed and did not. That makes the
        liquidity buffer implicit in how much the firm lends, which is the
        honest reading -- a bank does not separately choose a buffer, it chooses
        a book and lives with the remainder.
        """
        return torch.clamp(
            torch.clamp(fundable, min=0.0)
            + torch.clamp(state.deposits, min=0.0)
            - investment,
            min=0.0,
        )

    def reserve_income(
        self, state: FirmState, fundable: torch.Tensor, investment: torch.Tensor
    ) -> torch.Tensor:
        """What the funding the firm did not lend earns while it sits there.

        Reserves are the residual of the funding decision: what is on the
        balance sheet and not deployed. Without this term, deposits the firm
        cannot profitably lend are a pure deadweight loss and the model punishes
        a bank for having a franchise -- measured at 2.5x leverage on the
        pre-liability opportunity, firm value fell 5% purely because the firm
        was carrying funding it had no use for.

        Priced below the deposit rate, so idle funding carries a small negative
        spread. That spread is the entire reason a liquidity buffer will be a
        decision rather than a free good once withdrawals exist.

        Computed from the **balance sheet**, not from the funding capacity, and
        the difference is not cosmetic. With `funding_constrained=False` the
        capacity is `+inf`, so reserves were infinite, the period's income was
        infinite, and the next subtraction turned it into a NaN that propagated
        through every path. Capacity is a *limit* on what may be deployed;
        reserves are what actually sits there, and only the second is a
        quantity the firm can earn on.
        """
        rate = self.funding.period_reserve_rate(self.firm.periods_per_year)
        return rate * self.reserves_held(state, fundable, investment)

    def run_probability(
        self, stock: torch.Tensor, equity: torch.Tensor
    ) -> torch.Tensor:
        """Chance the deposit base runs this period, given where the quarter's
        losses left the firm.

            p = (rate / ppy) * exp(-alpha_o * G_o) * exp((target - kappa) / scale)

        Three factors, and each is a separate claim.

        The **base rate** is how often an incident becomes public at all. The
        **GRC term** is the only place operational controls still buy survival,
        and they now buy it by making the run less likely rather than by being
        told they reduce a death rate -- the same alpha and the same mitigation
        curve the loss channels use.

        The **capital term** is what makes this a run rather than a weather
        event. It is evaluated on equity *after* the quarter's losses, so a bad
        quarter draws the run that then makes the quarter worse: losses thin the
        capital, thin capital draws depositors out, meeting them forces a fire
        sale, the fire sale thins the capital further. That loop is the whole
        mechanism, and a withdrawal drawn independently of the firm's condition
        would not have it.

        Gentler and earlier than the death hazard's capital term, deliberately.
        Depositors do not wait for insolvency, they leave on the suspicion of
        it, which is how a bank that is solvent on paper gets killed.
        """
        base = self.profile.tensor(
            self.funding.period_run_rate(self.firm.periods_per_year)
        )
        mitigated = exponential_mitigation(base, stock[..., 1], self.alphas.operational)
        kappa = equity / self.firm.initial_equity
        stress = torch.exp(
            torch.clamp(
                (self.funding.run_capital_target - kappa) / self.funding.run_capital_scale,
                max=20.0,
            )
        )
        # Floored as well as capped. `withdrawal` takes the logit of this, and
        # logit(0) is -inf with an infinite derivative, so a run rate of zero --
        # the configuration someone reaches for to switch the channel off --
        # produced a NaN gradient that propagated through every path and every
        # other control. The forward pass was correct throughout; only the
        # backward pass was poisoned, so nothing looked wrong until a solve
        # returned NaN.
        #
        # Same trap as the compliance likelihood ratio at p0 = 0
        # (`path_weight`), in a new place. Flooring at `tiny` keeps the logit
        # finite and the event impossible.
        tiny = torch.finfo(self.profile.dtype).tiny
        return torch.clamp(mitigated * stress, min=tiny, max=1.0 - 1e-9)

    def withdrawal(
        self, state: FirmState, stock: torch.Tensor, equity: torch.Tensor, shock: Shock
    ):
        """How much of the deposit base walks out this period.

        Straight-through relaxed, exactly as `cliff_loss` is: the forward pass
        is the true hard indicator so the discrete jump and its tail are right,
        and the backward pass differentiates a tempered sigmoid of the same
        threshold. Smoothing the event into a certain small outflow is not
        available for the same reason it was not available there -- the cost of
        a run is convex in its size, because a small one comes out of reserves
        and a large one comes out of a fire sale.

        Returns (probability, amount) so the probability can be reported: it is
        the quantity operational GRC actually acts on, and the one worth showing
        a risk owner.
        """
        probability = self.run_probability(stock, equity)
        uniform = shock.run_uniform.clamp(1e-9, 1.0 - 1e-9)
        occurs = (uniform < probability).to(self.profile.dtype)
        relaxed = torch.sigmoid(
            (torch.logit(probability) - torch.logit(uniform))
            / self.funding.relaxation_temperature
        )
        indicator = occurs + relaxed - relaxed.detach()
        return probability, indicator * shock.run_fraction * torch.clamp(
            state.deposits, min=0.0
        )

    def meet_withdrawal(
        self,
        demanded: torch.Tensor,
        reserves: torch.Tensor,
        book: torch.Tensor,
    ):
        """Pay depositors out of reserves first, then by selling book early.

        Reserves are cash and go at par. The book has not matured, so raising
        `S` from it costs `S / (1 - h)` of book and destroys `S * h / (1 - h)`.
        That is the only loss in the model caused by the *timing* of an
        obligation rather than by anything going wrong with an asset, and it is
        what makes a liquidity failure a different thing from a solvency one.

        It is also what makes an idle reserve worth holding. Reserves earn less
        than the book and still cost deposit interest, so a firm that ignored
        runs would hold none; the fire-sale haircut is the price of that
        choice, paid only in the states where it matters.

        The liquidation is capped at the book available. A firm that cannot
        raise the cash even by selling everything has not met its obligations,
        and the shortfall stays owed -- which drives equity sharply negative and
        lets the capital hazard price it, rather than introducing a second hard
        death branch with no gradient.
        """
        from_reserves = torch.minimum(demanded, torch.clamp(reserves, min=0.0))
        shortfall = torch.clamp(demanded - from_reserves, min=0.0)
        haircut = self.funding.fire_sale_haircut
        wanted = shortfall / (1.0 - haircut)
        liquidated = torch.minimum(wanted, torch.clamp(book, min=0.0))
        return from_reserves, liquidated

    def funding_capacity(
        self, state: FirmState, wealth: torch.Tensor
    ) -> torch.Tensor:
        """The most the firm can deploy this quarter: what it has, plus what it
        has raised against it.

        With a liability side, that is simply equity plus the deposit stock --
        a bank cannot lend money it has not been given, and the constraint
        needs no free parameter because the balance sheet already states it.
        The procyclicality lives in `deposit_capacity`, one period upstream:
        this quarter's losses shrink the deposit base, which binds next
        quarter's lending. That lag is the Froot-Stein channel in its proper
        place -- the firm is not punished for a loss at the instant it takes
        it, it is punished by having less to work with afterwards.

        Without one, the pre-liability behaviour: a multiple of internal wealth
        times a market-access sigmoid, settled inside the period and costing
        nothing. Both terms collapse together in a bad quarter, which was that
        version's way of saying the same thing in one step instead of two.
        """
        if not self.funding_constrained:
            return torch.full_like(wealth, float("inf"))
        usable = torch.clamp(wealth, min=0.0)
        if self.funding is not None:
            return usable + torch.clamp(state.deposits, min=0.0)
        ratio = wealth / self.firm.initial_equity
        access = torch.sigmoid(
            (ratio - self.firm.market_access_ratio) / self.firm.market_access_scale
        )
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
        if self.funding is not None and state.deposits is None:
            raise ValueError(
                "funding parameters are set but the state carries no deposits; "
                "build the opening state with FirmEnv.reset, which decides this "
                "from the same config"
            )
        spend = action.total_grc()
        stock = self.grc_stock(state, action)

        # Lend first, then find out what defaults. A bank sets its book from
        # the capital it has at the start of the period; losses arrive on what
        # it lent. The previous ordering computed losses before the funding
        # decision, which cannot work once credit loss depends on the size of
        # the book -- the book would have to be known before it was chosen.
        #
        # The Froot-Stein channel survives the reordering and is arguably
        # better for it: a bad quarter now constrains lending in the *next*
        # quarter through lower equity, which is how the constraint actually
        # bites in a bank.
        fundable = state.equity - spend
        capacity = self.funding_capacity(state, fundable)
        investment = torch.minimum(action.investment, capacity)

        loss = self.mitigated_loss(stock, shock, investment)
        cliff = (
            self.cliff_loss(state, stock, shock)
            if self.cliff is not None
            else torch.zeros_like(loss)
        )
        loss = loss + cliff
        wealth = fundable - loss

        # The run is drawn on equity *after* the quarter's losses, so a bad
        # quarter draws the run that then makes it worse. Everything from here
        # to `produced` is the liability side of the period: who asks for their
        # money back, what the firm can pay them out of, and what it costs to
        # find the rest.
        if self.funding is not None:
            held = self.reserves_held(state, fundable, investment)
            run_probability, demanded = self.withdrawal(state, stock, wealth, shock)
            from_reserves, liquidated = self.meet_withdrawal(
                demanded, held, investment
            )
            raised = liquidated * (1.0 - self.funding.fire_sale_haircut)
            paid = from_reserves + raised
            # The haircut, reported separately because it is the only loss in
            # the model that no asset going wrong can explain.
            fire_sale = liquidated - raised
            book = investment - liquidated
            # As a share of what was asked for, not as an amount: the hazard
            # takes dimensionless arguments, and "could not pay a tenth of what
            # was demanded" means the same thing whatever currency the firm is
            # denominated in. Zero when nothing was demanded, which is most
            # periods.
            unmet = torch.clamp(demanded - paid, min=0.0)
            unmet_share = unmet / torch.clamp(demanded, min=1e-12)
            interest = self.deposit_interest(state.deposits)
            # Earned on what is left after depositors have been paid, not on
            # the opening balance: reserves that walked out mid-period did not
            # sit there earning for the firm.
            reserves = self.funding.period_reserve_rate(
                self.firm.periods_per_year
            ) * torch.clamp(held - from_reserves, min=0.0)
        else:
            zero = torch.zeros_like(wealth)
            held = run_probability = demanded = from_reserves = zero
            liquidated = raised = paid = fire_sale = zero
            book = investment
            interest = reserves = zero
            unmet = unmet_share = zero

        # Priced against wealth *after* losses, as before. Only the funding
        # capacity had to move ahead of the losses -- the book cannot be sized
        # by a quantity that depends on the book. Keeping the premium where it
        # was preserves the machine-precision identity against quant/model.py.
        external, premium = self.financing(wealth, investment)
        produced = production(
            book, self.firm.production_scale, self.firm.production_curvature
        )

        # The liquidated book leaves twice over: it stops producing, and it
        # comes back as cash worth (1 - h) of its face. `raised` is that cash.
        # Depositors take it, and the deposit base falls by the same amount, so
        # the two cancel out of equity and what remains is the haircut.
        gross = (
            wealth - investment + produced + raised - premium - interest + reserves
        )
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
                unmet_share,
            )
            if self.hazard is not None
            else None
        )
        # Depositors who left are gone before the base starts rebuilding, so a
        # run leaves the firm funding-impaired for several quarters rather than
        # for one. That persistence is what makes the run a state event and not
        # just a bad draw -- it is the same argument the GRC stock makes on the
        # asset side.
        deposits = (
            self.deposit_flow(
                torch.clamp(state.deposits - paid, min=0.0), equity
            )
            if self.funding is not None
            else None
        )
        moved = state.advance(
            equity=equity,
            grc_stock=stock,
            alive=state.alive & survives,
            deposits=deposits,
        )
        nxt = moved.freeze_dead(state)

        return StepResult(
            state=nxt,
            # Nothing is distributed before the horizon: all value is carried
            # in equity and realized by the terminal value. Stage 2 splits out
            # a dividend once there is a discount rate to trade it off against.
            reward=dividend,
            weight=self.path_weight(stock, shock),
            log_survival=self.log_survival(state, equity, stock, unmet_share),
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
                "cliff_loss": cliff,
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
                "interest": interest,
                "reserve_income": reserves,
                "reserves": held,
                "run_probability": run_probability,
                "withdrawal": demanded,
                "withdrawal_paid": paid,
                "liquidated": liquidated,
                "fire_sale_loss": fire_sale,
                # What the firm was asked for and could not raise even by
                # selling its whole book. Non-zero is a liquidity failure: the
                # obligation stays owed, equity carries it, *and* the liquidity
                # hazard channel fires on the share unpaid.
                "unmet_withdrawal": unmet,
                "unmet_share": unmet_share,
                # The opening balance sheet, so a diagnostic averaging these
                # reports what the firm was funding the quarter with rather
                # than what it closed on.
                "deposits": (
                    state.deposits
                    if state.deposits is not None
                    else torch.zeros_like(equity)
                ),
                "assets": state.assets(),
            },
        )
