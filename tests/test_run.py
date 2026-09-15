"""The run: a liquidity shock, a buffer, and the cost of selling early.

The liability side had a deposit stock before this, but nothing that could
happen to it suddenly. Deposits drifted toward the capacity the firm's capital
supported, which is a run in slow motion -- enough to make leverage a real
decision, not enough to make liquidity one.

Three claims are tested here, and they are separable:

    the shock      a run is discrete, its probability depends on the firm's
                   condition, and operational GRC reduces it
    the buffer     reserves absorb a withdrawal at par; the book does not
    the cost       raising S from an unmatured book destroys S*h/(1-h), which
                   is the only loss in the model no asset going wrong explains

The fourth claim -- that this *replaces* the assumed "depositors leave" death
rate rather than sitting alongside it -- is tested last, because it is the one
that would silently double-count if it were wrong.
"""

import math
from dataclasses import replace

import pytest
import torch

from quant.env.actions import FirmAction
from quant.env.env import EnvConfig, FirmEnv
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.numerics import REFERENCE
from quant.params import DEFAULTS, FundingParams

BATCH = 256


def run_env(horizon=1, funding=None, **overrides):
    config = EnvConfig.quarterly(
        horizon, funding=funding or FundingParams(), profile=REFERENCE, **overrides
    )
    return FirmEnv(config, MonteCarloSampler(DEFAULTS.sampler))


class Deploy:
    def __init__(self, grc: float, investment: float):
        self.grc, self.investment = grc, investment

    def __call__(self, state):
        batch = state.batch()
        return FirmAction(
            grc=REFERENCE.full((batch, 3), self.grc),
            investment=REFERENCE.full((batch,), self.investment),
            abandon=REFERENCE.full((batch,), 0.0),
            payout=REFERENCE.full((batch,), 0.0),
        )


def forced(shock, fraction: float, occurs: bool = True):
    """A shock whose run fires (or never fires) at a chosen severity."""
    like = shock.run_uniform
    return replace(
        shock,
        run_uniform=torch.full_like(like, 0.0 if occurs else 1.0 - 1e-12),
        run_fraction=torch.full_like(like, fraction),
    )


def step(env, policy, shock_fn=None, seed=0, batch=BATCH):
    crn = CommonRandomNumbers(seed, 1, batch, REFERENCE)
    state = env.reset(batch)
    shock = env.sampler(crn.at(0), REFERENCE)
    if shock_fn is not None:
        shock = shock_fn(shock)
    return state, env.dynamics.step(state, policy(state), shock)


# -- the identity ---------------------------------------------------------


def test_the_identity_survives_a_run():
    """equity' = equity - grc - loss - investment + production + cash raised
                        - premium - interest + reserve income - dividend.

    The liquidated book leaves twice over -- it stops producing, and it comes
    back as cash worth (1-h) of face. Depositors take that cash and the deposit
    base falls by the same amount, so the two cancel out of equity and what is
    left is the haircut. Asserted elementwise, because a fire sale that was
    double-counted or not counted at all would both average out plausibly.
    """
    env = run_env()
    state, result = step(env, Deploy(0.1, 60.0), lambda s: forced(s, 0.5))
    info = result.info
    raised = info["liquidated"] * (1.0 - DEFAULTS.funding.fire_sale_haircut)

    expected = (
        state.equity
        - info["grc_spend"]
        - info["loss"]
        - info["investment"]
        + info["production"]
        + raised
        - info["premium"]
        - info["interest"]
        + info["reserve_income"]
        - info["dividend"]
    )
    assert torch.allclose(result.state.equity, expected, atol=1e-10, rtol=0.0)


def test_no_run_reproduces_the_pre_run_transition():
    """A period in which nothing is withdrawn must be arithmetically identical
    to one from before the channel existed."""
    env = run_env()
    _, quiet = step(env, Deploy(0.1, 60.0), lambda s: forced(s, 0.5, occurs=False))
    for key in ("withdrawal", "liquidated", "fire_sale_loss", "unmet_withdrawal"):
        assert torch.count_nonzero(quiet.info[key]) == 0, key


# -- the buffer -----------------------------------------------------------


def test_reserves_absorb_a_small_withdrawal_at_par():
    """Cash goes at par; that is what makes it worth holding. A withdrawal
    inside the buffer must cost the firm nothing beyond the funding it lost."""
    env = run_env()
    # Lend well short of capacity, so the buffer is large.
    state, result = step(env, Deploy(0.1, 20.0), lambda s: forced(s, 0.05))
    info = result.info

    assert torch.all(info["reserves"] > info["withdrawal"]), "test needs a real buffer"
    assert torch.count_nonzero(info["liquidated"]) == 0
    assert torch.count_nonzero(info["fire_sale_loss"]) == 0
    assert torch.allclose(info["withdrawal_paid"], info["withdrawal"], atol=1e-12)


def test_a_withdrawal_past_the_buffer_costs_the_haircut():
    """Raising S out of an unmatured book costs S/(1-h) of book and destroys
    S*h/(1-h). Checked against the closed form rather than against itself."""
    env = run_env()
    state, result = step(env, Deploy(0.1, 1e6), lambda s: forced(s, 0.30))
    info = result.info
    haircut = DEFAULTS.funding.fire_sale_haircut

    shortfall = torch.clamp(info["withdrawal"] - info["reserves"], min=0.0)
    assert torch.all(shortfall > 0), "test needs the buffer to be exhausted"
    assert torch.allclose(info["liquidated"], shortfall / (1.0 - haircut), atol=1e-9)
    assert torch.allclose(
        info["fire_sale_loss"], shortfall * haircut / (1.0 - haircut), atol=1e-9
    )


def test_a_bigger_buffer_costs_less_to_run_on():
    """The buffer is worth something, which is the claim that makes it a
    decision. Same run, same severity, different amounts lent."""
    env = run_env()
    # A withdrawal of 0.30 against a deposit base of 64 is 19.2, so the levels
    # have to straddle that: a buffer larger than the run, one smaller, and
    # none at all.
    losses = []
    for lent in (20.0, 65.0, 1e6):
        _, result = step(env, Deploy(0.1, lent), lambda s: forced(s, 0.30))
        losses.append(result.info["fire_sale_loss"].mean().item())
    assert losses[0] < losses[1] < losses[2], losses
    assert losses[0] == pytest.approx(0.0, abs=1e-12)


def test_a_run_it_cannot_meet_is_reported_rather_than_hidden():
    """A firm that cannot raise the cash even by selling its whole book has not
    met its obligations. The shortfall stays owed and drives equity sharply
    negative, which the capital hazard prices -- rather than a second hard death
    branch carrying no gradient."""
    env = run_env()
    state, result = step(env, Deploy(0.1, 1e6), lambda s: forced(s, 1.0))
    info = result.info

    assert torch.all(info["unmet_withdrawal"] > 0)
    assert torch.all(info["liquidated"] <= info["investment"] + 1e-9)
    assert torch.all(result.state.equity < state.equity)


# -- the shock ------------------------------------------------------------


def test_operational_grc_makes_a_run_less_likely():
    """The only channel through which operational controls still buy survival,
    now that the asserted death rate is gone. They buy it through the balance
    sheet: a run avoided is a fire sale avoided."""
    env = run_env()
    equity = REFERENCE.full((4,), 16.0)
    probabilities = [
        env.dynamics.run_probability(REFERENCE.full((4, 3), g), equity)[0].item()
        for g in (0.0, 0.5, 1.0, 2.0)
    ]
    assert probabilities == sorted(probabilities, reverse=True), probabilities
    assert probabilities[-1] < 0.25 * probabilities[0]


def test_a_run_gets_likelier_as_capital_thins():
    """What makes this a run rather than a weather event, and what closes the
    loop: losses thin the capital, thin capital draws depositors out, meeting
    them forces a fire sale, which thins the capital further."""
    env = run_env()
    stock = REFERENCE.full((5, 3), 1.17)
    equity = REFERENCE.tensor([24.0, 16.0, 8.0, 4.0, 1.0])
    probabilities = env.dynamics.run_probability(stock, equity)
    assert torch.all(probabilities.diff() > 0), probabilities
    assert torch.all(probabilities <= 1.0)


def test_the_run_is_drawn_after_the_quarter_s_losses():
    """A bad quarter must draw the run that then makes it worse. Drawn on
    opening equity instead, the amplification loop would not exist and a run
    would be independent of everything the firm had just been through.
    """
    env = run_env()
    crn = CommonRandomNumbers(0, 1, BATCH, REFERENCE)
    state = env.reset(BATCH)
    shock = env.sampler(crn.at(0), REFERENCE)
    result = env.dynamics.step(state, Deploy(0.1, 60.0)(state), shock)

    opening = env.dynamics.run_probability(result.info["grc_stock"], state.equity)
    assert torch.all(result.info["run_probability"] >= opening - 1e-12)
    assert torch.any(result.info["run_probability"] > opening + 1e-9), (
        "losses must raise the run probability on at least some paths"
    )


def test_the_forward_pass_is_the_true_indicator_and_the_gradient_is_not_zero():
    """Straight-through, as the cliff channel is: `torch.bernoulli` has no
    gradient in its probability, so the forward pass uses the hard threshold and
    the backward pass differentiates a tempered sigmoid of the same comparison.
    """
    env = run_env()
    crn = CommonRandomNumbers(0, 1, 4096, REFERENCE)
    shock = env.sampler(crn.at(0), REFERENCE)
    state = env.reset(4096)

    scalar = REFERENCE.tensor(1.0, requires_grad=True)
    stock = torch.stack([scalar] * 3).expand(4096, 3)
    _, demanded = env.dynamics.withdrawal(state, stock, state.equity, shock)
    demanded.mean().backward()

    withdrawn = demanded.detach()
    assert torch.all((withdrawn == 0) | (withdrawn > 0.01)), "not a discrete event"
    assert scalar.grad is not None and scalar.grad.item() < 0.0, (
        "more operational GRC must lower expected withdrawals"
    )


def test_a_run_leaves_the_firm_funding_impaired():
    """Depositors who left are gone before the base starts rebuilding, so the
    damage outlasts the quarter. That persistence is what makes a run a state
    event rather than a bad draw, and it is the same argument the GRC stock
    makes on the asset side."""
    env = run_env(horizon=3)
    crn = CommonRandomNumbers(0, 3, BATCH, REFERENCE)
    state = env.reset(BATCH)
    policy = Deploy(0.1, 40.0)

    hit = env.dynamics.step(state, policy(state), forced(env.sampler(crn.at(0), REFERENCE), 0.5))
    spared = env.dynamics.step(
        state, policy(state), forced(env.sampler(crn.at(0), REFERENCE), 0.5, occurs=False)
    )
    assert torch.all(hit.state.deposits < spared.state.deposits)

    # And still behind a period later, having only partially rebuilt.
    shock = env.sampler(crn.at(1), REFERENCE)
    quiet = lambda s: forced(s, 0.0, occurs=False)
    after_hit = env.dynamics.step(hit.state, policy(hit.state), quiet(shock))
    after_spared = env.dynamics.step(spared.state, policy(spared.state), quiet(shock))
    assert torch.all(after_hit.state.deposits < after_spared.state.deposits)


def test_switching_the_run_off_does_not_poison_the_gradient():
    """A run rate of zero is the natural way to switch the channel off, and it
    took the logit of zero: -inf with an infinite derivative, so the *backward*
    pass returned NaN while the forward pass stayed correct. Nothing looked
    wrong until a whole solve came back NaN.

    Same trap as the compliance likelihood ratio at p0 = 0, in a new place.
    """
    env = run_env(funding=replace(FundingParams(), annual_run_rate=0.0))
    crn = CommonRandomNumbers(0, 1, 512, REFERENCE)
    shock = env.sampler(crn.at(0), REFERENCE)
    state = env.reset(512)

    scalar = REFERENCE.tensor(1.0, requires_grad=True)
    stock = torch.stack([scalar] * 3).expand(512, 3)
    _, demanded = env.dynamics.withdrawal(state, stock, state.equity, shock)
    assert torch.count_nonzero(demanded) == 0, "a zero rate must never fire"
    demanded.sum().backward()
    assert torch.isfinite(scalar.grad).all(), "zero rate poisoned the gradient"


def test_run_rates_round_trip_to_their_annual_value():
    funding = FundingParams()
    for ppy in (1, 4, 12, 52):
        escaped = (1.0 - funding.period_run_rate(ppy)) ** ppy
        assert escaped == pytest.approx(math.exp(-funding.annual_run_rate), rel=1e-12)


def test_failing_to_pay_is_a_different_death_from_being_insolvent():
    """The distinction the whole stage exists to draw.

    The shortfall lands on equity and the capital channel prices it, but a firm
    can be solvent and unable to pay -- which is the case a liquidity buffer
    exists for and the case a run produces. Two firms with the *same* closing
    equity, one of which gated withdrawals, are not in the same position.
    """
    env = run_env()
    _, met = step(env, Deploy(0.1, 40.0), lambda s: forced(s, 0.30))
    _, failed = step(env, Deploy(0.1, 1e6), lambda s: forced(s, 1.0))

    assert torch.count_nonzero(met.info["unmet_withdrawal"]) == 0
    assert torch.all(failed.info["unmet_withdrawal"] > 0)

    assert met.info["hazard"]["liquidity"].max().item() == 0.0
    assert torch.all(failed.info["hazard"]["liquidity"] > 0)
    # Failing on the whole demand is close to certain death within the quarter.
    assert torch.exp(failed.log_survival).max().item() < 0.01


def test_a_full_run_is_survivable_exactly_below_equity_over_the_haircut():
    """What stops liquidity being a complete substitute for controls, stated as
    the boundary it actually is rather than as a number that happened to work.

    Everything the firm can raise is reserves plus the discounted book,
    `E + D - hB`. Meeting a full withdrawal `D` needs `hB <= E`, so the book
    below which a run is always survivable is exactly `E / h` -- independent of
    the deposit base, which cancels.

    At a 20% haircut that boundary is 80 against equity of 16, which is where
    the leverage limit already puts the firm: no run was ever quite unmeetable,
    and the model concluded operational controls are worthless and one should
    hold cash instead. True, but only because the buffer was a complete defence.
    """
    env = run_env()
    haircut = DEFAULTS.funding.fire_sale_haircut
    boundary = DEFAULTS.firm.initial_equity / haircut

    safe = step(env, Deploy(0.0, boundary * 0.9), lambda s: forced(s, 1.0))[1]
    assert torch.count_nonzero(safe.info["unmet_withdrawal"]) == 0

    exposed = step(env, Deploy(0.0, boundary * 1.5), lambda s: forced(s, 1.0))[1]
    assert torch.all(exposed.info["unmet_share"] > 0)

    # And it is monotone in the book either side of it.
    unmet = [
        step(env, Deploy(0.0, lent), lambda s: forced(s, 1.0))[1]
        .info["unmet_share"].mean().item()
        for lent in (boundary, boundary * 1.3, boundary * 1.6)
    ]
    assert unmet[0] < unmet[1] < unmet[2], unmet


# -- the replacement ------------------------------------------------------


def test_the_asserted_depositors_leave_hazard_is_gone():
    """The claim that would silently double-count if it were wrong.

    The operational hazard asserted that an incident becomes public, depositors
    leave, and the firm therefore dies -- with no depositors in the model and no
    step between the second claim and the third. The run now carries that
    mechanism. Leaving both in place would charge the firm twice for one event,
    once through an assumed death rate and once through a withdrawal it has to
    meet, and the model would run and report plausible numbers either way.
    """
    assert DEFAULTS.hazard.annual_operational_rate == 0.0

    env = run_env()
    stock = REFERENCE.full((1, 3), 1.17)
    from quant.hazard import intensity_components

    parts = intensity_components(
        REFERENCE.full((1,), 16.0), stock, 16.0, DEFAULTS.hazard,
        DEFAULTS.alphas, DEFAULTS.firm.periods_per_year,
    )
    assert parts["operational"].item() == 0.0
    assert parts["licence"].item() > 0.0, "the licence channel is not the run and stays"
    assert parts["capital"].item() > 0.0


def test_operational_grc_still_buys_survival_through_the_balance_sheet():
    """Having removed the direct channel, the indirect one has to work: more
    operational GRC means fewer runs, fewer fire sales, and more equity left."""
    env = run_env(horizon=4)
    crn = CommonRandomNumbers(1, 4, 2048, REFERENCE)

    outcomes = []
    for g in (0.0, 0.6):
        policy = Deploy(g, 60.0)
        trajectory = env.rollout(policy, crn, differentiable=False)
        fire = sum(float(i["fire_sale_loss"].mean()) for i in trajectory.infos)
        outcomes.append(fire)
    assert outcomes[1] < outcomes[0], outcomes


def test_switching_the_cliff_off_does_not_poison_the_gradient():
    """`torch.logit(0)` is -inf with an infinite derivative, so a cliff rate of
    zero -- the natural way to switch the channel off -- returned NaN from the
    *backward* pass while the forward pass stayed correct.

    Third instance of this trap in `dynamics.py`: the compliance likelihood
    ratio at p0 = 0, then `run_probability`, and this. The run channel was
    floored when its version was found; this one was not, and the asymmetry
    survived until an ablation set the rate to zero and NaN'd from step one.
    """
    cliff = replace(DEFAULTS.cliff, period_probability=0.0)
    env = run_env(cliff=cliff)
    crn = CommonRandomNumbers(0, 1, 512, REFERENCE)
    shock = env.sampler(crn.at(0), REFERENCE)
    state = env.reset(512)

    scalar = REFERENCE.tensor(1.0, requires_grad=True)
    stock = torch.stack([scalar] * 3).expand(512, 3)
    loss = env.dynamics.cliff_loss(state, stock, shock)
    assert torch.count_nonzero(loss) == 0, "a zero rate must never fire"
    loss.sum().backward()
    assert torch.isfinite(scalar.grad).all(), "zero cliff rate poisoned the gradient"
