"""The liability side: deposits, what they cost, and what leverage does.

Rung 0 properties for the funding stock. Until it existed the firm had no
liabilities at all -- it funded its book out of equity plus a costless multiple
of equity, settled inside the period -- which deleted three risks a bank exists
to manage: leverage, the price of funding, and the funding leaving. A hazard
channel named "depositors leave" was a label on a constant, because the model
had no depositors in it.

The tests here are about the mechanism, not the calibration. Anything that
depends on where the parameters happen to sit belongs in the regime checks in
tests/test_env.py, which is the file that goes red when a recalibration is
wrong rather than merely different.
"""

import math
from dataclasses import replace

import pytest
import torch

from quant.env.actions import FirmAction
from quant.env.env import EnvConfig, FirmEnv, standard_env
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.numerics import REFERENCE
from quant.params import DEFAULTS, WEEKLY, FundingParams

BATCH = 256


def funded_env(horizon=1, funding=None, **overrides):
    """A quarterly environment with a liability side switched on."""
    config = EnvConfig.quarterly(
        horizon, funding=funding or FundingParams(), profile=REFERENCE, **overrides
    )
    return FirmEnv(config, MonteCarloSampler(DEFAULTS.sampler))


class Deploy:
    """Spend a fixed GRC budget and ask for `investment`, which the funding
    capacity may cut down."""

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


def one_step(env, policy, seed=0, batch=BATCH):
    crn = CommonRandomNumbers(seed, 1, batch, REFERENCE)
    state = env.reset(batch)
    shock = env.sampler(crn.at(0), REFERENCE)
    return state, env.dynamics.step(state, policy(state), shock)


# -- the identity ---------------------------------------------------------


def test_accounting_identity_carries_the_funding_cost():
    """equity' = equity - grc - loss - investment + production - premium
                        - interest + reserve income - dividend.

    Elementwise, as the pre-liability identity is. The two new terms are the
    ones most likely to be silently dropped: both are small next to everything
    else in the expression, so a test written on the mean would not notice
    either going missing.
    """
    env = funded_env()
    state, result = one_step(env, Deploy(grc=0.1, investment=60.0))
    info = result.info

    expected = (
        state.equity
        - info["grc_spend"]
        - info["loss"]
        - info["investment"]
        + info["production"]
        - info["premium"]
        - info["interest"]
        + info["reserve_income"]
        - info["dividend"]
    )
    assert torch.allclose(result.state.equity, expected, atol=1e-10, rtol=0.0)


def test_idle_funding_earns_rather_than_evaporates():
    """A bank with more deposits than lending opportunities parks them.

    Without this the model punishes a firm for having a franchise: funding it
    cannot deploy is charged interest and earns nothing, so the optimal
    balance sheet is the smallest one. The spread between the two rates is what
    keeps hoarding mildly costly, which is what will make a liquidity buffer a
    decision rather than a free good.
    """
    env = funded_env()
    _, idle = one_step(env, Deploy(grc=0.1, investment=0.0))
    _, busy = one_step(env, Deploy(grc=0.1, investment=1e6))

    assert torch.all(idle.info["reserve_income"] > 0)
    assert torch.all(busy.info["reserve_income"] < idle.info["reserve_income"])
    # Idle funding still carries a negative spread -- parked, not free.
    assert torch.all(idle.info["reserve_income"] < idle.info["interest"])


def test_no_liability_side_leaves_the_old_dynamics_untouched():
    """`funding=None` is not "deposits of zero" -- it is a firm without a
    liability side at all, and it must cost nothing.

    This is what keeps the closed form and the static regression valid: they
    run on a configuration that predates this stage and must not acquire a
    funding cost from it.
    """
    env = FirmEnv(
        EnvConfig.quarterly(1, funding=None, profile=REFERENCE),
        MonteCarloSampler(DEFAULTS.sampler),
    )
    state, result = one_step(env, Deploy(grc=0.1, investment=20.0))

    assert state.deposits is None
    assert result.state.deposits is None
    assert torch.count_nonzero(result.info["interest"]) == 0
    assert torch.count_nonzero(result.info["reserve_income"]) == 0
    assert torch.equal(state.assets(), state.equity)


# -- capacity -------------------------------------------------------------


def test_funding_capacity_is_equity_plus_deposits():
    """A bank cannot lend money it has not been given.

    The pre-liability version needed a free multiple to say this; with a
    balance sheet it needs no parameter, because the balance sheet already
    states it.
    """
    env = funded_env()
    state = env.reset(BATCH)
    spend = 0.3
    fundable = state.equity - spend
    capacity = env.dynamics.funding_capacity(state, fundable)
    assert torch.allclose(capacity, fundable + state.deposits, atol=1e-12)


def test_the_firm_cannot_deploy_more_than_it_funds():
    env = funded_env()
    state, result = one_step(env, Deploy(grc=0.1, investment=1e6))
    capacity = result.info["funding_capacity"]
    assert torch.all(result.info["investment"] <= capacity + 1e-9)
    assert torch.all(result.info["funding_binds"])


def test_deposit_capacity_collapses_with_the_capital_ratio():
    """Funding withdraws as the firm weakens, and the two terms collapse
    together: a smaller multiple of a smaller number, times a market that is
    closing. A firm with no capital can carry no deposits."""
    env = funded_env()
    equity = REFERENCE.tensor([16.0, 8.0, 4.0, 1.0, 0.0])
    capacity = env.dynamics.deposit_capacity(equity)

    assert torch.all(capacity[:-1].diff() < 0), "capacity must fall as equity does"
    assert capacity[-1].item() == pytest.approx(0.0, abs=1e-12)
    # At full health the limit binds rather than market access.
    assert capacity[0].item() == pytest.approx(4.0 * 16.0, rel=1e-3)


# -- the stock ------------------------------------------------------------


def test_deposits_adjust_partially_rather_than_snapping():
    """The whole reason deposits are a state variable.

    At full adjustment the base is a deterministic function of equity, the
    liability side collapses back into the asset side, and nothing is
    outstanding that could run. The stock has to be able to sit away from
    where the firm's capital says it belongs.
    """
    env = funded_env()
    deposits = REFERENCE.full((4,), 64.0)
    equity = REFERENCE.tensor([16.0, 12.0, 8.0, 4.0])
    target = env.dynamics.deposit_capacity(equity)
    moved = env.dynamics.deposit_flow(deposits, equity)

    shrinking = slice(1, None)
    assert torch.all(moved[shrinking] < deposits[shrinking]), "must move toward target"
    assert torch.all(moved[shrinking] > target[shrinking]), "must not arrive at once"
    # And it does arrive, given enough periods.
    for _ in range(40):
        moved = env.dynamics.deposit_flow(moved, equity)
    assert torch.allclose(moved, target, atol=1e-6)


def test_a_loss_starts_the_deposit_base_shrinking_the_same_period():
    """Adjustment is evaluated on closing equity, so this quarter's losses move
    the base this quarter -- which is what makes next quarter's lending
    capacity depend on this quarter's draw. That lag is the Froot-Stein
    underinvestment channel arriving through the liability side."""
    env = funded_env()
    state, result = one_step(env, Deploy(grc=0.1, investment=40.0))
    hurt = result.state.equity < state.equity
    assert torch.any(hurt), "the test needs some paths to have lost money"
    assert torch.all(result.state.deposits[hurt] < state.deposits[hurt])


# -- the price ------------------------------------------------------------


def test_interest_is_charged_on_the_base_not_on_what_is_deployed():
    """Deposits are a liability, not a drawdown facility: money left idle is
    still money someone is owed.

    This is also what the next stage needs in order for a liquidity buffer to
    be a real decision -- reserves that cost nothing would be free insurance.
    """
    env = funded_env()
    _, busy = one_step(env, Deploy(grc=0.1, investment=60.0))
    _, idle = one_step(env, Deploy(grc=0.1, investment=0.0))
    assert torch.allclose(busy.info["interest"], idle.info["interest"], atol=1e-12)
    assert torch.all(idle.info["interest"] > 0)


def test_deposit_rates_round_trip_to_their_annual_values():
    """The conversions must mean the same thing at any decision frequency, or
    the model silently hands a weekly firm thirteen times its annual funding
    cost -- the dimensional bug of the debug notes relocated to the time axis.
    """
    funding = FundingParams()
    for ppy in (1, 4, 12, 52):
        compounded = (1.0 + funding.period_deposit_rate(ppy)) ** ppy - 1.0
        assert compounded == pytest.approx(funding.annual_deposit_rate, rel=1e-12)

        earned = (1.0 + funding.period_reserve_rate(ppy)) ** ppy - 1.0
        assert earned == pytest.approx(funding.annual_reserve_rate, rel=1e-12)

        survives = (1.0 - funding.period_adjustment(ppy)) ** ppy
        assert survives == pytest.approx(
            math.exp(-funding.annual_adjustment_speed), rel=1e-12
        )


# -- what leverage does ---------------------------------------------------


def test_leverage_amplifies_an_asset_loss_onto_equity():
    """The mechanism the model could not express before.

    A one percent loss on a book funded five-to-one against capital is a five
    percent loss of capital. This is the textbook route by which *ordinary*
    credit losses -- not exotic ones -- take an intermediary down, and with no
    liability side there was nothing in the model to carry it.

    Measured as the gap between running the same paths with and without the
    credit draw, as a fraction of opening equity, so the firm's earnings on the
    larger book cancel out of the comparison.
    """
    crn = CommonRandomNumbers(0, 1, BATCH, REFERENCE)
    policy = Deploy(grc=0.1, investment=1e6)  # take the whole balance sheet

    damage = {}
    for multiple in (0.5, 2.0, 4.0):
        env = funded_env(funding=FundingParams(deposit_capacity=multiple))
        state = env.reset(BATCH)
        shock = env.sampler(crn.at(0), REFERENCE)
        with_credit = env.dynamics.step(state, policy(state), shock).state.equity
        without = env.dynamics.step(
            state, policy(state), replace(shock, credit_loss=torch.zeros_like(shock.credit_loss))
        ).state.equity
        damage[multiple] = ((without - with_credit) / state.equity).mean().item()

    levels = [damage[m] for m in (0.5, 2.0, 4.0)]
    assert levels[0] < levels[1] < levels[2], damage
    # Roughly proportional to the balance sheet, since the loss is a rate on a
    # book that is itself the balance sheet. Loose, because GRC mitigation and
    # the capacity sigmoid both bend it.
    assert levels[2] / levels[0] == pytest.approx((1 + 4.0) / (1 + 0.5), rel=0.2)


# -- wiring ---------------------------------------------------------------


def test_observation_widens_with_a_liability_side():
    """A constant column would be worse than harmless: a bias term the network
    pays parameters to learn around. Solvers read the width from `observe`."""
    assert funded_env().observe(funded_env().reset(3)).shape[-1] == 4 + 1
    plain = FirmEnv(
        EnvConfig.quarterly(1, funding=None, profile=REFERENCE),
        MonteCarloSampler(DEFAULTS.sampler),
    )
    assert plain.observe(plain.reset(3)).shape[-1] == 4


def test_funding_without_deposits_is_a_clear_error():
    """The one way to misuse this is to hand the dynamics a state built by
    hand. Say so, rather than reading `None` as zero and reporting a confident
    number for a firm with no liabilities."""
    env = funded_env()
    state = replace(env.reset(BATCH), deposits=None)
    crn = CommonRandomNumbers(0, 1, BATCH, REFERENCE)
    with pytest.raises(ValueError, match="deposits"):
        env.dynamics.step(state, Deploy(0.1, 10.0)(state), env.sampler(crn.at(0), REFERENCE))


def test_deposits_are_frozen_when_the_path_dies():
    """Death freezes the whole state, liability side included. A dead path
    whose deposit base kept adjusting would drift, and the diagnostics average
    over frozen states."""
    env = funded_env(horizon=4)
    crn = CommonRandomNumbers(0, 4, BATCH, REFERENCE)
    # Ask for far more than it can fund and spend nothing on controls: some
    # paths run out of capital.
    trajectory = env.rollout(Deploy(grc=0.0, investment=1e6), crn, differentiable=False)
    states = trajectory.states
    for earlier, later in zip(states, states[1:]):
        dead = ~earlier.alive
        if dead.any():
            assert torch.equal(later.deposits[dead], earlier.deposits[dead])
