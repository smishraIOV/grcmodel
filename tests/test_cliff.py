"""The cliff channel, and the straight-through estimator that differentiates it.

A severe but survivable hit whose frequency GRC reduces and whose severity it
does not. The state worth most avoiding, because a moderate loss and outright
failure both leave management with few real choices while this one leaves a
decision -- rebuild, run down, or wind up -- with funding capacity impaired and
the hazard raised.

The estimator is the interesting part. `torch.bernoulli` has no gradient in its
probability, so the forward pass uses the true hard indicator and the backward
pass differentiates a tempered sigmoid of the same threshold. That is biased,
so the bias is measured against an exact answer rather than assumed small.
"""

from dataclasses import replace

import pytest
import torch

from quant.env.env import ConstantPolicy, EnvConfig, FirmEnv, evaluate
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler, cliff_fraction
from quant.frictions import exponential_mitigation
from quant.numerics import REFERENCE
from quant.params import DEFAULTS

OPERATIONAL = 1


def env_with(cliff=DEFAULTS.cliff, quarters=1, **overrides):
    return FirmEnv(
        EnvConfig.quarterly(quarters, cliff=cliff, **overrides),
        MonteCarloSampler(DEFAULTS.sampler),
    )


def probe(env, stock_value, paths=100_000, seed=3):
    """Expected cliff loss and its gradient in the operational GRC stock."""
    shock = env.sampler(CommonRandomNumbers(seed, 1, paths, REFERENCE).at(0), REFERENCE)
    state = env.reset(paths)
    stock_scalar = REFERENCE.tensor(stock_value, requires_grad=True)
    stock = torch.stack([stock_scalar] * 3).expand(paths, 3)
    loss = env.dynamics.cliff_loss(state, stock, shock).mean()
    loss.backward()
    return loss.item(), stock_scalar.grad.item()


def exact(env, stock_value, paths=100_000, seed=3):
    """The analytic mixture: p(G) x severity, exact and zero-variance.

    Affordable only because the event is binary and the horizon is one. Over T
    periods a survivable event branches 2**T, which is why this is a check
    rather than the implementation.
    """
    shock = env.sampler(CommonRandomNumbers(seed, 1, paths, REFERENCE).at(0), REFERENCE)
    state = env.reset(paths)
    stock_scalar = REFERENCE.tensor(stock_value, requires_grad=True)
    probability = exponential_mitigation(
        REFERENCE.tensor(DEFAULTS.cliff.period_probability),
        stock_scalar,
        DEFAULTS.alphas.operational,
    )
    loss = (probability * shock.cliff_fraction * torch.clamp(state.equity, min=0.0)).mean()
    loss.backward()
    return loss.item(), stock_scalar.grad.item()


def test_the_forward_pass_is_the_true_hard_indicator():
    """Straight-through means the value is exact and only the gradient is
    approximated. If the forward pass drifted, the discrete jump and its tail
    would both be wrong -- which is the whole reason the event is not simply
    replaced by its expectation."""
    env = env_with()
    for stock in (0.0, 0.65):
        relaxed, _ = probe(env, stock)
        analytic, _ = exact(env, stock)
        assert relaxed == pytest.approx(analytic, rel=0.05), stock


def test_the_gradient_is_checked_against_an_exact_answer():
    """The bias is measured, not assumed. Large-sample ratios against the exact
    mixture: 3.40 at temperature 1.0, 1.48 at 0.5, 1.11 at 0.25, 1.03 at 0.12.
    The default sits where bias has flattened and variance has not yet taken
    over.
    """
    env = env_with()
    for stock in (0.0, 0.65, 1.2):
        _, relaxed = probe(env, stock)
        _, analytic = exact(env, stock)
        assert relaxed / analytic == pytest.approx(1.0, abs=0.15), stock


def test_a_hot_relaxation_is_visibly_worse():
    """Guards the temperature choice rather than trusting it. At 1.0 the
    gradient is several times too large, which would silently overstate what
    operational GRC buys."""
    hot = env_with(replace(DEFAULTS.cliff, relaxation_temperature=1.0))
    _, biased = probe(hot, 0.65)
    _, analytic = exact(hot, 0.65)
    assert biased / analytic > 2.0


def test_the_gradient_always_points_the_right_way():
    """The property that actually matters for a descent direction. More
    operational GRC must lower expected cliff loss at every level tested."""
    env = env_with()
    for stock in (0.0, 0.3, 0.65, 1.2):
        _, gradient = probe(env, stock)
        assert gradient < 0.0, stock


def test_grc_reduces_frequency_and_not_severity():
    """The asymmetry the channel exists to express. A bridge is either drained
    or it is not; controls make the exploit less likely, they do not make it
    smaller. Modelling GRC as reducing both would let one alpha buy the same
    protection twice."""
    env = env_with()
    shock = env.sampler(CommonRandomNumbers(3, 1, 50_000, REFERENCE).at(0), REFERENCE)
    state = env.reset(50_000)

    rates, severities, errors = [], [], []
    for stock_value in (0.0, 1.2):
        stock = REFERENCE.full((50_000, 3), stock_value)
        loss = env.dynamics.cliff_loss(state, stock, shock)
        struck = loss > 0
        share = loss[struck] / state.equity[struck]
        rates.append(struck.double().mean().item())
        severities.append(share.mean().item())
        errors.append((share.std() / share.numel() ** 0.5).item())

    assert rates[1] < 0.5 * rates[0], "GRC must reduce how often it happens"

    # Compared against the sampling error rather than a fixed tolerance. At the
    # higher stock only about a hundred of fifty thousand paths are struck, so
    # the conditional mean carries roughly 10% standard error -- a flat 10%
    # tolerance here would be testing the sample size, not the model.
    spread = abs(severities[1] - severities[0])
    assert spread < 3.0 * (errors[0] ** 2 + errors[1] ** 2) ** 0.5, (
        f"severity moved with GRC: {severities[0]:.4f} -> {severities[1]:.4f}"
    )


def test_switching_the_cliff_off_restores_the_earlier_model():
    """The reduction. Every stage before this one ran without the channel, and
    their regression tests still have to mean what they meant."""
    policy = ConstantPolicy(grc=(0.3, 0.3, 0.3), investment=10.0, profile=REFERENCE)
    crn = CommonRandomNumbers(0, 4, 1024, REFERENCE)
    off = evaluate(policy, env_with(cliff=None, quarters=4), crn)
    assert off.cliff_rate == 0.0
    on = evaluate(policy, env_with(quarters=4), crn)
    assert on.cliff_rate > 0.0
    assert on.value < off.value, "a new loss channel cannot make the firm richer"


def test_a_struck_firm_cannot_fund_its_opportunity():
    """The limbo the channel exists to create.

    A cliff does not merely cost money: it impairs the balance sheet that gates
    investment, so the firm emerges alive but unable to deploy what it wants.
    Measured, underinvestment roughly doubles for struck paths against clean
    ones -- which is the state where the decision is genuinely hard, and the
    reason a moderate-or-fatal model has nothing to say about it.
    """
    env = env_with(quarters=6)
    crn = CommonRandomNumbers(0, 6, 8192, REFERENCE)
    policy = ConstantPolicy(grc=(0.3, 0.4, 0.3), investment=11.0, profile=REFERENCE)
    trajectory = env.rollout(policy, crn, False)

    struck = trajectory.infos[1]["cliff_loss"] > 0
    assert struck.sum() > 20, "too few strikes to compare against"

    later = trajectory.infos[3]["funding_binds"].double()
    assert later[struck].mean() > 1.5 * later[~struck].mean()
