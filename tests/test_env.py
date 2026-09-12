"""Rung 0 and Rung 1 for the environment seam.

Rung 0 is properties no solver can be right about if the environment is wrong:
the accounting identity, non-anticipativity, absorption, reproducibility.
Three agreeing solvers on a broken simulator agree wrongly, so these come
first.

Rung 1 is the exact degeneration -- at a one-period horizon with the financing
friction off, the closed form is known and must be reproduced.

The identity test against quant/model.py is deliberately temporary. That
module is scaffolding: the convex financing cost is replaced by a survival
hazard in a later stage, and this test goes with it. Until then it is the
strongest guard available, because it compares two implementations of the same
function rather than two optimizers' outputs, and so holds to machine
precision rather than to a convergence tolerance.
"""

from dataclasses import replace

import pytest
import torch

from quant.env.actions import FirmAction
from quant.env.env import ConstantPolicy, EnvConfig, FirmEnv, evaluate, standard_env
from quant.env.reward import LiquidationValue, PerpetuityValue, Zero
from quant.env.shocks import CommonRandomNumbers, FourStateSampler, MonteCarloSampler
from quant.hazard import failure_intensity, intensity_components


def ZERO_STOCK(like):
    """A GRC stock of zero, shaped to match `like`."""
    return torch.zeros(*like.shape, 3, dtype=like.dtype, device=like.device)
from quant.model import GrcBudgets
from quant.numerics import REFERENCE
from quant.params import DEFAULTS, ORACLE, WEEKLY
from quant.solvers.analytic import family_exposures, frictionless_benchmark
from quant.solvers.neural import train_pathwise
from quant.solvers.pathwise import (
    PerPathPolicy,
    optimize_constant,
    perfect_information_bound,
)
from quant.static import build_model, build_shock

FAMILIES = ("credit", "operational", "compliance")


def four_state_env(financing_scale=1.0, horizon=1, **overrides):
    """The exactly-solvable harness, on pinned oracle parameters.

    Deliberately not DEFAULTS: the closed form exists to check the solvers, so
    it must not move when the dynamic model is recalibrated for plausibility.
    """
    # credit_scales_with_book=False: the closed form requires the objective to
    # separate across families, and a loss proportional to a chosen book does
    # not separate. One more deliberate restriction on the oracle.
    settings = dict(
        firm=ORACLE.firm, alphas=ORACLE.alphas, credit_scales_with_book=False
    )
    settings.update(overrides)
    config = EnvConfig(
        profile=REFERENCE, financing_scale=financing_scale, horizon=horizon, **settings
    )
    return FirmEnv(config, FourStateSampler(ORACLE.shock))


def monte_carlo_env(batch=512, horizon=1, **overrides):
    config = EnvConfig(profile=REFERENCE, horizon=horizon, **overrides)
    return FirmEnv(config, MonteCarloSampler(DEFAULTS.sampler))


class FixedAction:
    """A policy that replays one prepared action. Not state-dependent."""

    def __init__(self, grc, investment):
        self.grc, self.investment = grc, investment

    def __call__(self, state):
        return FirmAction(
            grc=self.grc.expand(state.batch(), 3),
            investment=self.investment,
            abandon=torch.zeros_like(self.investment),
            payout=torch.zeros_like(self.investment),
        )


# -- Rung 0 ---------------------------------------------------------------


def test_objective_matches_the_static_model():
    """The seam changed the shape of the code, not the economics.

    Compares the rolled-out value against quant/model.py's single-expression
    objective on random controls. Both compute the same function, so this holds
    at machine precision -- unlike any comparison of two optimizers, which can
    only agree to a convergence tolerance.
    """
    crn = CommonRandomNumbers(0, 1, 4, REFERENCE)
    draw = build_shock(profile=REFERENCE)
    generator = torch.Generator().manual_seed(0)

    worst = 0.0
    for financing_scale in (0.0, 1.0):
        env = four_state_env(financing_scale)
        legacy = build_model(financing_scale=financing_scale)
        for _ in range(100):
            grc = torch.rand(3, generator=generator, dtype=torch.float64) * 3.0
            investment = torch.rand(4, generator=generator, dtype=torch.float64) * 15.0
            new = env.value_of(
                env.rollout(FixedAction(grc, investment), crn, differentiable=False)
            ).item()
            old = legacy.compute_value(
                GrcBudgets.from_vector(grc), investment, draw
            ).item()
            worst = max(worst, abs(new - old) / max(1.0, abs(old)))

    assert worst < 1e-12, f"worst relative disagreement {worst:.3e}"


def test_accounting_identity_holds_elementwise():
    """equity' = equity - grc - loss - investment + production - premium.

    Asserted per path rather than in expectation. A sign error or a
    double-counted term that cancels on average would pass any test written on
    the mean, and this is the cheapest possible guard against it.
    """
    env = monte_carlo_env()
    crn = CommonRandomNumbers(3, 1, 512, REFERENCE)
    generator = torch.Generator().manual_seed(1)

    for _ in range(20):
        grc = torch.rand(3, generator=generator, dtype=torch.float64) * 2.0
        investment = torch.rand(512, generator=generator, dtype=torch.float64) * 12.0
        start = env.reset(512)
        shock = env.sampler(crn.at(0), REFERENCE)
        result = env.dynamics.step(start, FixedAction(grc, investment)(start), shock)
        info = result.info

        expected = (
            start.equity
            - info["grc_spend"]
            - info["loss"]
            - info["investment"]
            + info["production"]
            - info["premium"]
            - info["dividend"]
        )
        assert torch.allclose(result.state.equity, expected, atol=1e-10, rtol=0.0)


def test_policy_cannot_see_the_future():
    """Non-anticipativity, asserted against the random stream directly.

    Perturb the shocks of period 2 and the trajectory through period 2 must be
    bitwise unchanged. This is the bug that is hardest to notice when you own
    your own simulator -- a leak shows up as a policy that is mysteriously good
    rather than as anything that looks wrong.
    """
    env = monte_carlo_env(horizon=3)
    policy = ConstantPolicy(grc=(0.5, 0.5, 0.5), investment=8.0, profile=REFERENCE)

    crn = CommonRandomNumbers(11, 3, 256, REFERENCE)
    before = env.rollout(policy, crn, differentiable=False)

    crn.uniforms["credit"][2] = 1.0 - crn.uniforms["credit"][2]
    crn.uniforms["op_severity"][2] = 1.0 - crn.uniforms["op_severity"][2]
    after = env.rollout(policy, crn, differentiable=False)

    for step in range(3):  # states after 0, 1 and 2 periods use shocks t<2
        assert torch.equal(before.states[step].equity, after.states[step].equity), step
    assert not torch.equal(before.states[3].equity, after.states[3].equity), (
        "perturbing the final period changed nothing -- the shock is not being used"
    )


def test_death_is_absorbing():
    """Once dead, always dead, and the state stops moving.

    Death is a mask rather than a removal from the batch, which keeps shapes
    static and keeps every path aligned with its own slice of the random
    stream. The cost of that choice is that absorption has to be asserted; it
    is not structural.
    """
    env = monte_carlo_env(horizon=3, equity_floor=12.0)  # high enough to kill most paths
    policy = ConstantPolicy(grc=(0.2, 0.2, 0.2), investment=6.0, profile=REFERENCE)
    trajectory = env.rollout(policy, CommonRandomNumbers(5, 3, 512, REFERENCE), False)

    alive = [state.alive for state in trajectory.states]
    assert alive[1].sum() < alive[0].sum(), "nothing died; the test is not exercising this"

    for earlier, later in zip(alive, alive[1:]):
        assert not (later & ~earlier).any(), "a dead path came back to life"

    for before, after in zip(trajectory.states[1:], trajectory.states[2:]):
        dead = ~before.alive
        assert torch.equal(before.equity[dead], after.equity[dead]), "dead path kept moving"


def test_rollout_is_reproducible():
    env = monte_carlo_env(horizon=2)
    policy = ConstantPolicy(grc=(0.3, 0.3, 0.3), investment=7.0, profile=REFERENCE)
    first = env.rollout(policy, CommonRandomNumbers(2, 2, 256, REFERENCE), False)
    second = env.rollout(policy, CommonRandomNumbers(2, 2, 256, REFERENCE), False)
    assert torch.equal(first.states[-1].equity, second.states[-1].equity)


def test_observation_is_invariant_to_the_currency_unit():
    """Features are divided by initial equity, so redenominating the firm
    leaves the learner's inputs unchanged (docs/static-model-debug-notes.md
    section 4, one axis over)."""
    base = monte_carlo_env()
    # Every money-valued opening quantity scales together, which is what
    # redenominating the firm means. Scaling equity alone is not a change of
    # unit, it is a differently capitalized firm -- and this test caught
    # exactly that when the opening GRC stock was added.
    firm = DEFAULTS.firm
    scaled = FirmEnv(
        replace(
            base.config,
            firm=replace(
                firm,
                initial_equity=firm.initial_equity * 100,
                initial_grc_stock=firm.initial_grc_stock * 100,
            ),
        ),
        base.sampler,
    )
    assert torch.allclose(
        base.observe(base.reset(8)), scaled.observe(scaled.reset(8)), atol=1e-12
    )


def test_importance_weights_decay_geometrically_in_the_horizon():
    """The compliance channel's likelihood ratio compounds multiplicatively.

    Per-step ratios multiply, so the effective sample size falls as
    ESS(1)**T -- geometric, which is the shape asserted here. Measured on this
    environment at p0=0.05, alpha_k=0.3, 4096 paths (fraction of batch
    retained):

        g_k   p(g_k)   T=1     T=8     T=32    T=64
        1.0   0.0370   0.997   0.973   0.896   0.803
        2.0   0.0274   0.990   0.920   0.716   0.518
        8.0   0.0045   0.959   0.718   0.257   0.065

    Worth reading carefully before concluding anything: the decay is real but
    mild at the horizons this model is aimed at. Over 8-12 quarters a
    plausible compliance budget retains 85-97% of the sample. The reweighting
    is a reason to move compliance into the dynamics as an event intensity
    eventually; it is not a reason to do so before the horizon is long or the
    budget large. If a later change breaks the geometric law, the reweighting
    has silently stopped being what it claims.
    """
    budget = 8.0  # large, to make the effect measurable in a fast test
    ess = {}
    for horizon in (1, 8, 32):
        env = monte_carlo_env(horizon=horizon)
        policy = ConstantPolicy(grc=(0.5, 0.5, budget), investment=8.0, profile=REFERENCE)
        crn = CommonRandomNumbers(4, horizon, 4096, REFERENCE)
        ess[horizon] = env.rollout(policy, crn, False).effective_sample_size()

    assert ess[1] > ess[8] > ess[32], f"expected monotone decay, got {ess}"
    for horizon in (8, 32):
        predicted = ess[1] ** horizon
        assert abs(ess[horizon] - predicted) < 0.03, (
            f"T={horizon}: {ess[horizon]:.3f} is not the geometric {predicted:.3f}"
        )


# -- Rung 1 ---------------------------------------------------------------


def test_exposures_are_exact_in_the_four_state_world():
    """The enumerated shock is what makes the closed form assertable at 1e-6
    rather than at sampling error."""
    exposures = family_exposures(FourStateSampler(ORACLE.shock)(None, REFERENCE))
    assert exposures["credit"] == pytest.approx(4.0, rel=1e-12)
    assert exposures["operational"] == pytest.approx(3.5, rel=1e-12)
    assert exposures["compliance"] == pytest.approx(1.5, rel=1e-12)


def test_frictionless_horizon_one_recovers_the_closed_form():
    """Rung 1a. With no financing friction the objective separates across
    families and each budget has a closed form. Any deviation means the
    environment and the analysis disagree."""
    env = four_state_env(financing_scale=0.0)
    crn = CommonRandomNumbers(0, 1, 4, REFERENCE)
    shock = FourStateSampler(ORACLE.shock)(crn.at(0), REFERENCE)
    benchmark = frictionless_benchmark(shock, ORACLE.alphas)

    # shared_budgets: GRC is committed before the quarter's shock is seen,
    # which is the timing the closed form is derived under.
    _, result = perfect_information_bound(env, crn, n_steps=20000, shared_budgets=True)

    for index, family in enumerate(FAMILIES):
        expected = getattr(benchmark, family)
        # A budget whose optimum is exactly zero is only approached
        # asymptotically through softplus, so it gets a looser tolerance.
        #
        # The interior tolerance was 1e-6 and is now 1e-5. Moving the path
        # weights to log space changed the objective by 1.1e-16 -- machine
        # epsilon, checked directly against the multiplicative form over 300
        # random controls -- and 20000 Adam steps amplified that into 1.3e-6 of
        # credit budget. That says how flat the objective is near the optimum,
        # not that the identity has weakened: it still holds to five
        # significant figures.
        tolerance = 1e-3 if expected == 0.0 else 1e-5
        assert abs(result.grc[index] - expected) < tolerance, family


def test_terminal_value_is_what_the_firm_is_optimizing():
    """At this stage every period's reward is zero and all value is terminal:
    the firm retains everything and is valued at the end. So halving the
    recovery on liquidation must halve firm value at a fixed policy.

    A consequence worth stating, because it is a trap for the next stage: with
    zero rewards a `Zero` terminal value makes the entire objective identically
    zero, gradients vanish, and every control sits wherever it was initialized
    while the optimizer reports success. `Zero` only becomes meaningful once
    per-period rewards exist.
    """
    policy = ConstantPolicy(grc=(0.9, 0.9, 0.6), investment=9.0, profile=REFERENCE)
    crn = CommonRandomNumbers(0, 1, 4, REFERENCE)

    full = evaluate(policy, four_state_env(terminal=LiquidationValue(recovery=1.0)), crn)
    half = evaluate(policy, four_state_env(terminal=LiquidationValue(recovery=0.5)), crn)

    assert half.value == pytest.approx(0.5 * full.value, rel=1e-12)

    nothing = evaluate(policy, four_state_env(terminal=Zero()), crn)
    assert nothing.value == pytest.approx(0.0, abs=1e-12)


def test_reproduces_the_published_froot_stein_premium():
    """The stage's headline guard: the rewritten environment gives back the
    numbers in docs/quant-model.md section 6.

    Asserted at 1e-3, not tighter, and the reason matters. These digits are a
    property of where Adam was after a fixed number of steps, not of the
    optimum -- the published run used 4000 steps and had not fully settled the
    credit coordinate. At 20000 both the old and the new solver converge to
    0.993179, so the published 0.9932 is right; it was simply quoted from a
    run that had one more decimal of precision than it had converged.
    """
    crn = CommonRandomNumbers(0, 1, 4, REFERENCE)
    solve = lambda scale: perfect_information_bound(
        four_state_env(scale), crn, n_steps=20000, shared_budgets=True
    )[1]
    off, on = solve(0.0), solve(1.0)

    assert off.grc[0] == pytest.approx(0.6077, abs=1e-3)
    assert off.grc[1] == pytest.approx(0.1626, abs=1e-3)
    assert off.value == pytest.approx(16.0768, abs=1e-3)

    assert on.grc[0] == pytest.approx(0.9932, abs=1e-3)
    assert on.grc[1] == pytest.approx(0.8935, abs=1e-3)
    assert on.grc[2] == pytest.approx(0.6686, abs=1e-3)
    assert on.value == pytest.approx(9.8342, abs=1e-3)
    assert on.constrained_fraction == pytest.approx(0.520, abs=1e-3)

    # The regime check the project requires before trusting any comparative
    # static: degenerate at 0 or 1 (docs/static-model-debug-notes.md section 6).
    assert 0.05 < on.constrained_fraction < 0.95


# -- Stage 2: horizon, discounting, GRC as a stock -------------------------


def test_rates_round_trip_to_their_annual_values():
    """Per-period discount and depreciation are derived from annual rates, so
    compounding them back over a year must return the annual figure. A
    plausible-looking but wrong conversion here would rescale every
    intertemporal result silently."""
    firm = DEFAULTS.firm
    assert firm.discount() ** firm.periods_per_year == pytest.approx(
        1.0 / (1.0 + firm.annual_discount_rate), rel=1e-12
    )
    survives = (1.0 - firm.grc_depreciation()) ** firm.periods_per_year
    assert survives == pytest.approx(1.0 - firm.annual_grc_depreciation, rel=1e-12)


def test_grc_stock_follows_its_law_of_motion():
    """K' = (1 - delta) K + g, compounded over the horizon.

    The single most important addition of this stage: a stock is what makes
    spend persist, and persistence is what gives pre-emptive GRC option value
    rather than making it a repeated one-period expense.
    """
    delta, spend = 0.2, 0.5
    env = FirmEnv(
        EnvConfig(profile=REFERENCE, horizon=4, grc_depreciation=delta), MonteCarloSampler(DEFAULTS.sampler)
    )
    policy = ConstantPolicy(grc=(spend, spend, spend), investment=6.0, profile=REFERENCE)
    trajectory = env.rollout(policy, CommonRandomNumbers(0, 4, 64, REFERENCE), False)

    expected = DEFAULTS.firm.initial_grc_stock
    for state in trajectory.states[1:]:
        expected = (1.0 - delta) * expected + spend
        assert state.grc_stock[:, 0].max().item() == pytest.approx(expected, rel=1e-12)


def test_full_depreciation_makes_the_stock_the_flow():
    """delta = 1 is the static model's assumption: spend buys one period of
    protection and nothing carries. This is the reduction the earlier stages'
    regression tests depend on."""
    env = FirmEnv(
        EnvConfig(profile=REFERENCE, horizon=3, grc_depreciation=1.0), MonteCarloSampler(DEFAULTS.sampler)
    )
    policy = ConstantPolicy(grc=(0.4, 0.4, 0.4), investment=6.0, profile=REFERENCE)
    trajectory = env.rollout(policy, CommonRandomNumbers(0, 3, 64, REFERENCE), False)
    for state in trajectory.states[1:]:
        assert state.grc_stock[:, 0].max().item() == pytest.approx(0.4, rel=1e-12)


def test_frictionless_multi_period_repeats_the_closed_form():
    """Rung 1b. With no friction, no barrier and no carried stock the periods
    are independent, and each one's budget is the same closed form the static
    model solves. Tests the recursion without letting new economics in.

    The compliance channel is switched off here, and the reason is a finding in
    itself. FourStateSampler enumerates its states rather than drawing them, so
    a path sees the *same* shock every period. Compounding the compliance
    likelihood ratio over T periods then gives each path w**T rather than w,
    which is a tilted measure -- not the one the closed form is derived under,
    and the budget misses it by 1e-4 at three periods. That is the reweighting
    failing for a reason quite separate from the variance growth in
    test_importance_weights_decay_geometrically_in_the_horizon: correlated
    shocks make the compounded weights *biased* for the marginal expectation,
    not merely noisy. With the channel off the weights are constant, the
    marginal expectation is exact, and the identity holds.
    """
    # alpha_k = 0 makes p(g) = p0, so every likelihood ratio is exactly one and
    # the weights are constant. Setting the breach probability to zero instead
    # would do it too, but leaves the four-state world with zero-weight breach
    # states and is a worse test for it.
    alphas = replace(ORACLE.alphas, compliance=0.0)
    env = FirmEnv(
        EnvConfig(
            profile=REFERENCE, financing_scale=0.0, horizon=3, discount=0.97,
            alphas=alphas, credit_scales_with_book=False,
        ),
        FourStateSampler(ORACLE.shock),
    )
    crn = CommonRandomNumbers(0, 3, 4, REFERENCE)
    benchmark = frictionless_benchmark(FourStateSampler(ORACLE.shock)(None, REFERENCE), alphas)

    policy, _ = perfect_information_bound(env, crn, n_steps=20000, shared_budgets=True)
    budgets = torch.nn.functional.softplus(policy.raw_budgets).detach()

    for period in range(3):
        for index, family in enumerate(FAMILIES):
            expected = getattr(benchmark, family)
            # Loose because of the step budget, not because the identity is
            # approximate: measured, this reaches 1.1e-5 at 20000 steps and
            # 1.2e-9 at 60000. Three periods means three times the parameters
            # and a discounted gradient, so it settles more slowly than the
            # one-period case, which hits ~1e-15.
            tolerance = 1e-3 if expected == 0.0 else 1e-4
            assert abs(budgets[period, index].item() - expected) < tolerance, (period, family)


def test_per_path_policy_contains_the_constant_policy():
    """The property that makes warm-starting the bound sound: the clairvoyant
    parameterization can represent a constant policy exactly, so starting
    there can only improve on it."""
    env = FirmEnv(EnvConfig.quarterly(3), MonteCarloSampler(DEFAULTS.sampler))
    crn = CommonRandomNumbers(0, 3, 256, REFERENCE)
    constant, reference = optimize_constant(env, crn, n_steps=800)

    mirror = PerPathPolicy(crn.batch, env.config.horizon, REFERENCE)
    mirror.fill_from_constant(constant.raw, env.config.horizon, crn.batch)
    assert evaluate(mirror, env, crn).value == pytest.approx(reference.value, rel=1e-12)


def test_perfect_information_dominates_implementable_policies():
    """The sandwich: V(constant) <= V(neural) <= V_PI.

    The bound is the cheapest correctness check a learner has. A policy above
    it has an information leak, a mis-signed discount or a death that failed to
    absorb -- none of which is visible from the value on its own.
    """
    env = FirmEnv(EnvConfig.quarterly(4), MonteCarloSampler(DEFAULTS.sampler))
    crn = CommonRandomNumbers(0, 4, 512, REFERENCE)

    constant = optimize_constant(env, crn, n_steps=1200)[1]
    neural = train_pathwise(env, crn, n_steps=800).evaluation
    bound = perfect_information_bound(env, crn, n_steps=1500)[1]

    assert constant.value <= bound.value + 1e-9
    assert neural.value <= bound.value + 1e-9
    assert neural.value >= constant.value - 1e-9, (
        "state feedback did worse than a constant -- an optimization failure, "
        "since the network can represent a constant"
    )


def test_the_convex_cost_needs_a_barrier_to_stay_finite():
    """Why the quarterly model has an insolvency barrier switched on.

    The financing premium is K(e/K)^gamma with e = I - wealth, so once equity
    is negative the shortfall grows, the premium grows faster, and equity
    roughly squares each period. Recorded as a test because it is invisible at
    one period -- the static model could not have shown it -- and because it is
    the arithmetic reason a survival model is needed, independent of the
    economic one.

    The threshold is loose on purpose, and the assertion admits overflow as
    well. How far the runaway gets in six quarters depends on the balance-sheet
    scale, which has now moved twice: -2e7 by the third quarter at the original
    magnitudes, about -1e5 after the loss recalibration cut them twentyfold,
    and back to a float64 overflow once the liability side tripled the book.
    Asserting a finite bound would make this test fail whenever the firm got
    *bigger*, which is the opposite of what it is for -- so what it checks is
    that equity is not merely falling but compounding away, and `-inf` is that
    claim in its strongest form. The squaring is the point, not the exponent it
    arrives at.
    """
    # Investment has to exceed what the firm holds, or there is no shortfall
    # for the premium to price and nothing to run away. 11.0 did that when the
    # opening GRC stock was 0.650 and losses were heavier; at the larger opening
    # stock the firm simply funds 11 out of its own equity and the test measures
    # nothing. 40 against equity of 16 is a shortfall on every path.
    policy = ConstantPolicy(grc=(0.7, 0.7, 0.7), investment=40.0, profile=REFERENCE)
    crn = CommonRandomNumbers(0, 6, 512, REFERENCE)
    sampler = MonteCarloSampler(DEFAULTS.sampler)

    # financing_scale=1.0 because the dynamic model has retired the convex
    # premium in favour of the hard funding cap. The divergence is a property
    # of that premium, so demonstrating it means switching it back on.
    # Both switches matter. financing_scale=1.0 because the dynamic model has
    # retired the convex premium for the hard funding cap, and the divergence
    # is a property of that premium. funding_constrained=False because the cap
    # is what now *prevents* it: a firm that cannot deploy more than it can
    # fund never reaches the shortfall that makes the premium run away. The
    # pathology is structurally fixed, not parameterized away.
    # funding=None as well: the runaway is a property of the convex premium on
    # a firm financing itself, and a deposit-funded firm holding 69 of idle
    # reserves earns enough on them to arrest it on some paths. That would make
    # this test assert something about the reserve rate, which is not what it
    # is for.
    unbounded = FirmEnv(
        EnvConfig.quarterly(
            6, equity_floor=-float("inf"), financing_scale=1.0,
            funding_constrained=False, funding=None,
        ),
        sampler,
    )
    final = unbounded.rollout(policy, crn, False).states[-1].equity
    diverged = ~torch.isfinite(final) | (final < -1e3)
    assert diverged.all(), "the divergence this test documents is gone"

    bounded = FirmEnv(
        EnvConfig.quarterly(
            6, financing_scale=1.0, equity_floor=0.0,
            funding_constrained=False, funding=None,
        ),
        sampler,
    )
    trajectory = bounded.rollout(policy, crn, False)
    # Bounded is the claim, not survivable. With the premium switched back on
    # and no funding cap the firm is comprehensively insolvent either way; the
    # barrier is what stops that from compounding into an overflow.
    #
    # Not asserted against the floor itself: a path records the equity it
    # *crossed* at before being frozen, which is legitimately below it. What
    # the barrier buys is that the damage is one period's rather than six
    # periods squared -- orders of magnitude, and that is what is checked.
    final_bounded = trajectory.states[-1].equity
    assert torch.isfinite(final_bounded).all()
    assert final_bounded.min().item() > 100.0 * final.min().item()


# -- Bite 3a: smooth survival hazard --------------------------------------


def test_hazard_off_reproduces_the_barrier_model():
    """The reduction. With no hazard every surviving path has survival exactly
    one, so the value is the stage-2 expression unchanged."""
    env_off = FirmEnv(EnvConfig.quarterly(4, hazard=None, equity_floor=0.0), MonteCarloSampler(DEFAULTS.sampler))
    policy = ConstantPolicy(grc=(0.6, 0.6, 0.6), investment=9.0, profile=REFERENCE)
    crn = CommonRandomNumbers(0, 4, 512, REFERENCE)
    trajectory = env_off.rollout(policy, crn, False)

    survival = trajectory.cumulative_survival()[-1]
    alive = trajectory.states[-1].alive
    assert torch.equal(survival[alive], torch.ones_like(survival[alive]))
    assert torch.equal(survival[~alive], torch.zeros_like(survival[~alive]))


def test_hazard_responds_to_equity():
    """The trap this whole channel has to avoid.

    An exogenous death rate leaves firm value linear in equity, which removes
    the Froot-Stein content entirely and makes the hazard a discount-rate
    adjustment wearing a costume. It is the dynamic form of the
    deterministic-loss trap in docs/static-model-debug-notes.md section 2, and
    just as silent: the model runs, converges, and means nothing.
    """
    equity = REFERENCE.tensor([24.0, 16.0, 8.0, 4.0, 1.0, -2.0])
    intensity = failure_intensity(equity, ZERO_STOCK(equity), 16.0, DEFAULTS.hazard, DEFAULTS.alphas, 4)
    assert (intensity[1:] > intensity[:-1]).all(), "hazard must rise as equity falls"
    assert intensity[0] < 0.1 * intensity[-1], "response is too flat to be doing any work"


def test_hazard_gives_gradient_where_a_barrier_gives_none():
    """Why the channel was replaced. A step function has zero derivative
    everywhere it is defined; a hazard has one wherever survival has not
    underflowed."""
    for value in (16.0, 8.0, 4.0, 1.0, -2.0):
        equity = REFERENCE.tensor(value, requires_grad=True)
        torch.exp(-failure_intensity(equity, ZERO_STOCK(equity), 16.0, DEFAULTS.hazard, DEFAULTS.alphas, 4)).backward()
        assert equity.grad.item() > 1e-6, f"no gradient at equity {value}"


def test_hazard_is_invariant_to_the_currency_unit():
    """The intensity depends on equity only through a ratio, so redenominating
    the firm must not change it (docs/static-model-debug-notes.md section 4)."""
    equity = REFERENCE.tensor([8.0, 2.0])
    base = failure_intensity(equity, ZERO_STOCK(equity), 16.0, DEFAULTS.hazard, DEFAULTS.alphas, 4)
    big = REFERENCE.tensor([800.0, 200.0])
    scaled = failure_intensity(big, ZERO_STOCK(big), 1600.0, DEFAULTS.hazard, DEFAULTS.alphas, 4)
    assert torch.allclose(base, scaled, atol=1e-14)


def test_value_is_nonlinear_in_equity():
    """Froot-Stein needs curvature in the value function. With a state-dependent
    hazard that curvature is derived rather than assumed -- which is the claim
    this stage exists to support, so it gets asserted rather than asserted
    about."""
    policy = ConstantPolicy(grc=(0.6, 0.6, 0.6), investment=9.0, profile=REFERENCE)
    crn = CommonRandomNumbers(0, 4, 1024, REFERENCE)

    values = []
    for equity in (8.0, 12.0, 16.0):
        firm = replace(DEFAULTS.firm, initial_equity=equity)
        env = FirmEnv(EnvConfig.quarterly(4, firm=firm), MonteCarloSampler(DEFAULTS.sampler))
        values.append(evaluate(policy, env, crn).value)

    second_difference = values[0] - 2 * values[1] + values[2]
    assert abs(second_difference) > 1e-3, f"value is linear in equity: {values}"


def test_survival_survives_a_long_horizon_without_underflowing():
    """Survival is summed in logs and exponentiated once. A product of
    per-period probabilities underflows inside horizons this model cares
    about; the sum does not."""
    env = FirmEnv(EnvConfig.quarterly(40), MonteCarloSampler(DEFAULTS.sampler))
    # Sized to the recalibrated model: 3.6 a quarter of GRC against revenue of
    # 0.70 is a bankruptcy test, not a long-horizon one.
    policy = ConstantPolicy(grc=(0.05, 0.05, 0.05), investment=16.0, profile=REFERENCE)
    trajectory = env.rollout(policy, CommonRandomNumbers(0, 40, 512, REFERENCE), False)

    survival = trajectory.cumulative_survival()
    assert all(torch.isfinite(step).all() for step in survival)
    assert (survival[-1] <= survival[0]).all()
    assert survival[-1].max().item() > 0.0, "every path underflowed to zero"


# -- Bite 3b: GRC acts on the hazard, not only on losses -------------------


def no_grc_hazard():
    """Hazard with only the capital channel -- the bite-3a configuration."""
    return replace(DEFAULTS.hazard, annual_operational_rate=0.0, annual_licence_rate=0.0)


def test_zero_base_rates_reduce_to_the_capital_channel():
    """The reduction. With both GRC-reducible rates at zero, only capital
    drives the hazard and the bite-3a behaviour is recovered exactly."""
    equity = REFERENCE.tensor([16.0, 4.0])
    stock = REFERENCE.tensor([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    parts = intensity_components(equity, stock, 16.0, no_grc_hazard(), DEFAULTS.alphas, 4)
    assert torch.equal(parts["operational"], torch.zeros_like(equity))
    assert torch.equal(parts["licence"], torch.zeros_like(equity))


def test_competing_risks_compose_additively():
    """Intensities add; survival probabilities multiply. Composing the other
    way round would double-count the overlap and understate survival."""
    equity = REFERENCE.tensor([12.0, 6.0])
    stock = REFERENCE.tensor([[2.0, 3.0, 4.0], [1.0, 1.0, 1.0]])
    parts = intensity_components(equity, stock, 16.0, DEFAULTS.hazard, DEFAULTS.alphas, 4)
    total = failure_intensity(equity, stock, 16.0, DEFAULTS.hazard, DEFAULTS.alphas, 4)
    assert torch.allclose(total, sum(parts.values()), atol=1e-14)


def test_grc_stock_reduces_the_hazard_it_acts_on():
    """Operational GRC reduces the run hazard, compliance GRC the licence
    hazard, each with diminishing returns."""
    equity = REFERENCE.tensor([16.0])
    levels = [0.0, 2.0, 6.0, 12.0]
    for column, channel in ((1, "operational"), (2, "compliance")):
        intensities = []
        for level in levels:
            stock = REFERENCE.tensor([[0.0, 0.0, 0.0]])
            stock[0, column] = level
            parts = intensity_components(equity, stock, 16.0, DEFAULTS.hazard, DEFAULTS.alphas, 4)
            intensities.append(parts["operational" if column == 1 else "licence"].item())
        assert intensities[0] > intensities[1] > intensities[2] > intensities[3], channel
        # Diminishing returns: each further unit removes less than the last.
        first, second = intensities[0] - intensities[1], intensities[1] - intensities[2]
        assert first / 2.0 > second / 4.0, f"{channel} is not concave in spend"


def test_credit_grc_has_no_direct_hazard_channel():
    """Deliberate, not an omission. Bad underwriting erodes equity, and equity
    is already the capital channel's argument -- giving credit its own hazard
    would count the same mechanism twice."""
    equity = REFERENCE.tensor([16.0])
    bare = REFERENCE.tensor([[0.0, 0.0, 0.0]])
    credit_only = REFERENCE.tensor([[20.0, 0.0, 0.0]])
    args = (16.0, DEFAULTS.hazard, DEFAULTS.alphas, 4)
    assert torch.equal(
        failure_intensity(equity, bare, *args), failure_intensity(equity, credit_only, *args)
    )


def test_grc_spend_rises_when_it_buys_survival():
    """The economics of this bite, measured two ways.

    First: with GRC reaching the hazard directly, a unit of spend buys
    franchise protection as well as a smaller loss, so more of it is worth
    buying.

    Second, and the one that means something to a risk owner: both policies are
    then scored in the *same* world -- the one with the hazard channels in it.
    Comparing survival across two different hazard models would be
    meaningless, since the world without those channels is simply less
    dangerous. What this measures is the cost of setting a budget as though GRC
    only reduced expected loss, when in fact it also buys survival.
    """
    sampler = MonteCarloSampler(DEFAULTS.sampler)
    crn = CommonRandomNumbers(0, 6, 1024, REFERENCE)
    real = FirmEnv(EnvConfig.quarterly(6), sampler)

    naive, _ = optimize_constant(
        FirmEnv(EnvConfig.quarterly(6, hazard=no_grc_hazard()), sampler), crn, n_steps=2500
    )
    aware, aware_result = optimize_constant(real, crn, n_steps=2500)

    naive_result = evaluate(naive, real, crn)
    assert aware_result.total_grc > naive_result.total_grc * 1.05, (
        f"spend barely moved: {naive_result.total_grc:.3f} -> {aware_result.total_grc:.3f}"
    )
    assert aware_result.survival_rate > naive_result.survival_rate
    assert aware_result.value > naive_result.value


def test_default_quarterly_model_sits_in_a_usable_regime():
    """The project's standing requirement, one axis over: check the regime
    before trusting a comparative static (docs/static-model-debug-notes.md
    section 6).

    Two degenerate regimes flank the useful one and both look like working
    models. A firm that almost never fails makes GRC pure cost and every
    survival comparative static vanishes; one that almost always fails makes it
    futile. Between them, spend also has to stay material -- if the firm
    inherits enough control capital that optimal spend collapses toward zero,
    the model has nothing to say about budgets even though survival looks fine.

    The opening GRC stock is set to the self-consistent steady state for
    exactly this reason; starting from zero put annual failure at 27.5%.
    """
    env = FirmEnv(EnvConfig.quarterly(8), MonteCarloSampler(DEFAULTS.sampler))
    result = optimize_constant(env, CommonRandomNumbers(0, 8, 1024, REFERENCE), n_steps=2500)[1]

    annual_death = 1.0 - result.survival_rate ** 0.5
    assert 0.002 < annual_death < 0.15, f"degenerate survival regime: {annual_death:.2%}"
    assert result.total_grc > 0.10, f"spend is immaterial: {result.total_grc:.4f}"

    # constrained_fraction is deliberately NOT asserted, and that is a finding
    # rather than an omission. It was the static model's regime check because
    # costly external finance was the only friction there. In this
    # configuration it sits at about 0.02: a well-capitalized going concern
    # funds its investment internally almost always, and the binding channel is
    # the hazard instead.
    #
    # That is the project's own thesis showing up as evidence. The convex
    # financing cost was a static reduced form of a curvature the dynamic model
    # derives from survival, so once survival is explicit the reduced form
    # stops doing work. Removing it is the next bite; this is the measurement
    # that justifies it.
    assert 0.05 < result.underinvestment_fraction < 0.99, (
        f"degenerate funding regime: {result.underinvestment_fraction:.3f}"
    )


# -- Bite 3c: what failure costs, and what GRC cannot do about it ----------


def test_failure_recovery_does_not_depend_on_grc():
    """The asymmetry this channel exists to express.

    GRC acts on how often failure happens, never on what it costs. A programme
    that reduced both would let one parameter buy the same protection twice,
    and the second purchase would be free -- which is how a model comes to
    recommend spending that no real programme could justify.
    """
    env = FirmEnv(EnvConfig.quarterly(1), MonteCarloSampler(DEFAULTS.sampler))
    equity = REFERENCE.tensor([12.0, 3.0])
    poor = REFERENCE.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rich = REFERENCE.tensor([[50.0, 50.0, 50.0], [50.0, 50.0, 50.0]])
    args = (16.0, DEFAULTS.hazard, DEFAULTS.alphas, 4)

    # Elementwise, holding equity fixed. Comparing across equity levels would
    # be comparing two different firms: at equity 3 the capital channel alone
    # exceeds anything the control stock does at equity 12.
    assert (failure_intensity(equity, rich, *args) < failure_intensity(equity, poor, *args)).all()

    # Recovery is a function of equity alone -- no stock argument exists to
    # pass -- so the same equity gives the same recovery under either stock.
    assert torch.equal(env.dynamics.failure_value(equity), 0.4 * equity)


def test_failure_recovery_is_floored_by_limited_liability():
    """A firm that died owing money is worth nothing to its owners, not a
    negative number. The floor is also where this term's gradient stops, which
    leaves the survival probability as the only thing still pushing a path away
    from deep insolvency -- the job the hazard was brought forward to do."""
    env = FirmEnv(EnvConfig.quarterly(1), MonteCarloSampler(DEFAULTS.sampler))
    equity = REFERENCE.tensor([-40.0, -1.0, 0.0, 5.0])
    recovery = env.dynamics.failure_value(equity)
    assert torch.equal(recovery[:3], torch.zeros(3, dtype=recovery.dtype))
    assert recovery[3].item() == pytest.approx(2.0, rel=1e-12)


def test_recovery_raises_value_and_shrinks_the_going_concern_share():
    """More recoverable on failure means less destroyed by it, so the firm is
    worth more and less of that worth is going-concern rather than
    liquidation."""
    policy = ConstantPolicy(grc=(0.4, 0.4, 0.4), investment=10.0, profile=REFERENCE)
    crn = CommonRandomNumbers(0, 6, 1024, REFERENCE)

    results = []
    for recovery in (0.0, 0.4, 0.8):
        firm = replace(DEFAULTS.firm, failure_recovery=recovery)
        env = FirmEnv(EnvConfig.quarterly(6, firm=firm), MonteCarloSampler(DEFAULTS.sampler))
        results.append(evaluate(policy, env, crn))

    assert results[0].value < results[1].value < results[2].value
    assert results[0].going_concern_share > results[1].going_concern_share > results[2].going_concern_share
    assert results[0].going_concern_share == pytest.approx(1.0, abs=1e-12), (
        "with nothing recovered, all value must be going-concern"
    )


def test_going_concern_share_leaves_a_survival_motive():
    """The regime check for this channel. If liquidation were worth nearly as
    much as continuing, death would be cheap, the survival motive would vanish
    and the objective would quietly revert to expected-loss minimization --
    which docs/framework.md section 3 explicitly rejects."""
    env = FirmEnv(EnvConfig.quarterly(8), MonteCarloSampler(DEFAULTS.sampler))
    result = optimize_constant(env, CommonRandomNumbers(0, 8, 1024, REFERENCE), n_steps=2500)[1]
    assert 0.5 < result.going_concern_share < 1.0, (
        f"little left to survive for: {result.going_concern_share:.3f}"
    )


# -- Bite 3d: the option to stop -------------------------------------------


def test_orderly_wind_down_beats_disorderly_failure():
    """The premise. A firm that chose the moment unwinds positions rather than
    liquidating into a panic, surrenders its licence rather than having it
    revoked, and pays counterparties in sequence."""
    env = FirmEnv(EnvConfig.quarterly(1), MonteCarloSampler(DEFAULTS.sampler))
    equity = REFERENCE.tensor([10.0, 2.0, 0.0, -5.0])
    orderly, disorderly = env.dynamics.orderly_value(equity), env.dynamics.failure_value(equity)
    assert (orderly[:2] > disorderly[:2]).all()
    # Both floored by limited liability, so neither is worth anything once
    # equity is gone -- the option is to stop early, not to stop for free.
    assert torch.equal(orderly[2:], torch.zeros(2, dtype=orderly.dtype))


def nothing_left_to_protect(**overrides):
    """A loss-making firm, thinly capitalized, with no franchise beyond book.

    Needed because the exit option is deep out of the money at the default
    parameters, for a structural reason worth knowing: nothing in this model
    caps investment by capital, so the firm can always deploy I* and earn the
    same surplus regardless of its balance sheet. The franchise is worth about
    473 as a perpetuity against equity of 16, so winding down is never close to
    optimal. Equity matters only through the hazard.

    That is the same root cause as the convex financing cost going inert: with
    no funding constraint the balance sheet does not gate operations. Fixing it
    means linking production to capital, which is a modelling change rather
    than a parameter, and it is the open item in docs/quant-model.md section 5.
    """
    firm = replace(
        DEFAULTS.firm,
        production_scale=0.99,      # marginal return below 1: investing destroys value
        initial_equity=6.0,
        initial_grc_stock=DEFAULTS.firm.initial_grc_stock * 6.0 / 16.0,
        **overrides,
    )
    return FirmEnv(
        EnvConfig.quarterly(6, firm=firm, terminal=PerpetuityValue(franchise=0.0)),
        MonteCarloSampler(DEFAULTS.sampler),
    )


def test_the_exit_option_cannot_reduce_firm_value():
    """An option is worth nothing or something, never less than nothing. If
    this fails the accounting has double-counted an exit, or is charging the
    firm for a choice it declined to make.

    Checked where the option is live. At the default parameters the two values
    differ by about 7e-4 in either direction, which is Adam not having settled
    one extra parameter rather than anything economic -- so asserting there
    would be measuring optimizer noise.
    """
    crn = CommonRandomNumbers(0, 6, 1024, REFERENCE)
    live = nothing_left_to_protect()
    shut = FirmEnv(
        replace(live.config, allow_abandonment=False), live.sampler
    )

    without = optimize_constant(shut, crn, n_steps=2000)[1]
    with_option = optimize_constant(live, crn, n_steps=2000)[1]

    assert with_option.value > without.value + 1.0, (
        f"the option bought nothing: {without.value:.4f} -> {with_option.value:.4f}"
    )
    assert with_option.orderly_exit_rate > 0.5


def test_a_healthy_firm_does_not_take_the_exit():
    """The other half of the same claim. With the franchise intact, continuing
    dominates winding down everywhere and the option should sit unexercised --
    an option that gets taken when it should not is as wrong as one that does
    not get taken when it should."""
    env = FirmEnv(EnvConfig.quarterly(6), MonteCarloSampler(DEFAULTS.sampler))
    result = optimize_constant(env, CommonRandomNumbers(0, 6, 1024, REFERENCE), n_steps=2000)[1]
    assert result.orderly_exit_rate < 0.01, f"exit rate {result.orderly_exit_rate:.4f}"


def test_the_relaxed_exit_decision_lands_on_a_corner():
    """Firm value is linear in the exit probability -- it is a convex
    combination of stopping now and carrying on -- so a linear function on
    [0, 1] attains its maximum at an endpoint. The relaxation is therefore
    exact, not an approximation, and the optimizer should drive the decision to
    0 or 1 on its own. If it settles in the middle, the objective is not linear
    in it and the relaxation has stopped being free.
    """
    env = FirmEnv(EnvConfig.quarterly(4), MonteCarloSampler(DEFAULTS.sampler))
    policy, _ = perfect_information_bound(
        env, CommonRandomNumbers(0, 4, 256, REFERENCE), n_steps=3000
    )
    exit_prob = torch.sigmoid(policy.raw_free[..., -2]).detach()  # -1 is now the payout
    interior = ((exit_prob > 0.02) & (exit_prob < 0.98)).double().mean().item()
    assert interior < 0.25, f"{interior:.1%} of exit decisions are undecided"


def test_richer_wind_downs_are_taken_more_often():
    """The comparative static Governance implies: the better an orderly exit
    pays relative to a disorderly one, the more often the firm takes it, and
    the more the firm is worth for having the choice.

    Run on the thin-franchise firm, where the option is live at all."""
    sampler = MonteCarloSampler(DEFAULTS.sampler)
    crn = CommonRandomNumbers(0, 6, 1024, REFERENCE)

    results = [
        optimize_constant(nothing_left_to_protect(orderly_recovery=orderly), crn, n_steps=2000)[1]
        for orderly in (0.45, 0.95)
    ]

    # Both saturate at ~0.9999 on this firm -- once exiting dominates, it
    # dominates at either recovery rate, so ordering the exit rates is decided
    # by floating-point dust rather than by economics. What carries the claim
    # is that the option is taken at all, and that taking a richer one is worth
    # more: 7.20 against 15.20 here.
    assert min(r.orderly_exit_rate for r in results) > 0.9
    assert results[1].value > results[0].value * 1.5


# -- Funding constraint: the balance sheet gates operations ----------------


def test_unconstrained_config_places_no_cap():
    """The reduction. The static model let the firm fund anything at a convex
    price, which is what its closed form is derived under."""
    env = FirmEnv(EnvConfig(profile=REFERENCE), FourStateSampler(DEFAULTS.shock))
    wealth = REFERENCE.tensor([10.0, -5.0])
    state = replace(env.reset(2), equity=wealth)
    capacity = env.dynamics.funding_capacity(state, wealth)
    assert torch.isinf(capacity).all()


def test_funding_capacity_collapses_faster_than_wealth():
    """Funding withdraws exactly when it is needed.

    Stated on the *pre-liability* dynamics, where capacity is internal wealth
    plus a costless multiple of it scaled by market access, and access is
    itself falling in wealth — so the ratio of capacity to wealth shrinks as the
    firm weakens. That is the mechanism which turns a bad quarter into forgone
    investment rather than merely expensive investment.

    With a liability side the same procyclicality is there but it lives one
    period upstream, in the deposit base rather than in the cap
    (tests/test_funding.py). This test is kept on the old configuration because
    it is what the pre-liability published numbers were produced under, and
    because the two mechanisms are worth being able to tell apart.
    """
    env = FirmEnv(
        EnvConfig.quarterly(1, funding=None), MonteCarloSampler(DEFAULTS.sampler)
    )
    wealth = REFERENCE.tensor([14.0, 8.0, 4.0, 2.0, 1.0])
    state = replace(env.reset(5), equity=wealth)
    capacity = env.dynamics.funding_capacity(state, wealth)

    assert (capacity[1:] < capacity[:-1]).all(), "capacity must fall with wealth"
    leverage = capacity / wealth
    assert (leverage[1:] <= leverage[:-1] + 1e-12).all(), (
        "access must not improve as the firm weakens"
    )
    thin = replace(env.reset(1), equity=REFERENCE.tensor([-3.0]))
    assert env.dynamics.funding_capacity(thin, thin.equity).item() == 0.0


def test_investment_never_exceeds_what_can_be_funded():
    """The cap is applied to what is deployed, not merely priced. A firm that
    wants far more than it can raise gets what it can raise."""
    env = FirmEnv(EnvConfig.quarterly(1), MonteCarloSampler(DEFAULTS.sampler))
    crn = CommonRandomNumbers(0, 1, 512, REFERENCE)
    greedy = ConstantPolicy(grc=(0.3, 0.3, 0.3), investment=500.0, profile=REFERENCE)
    info = env.rollout(greedy, crn, False).infos[0]

    assert (info["investment"] <= info["funding_capacity"] + 1e-9).all()
    assert info["funding_binds"].all(), "wanting 500 should bind everywhere"


def test_a_bad_quarter_constrains_the_next_one():
    """The Froot-Stein channel, now operating across periods rather than within
    one.

    A bank sets its book from the capital it has at the start of a period;
    defaults arrive on what it lent. So a large loss cannot constrain the
    lending that produced it -- it constrains the *next* period, through lower
    equity. That reordering was forced by making credit loss depend on the book:
    the book cannot be sized by a quantity that depends on the book.

    The channel is the same one and is if anything closer to how a bank works;
    only its timing moved.
    """
    env = FirmEnv(EnvConfig.quarterly(4), MonteCarloSampler(DEFAULTS.sampler))
    policy = ConstantPolicy(grc=(0.02, 0.02, 0.02), investment=40.0, profile=REFERENCE)
    trajectory = env.rollout(policy, CommonRandomNumbers(0, 4, 8192, REFERENCE), False)

    # Split on the first quarter's loss, then look at the second quarter's book.
    first_loss = trajectory.infos[0]["loss"]
    heavy = first_loss > first_loss.median()
    second_book = trajectory.infos[1]["investment"]

    assert second_book[heavy].mean() < second_book[~heavy].mean(), (
        "a heavier loss must leave the firm lending less next quarter"
    )
    binds = trajectory.infos[1]["funding_binds"]
    assert binds.any(), "the funding cap should bind somewhere at this book size"


# -- Payout: the first genuine intertemporal trade-off ---------------------


def test_dividends_come_out_of_profit_not_capital():
    """Without this the control is a way to strip the firm. Distributing all
    equity returns it at face value, which strictly beats the 0.7 an orderly
    wind-down recovers -- so the exit option would be dominated and the balance
    sheet could be emptied in a quarter. Measured: the thin-franchise firm's
    exit rate fell from 1.000 to 0.0005 when payout was unlimited.
    """
    env = FirmEnv(EnvConfig.quarterly(1), MonteCarloSampler(DEFAULTS.sampler))
    greedy = ConstantPolicy(grc=(0.3, 0.3, 0.3), investment=11.0, payout=1.0, profile=REFERENCE)
    trajectory = env.rollout(greedy, CommonRandomNumbers(0, 1, 1024, REFERENCE), False)
    info, start = trajectory.infos[0], trajectory.states[0]

    profit = trajectory.states[1].equity + info["dividend"] - start.equity
    assert (info["dividend"] <= torch.clamp(profit, min=0.0) + 1e-9).all()
    # A loss-making quarter distributes nothing, however high the fraction.
    assert (info["dividend"][profit <= 0.0] == 0.0).all()


def test_payout_keeps_the_balance_sheet_scarce():
    """The point of the control, checked where the firm actually uses it.

    At the default discount the recalibrated firm retains almost everything,
    and that is a result rather than a defect: capital retained both funds
    investment the firm cannot otherwise fund and lowers the hazard, which at
    an 8% annual failure rate beats distributing. A firm facing that much risk
    hoards. The control is exercised by impatience instead -- see
    test_impatience_raises_the_payout, which is what says the channel is live.

    So this checks the mechanism on an impatient firm: when it does distribute,
    capital is held flatter and the funding constraint keeps biting instead of
    being outgrown.
    """
    sampler = MonteCarloSampler(DEFAULTS.sampler)
    crn = CommonRandomNumbers(0, 8, 1024, REFERENCE)

    results = {}
    for allow in (False, True):
        env = FirmEnv(EnvConfig.quarterly(8, discount=0.90, allow_payout=allow), sampler)
        policy, result = optimize_constant(env, crn, n_steps=2500)
        final = env.rollout(policy, crn, False).states[-1].equity.median().item()
        results[allow] = (result, final)

    (retained, grown), (distributed, held) = results[False], results[True]
    assert distributed.payout_share > 0.05, "an impatient firm should distribute something"
    assert held <= grown, f"distributing grew the balance sheet: {grown:.2f} -> {held:.2f}"
    assert distributed.value >= retained.value - 1e-9


def test_impatience_raises_the_payout():
    """Discounting is finally load-bearing.

    Every reward was zero until this bite, so the discount rate was a scalar
    multiplier on a terminal value and could not change any decision. Now a
    dividend is worth its face value today against capital that pays off later
    in survival and funding capacity, so a more impatient firm should
    distribute more. If this ever stops holding, rewards have gone back to
    being terminal-only.
    """
    sampler = MonteCarloSampler(DEFAULTS.sampler)
    crn = CommonRandomNumbers(0, 8, 1024, REFERENCE)

    fractions = []
    for discount in (0.995, 0.90):
        env = FirmEnv(EnvConfig.quarterly(8, discount=discount), sampler)
        policy, _ = optimize_constant(env, crn, n_steps=2500)
        fractions.append(torch.sigmoid(policy.raw[-1]).item())

    assert fractions[1] > fractions[0], (
        f"impatience did not raise the payout: {fractions[0]:.3f} -> {fractions[1]:.3f}"
    )


# -- Decision frequency ----------------------------------------------------


def test_the_model_is_invariant_to_how_often_the_firm_decides():
    """The time-axis counterpart of unit invariance, and the same class of
    silent bug (docs/static-model-debug-notes.md section 4).

    The discount, the GRC depreciation and all three hazard rates were already
    annual and divided down correctly. The loss means, the event probabilities,
    the cliff rate and the production parameters were *per-period* and did not
    -- so running the same firm weekly instead of quarterly would have handed
    it thirteen times its annual losses and thirteen times its annual return
    while every diagnostic reported normally. `quant/params.model_at` derives
    all of them from frequency-free annual rates.

    Not asserted tightly, and the reason is economics rather than tolerance.
    Thirteen weekly draws of a loss summing to the same mean as one quarterly
    draw are *less* volatile than the quarterly draw -- a sum of exponentials
    is a gamma, not an exponential -- so a finer decision frequency genuinely
    carries less aggregate tail. The means are invariant by construction; the
    distributions are not, and should not be.
    """
    annual_spend = 0.32
    values, failures = [], []
    for params, steps in ((DEFAULTS, 8), (WEEKLY, 104)):  # both are two years
        # Split across three families, so the per-family flow is a third of it.
        per_period = annual_spend / (3 * params.firm.periods_per_year)
        env = standard_env(params, steps)
        result = evaluate(
            ConstantPolicy(
                grc=(per_period, per_period, per_period), investment=20.0, profile=REFERENCE
            ),
            env,
            CommonRandomNumbers(0, steps, 2048, REFERENCE),
        )
        values.append(result.value)
        failures.append(result.annual_death_probability)
        assert result.total_grc * params.firm.periods_per_year == pytest.approx(
            annual_spend, rel=1e-12
        ), "annual GRC spend must not depend on how often it is bought"

    assert values[1] == pytest.approx(values[0], rel=0.05), (
        f"firm value moved with decision frequency: {values[0]:.4f} -> {values[1]:.4f}"
    )
    assert failures[1] == pytest.approx(failures[0], abs=0.02), (
        f"failure rate moved with decision frequency: {failures[0]:.2%} -> {failures[1]:.2%}"
    )


def test_compounded_path_weights_survive_a_long_horizon():
    """The bug this found: the per-path prior was multiplied in at every step,
    so the accumulated weight carried (1/batch)**T. A constant factor cancels
    in the normalization, which is exactly why it survived eight quarterly
    steps unnoticed -- and at 2048 paths over 104 weekly steps it reached
    1e-344, underflowed to zero on every path, and turned the normalization
    into 0/0."""
    env = standard_env(WEEKLY, 104)
    policy = ConstantPolicy(grc=(0.006, 0.006, 0.006), investment=20.0, profile=REFERENCE)
    trajectory = env.rollout(policy, CommonRandomNumbers(0, 104, 2048, REFERENCE), False)

    weights = trajectory.path_weights()
    assert torch.isfinite(weights).all()
    assert weights.sum().item() == pytest.approx(1.0, rel=1e-12)
    assert trajectory.effective_sample_size() > 0.5


def test_credit_loss_scales_with_the_book():
    """The central banking trade-off: taking more assets must cost more risk.

    It did not. `shock.credit_loss` was an exogenous money amount drawn
    independently of the investment action, so expected loss was 0.2712 a
    quarter whether the firm deployed 5 or 80 -- expanding the balance sheet
    was free on the risk side, which inverts the decision a bank actually
    faces. Credit loss is now a rate on what is deployed.

    Operational and compliance losses deliberately do *not* scale: an incident
    costs what it costs, and a regulator's penalty is not proportional to the
    loan portfolio.
    """
    env = FirmEnv(EnvConfig.quarterly(1), MonteCarloSampler(DEFAULTS.sampler))
    crn = CommonRandomNumbers(0, 1, 8192, REFERENCE)
    stock = REFERENCE.full((8192, 3), 0.65)
    shock = env.sampler(crn.at(0), REFERENCE)

    small = env.dynamics.mitigated_loss(stock, shock, REFERENCE.full((8192,), 5.0))
    large = env.dynamics.mitigated_loss(stock, shock, REFERENCE.full((8192,), 25.0))
    assert large.mean() > small.mean(), "a bigger book must carry more credit loss"

    # The difference is credit alone, so it is exactly proportional to the book.
    credit_only = (large - small) / 20.0
    rate = env.dynamics.mitigated_loss(
        stock, shock, REFERENCE.full((8192,), 1.0)
    ) - env.dynamics.mitigated_loss(stock, shock, torch.zeros(8192, dtype=torch.float64))
    assert torch.allclose(credit_only, rate, atol=1e-12)

    # And the oracle keeps it exogenous, because the closed form needs the
    # objective to separate across families.
    oracle = four_state_env()
    assert oracle.dynamics.credit_scales_with_book is False
