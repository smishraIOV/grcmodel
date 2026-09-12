"""Rung 2: the grid solver, and what agreeing with it does and does not prove.

Every other solver in this project is a gradient method on the same objective,
so they can all be wrong in the same way. This one shares only the dynamics --
no gradients, no policy parameterization, no Adam -- which is what makes
agreement evidence rather than coincidence.
"""

from dataclasses import replace

import pytest
import torch

from quant.env.env import EnvConfig, FirmEnv, evaluate
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.numerics import REFERENCE
from quant.params import DEFAULTS
from quant.solvers.gridvi import (
    GridSolver,
    ReducedSpec,
    TabularPolicy,
    policy_improvement_gap,
)
from quant.solvers.pathwise import optimize_constant

HORIZON = 3


def reduced_env(**overrides):
    return FirmEnv(
        EnvConfig.quarterly(HORIZON, allow_abandonment=False, **overrides),
        MonteCarloSampler(DEFAULTS.sampler),
    )


def small_spec(**overrides):
    return replace(
        ReducedSpec(), n_equity=16, n_stock=8, n_spend=6, n_investment=6,
        n_shocks=512, **overrides,
    )


def test_interpolation_is_exact_on_grid_nodes():
    """Bilinear interpolation must reproduce the table where the table is
    defined, or every value the solver reports is off by an amount nothing
    measures."""
    solver = GridSolver(reduced_env(), small_spec())
    values = torch.arange(solver.n_nodes(), dtype=torch.float64)
    recovered = solver.interpolate(values, solver.node_equity, solver.node_stock)
    assert torch.allclose(recovered, values, atol=1e-10)


def test_edge_mass_reports_grid_truncation():
    """Interpolation clamps past the grid, so a firm that grows out of it gets
    a lower bound rather than a value. Silent truncation would look like a
    solver that suddenly stops rewarding growth."""
    solver = GridSolver(reduced_env(), small_spec())
    inside = REFERENCE.tensor([1.0, 2.0])
    assert solver.edge_mass(inside, inside) == 0.0
    outside = REFERENCE.tensor([1e6, 1e6])
    assert solver.edge_mass(outside, outside) == 1.0


def test_grid_and_rollout_agree_on_the_grids_own_policy():
    """The headline check. The grid's value at the opening state and an honest
    Monte Carlo rollout of the policy it implies are two independent routes to
    the same number.

    Tolerance is percent, not machine epsilon, and the sources are known: the
    grid takes its expectation over a finite shock sample and interpolates
    between nodes. Measured against a 4096-path rollout the gap runs 7.7% at 48
    shocks, 3.6% at 768 and 2.0% at 3072 -- monotone in the sample, which is
    what says it is sampling rather than disagreement.
    """
    env, spec = reduced_env(), small_spec()
    solver = GridSolver(env, spec)
    values, policies = solver.solve()
    firm = DEFAULTS.firm

    grid_value = solver.interpolate(
        values[0],
        REFERENCE.tensor([firm.initial_equity]),
        REFERENCE.tensor([firm.initial_grc_stock]),
    ).item()
    rollout = evaluate(
        TabularPolicy(solver, policies, spec), env, CommonRandomNumbers(0, HORIZON, 2048, REFERENCE)
    )
    assert abs(grid_value - rollout.value) < 0.10 * rollout.value


def test_the_grid_is_self_consistent_under_its_own_policy():
    """Q(s, a_grid(s)) - V(s) must be about zero: the grid's chosen action is
    by construction the one its value function rates highest, so anything large
    means the backup and the greedy extraction disagree."""
    env, spec = reduced_env(), small_spec()
    solver = GridSolver(env, spec)
    values, policies = solver.solve()
    policy = TabularPolicy(solver, policies, spec)

    trajectory = env.rollout(policy, CommonRandomNumbers(0, HORIZON, 512, REFERENCE), False)
    gap = policy_improvement_gap(solver, values, policy, trajectory)
    reference = solver.interpolate(
        values[0], trajectory.states[0].equity, trajectory.states[0].grc_stock[..., 0]
    )
    assert abs((gap / reference).mean().item()) < 0.02


def test_a_gradient_solver_agrees_with_the_grid():
    """The point of the rung. The constant-policy solver reaches its answer by
    Adam on a differentiable rollout; the grid reaches its by enumeration and
    backward induction. They share only the dynamics, so agreement is evidence
    that the dynamics and the objective are what both think they are.

    Checked on the *same restriction* the grid solves -- a policy using actions
    the grid holds fixed can legitimately score below the grid's best while
    beating it on honest evaluation, which is a statement about the
    restriction rather than about the policy.
    """
    env, spec = reduced_env(), small_spec()
    solver = GridSolver(env, spec)
    values, _ = solver.solve()
    crn = CommonRandomNumbers(0, HORIZON, 1024, REFERENCE)

    policy, _ = optimize_constant(env, crn, n_steps=1500)
    trajectory = env.rollout(policy, crn, False)
    gap = policy_improvement_gap(solver, values, policy, trajectory)
    reference = solver.interpolate(
        values[0], trajectory.states[0].equity, trajectory.states[0].grc_stock[..., 0]
    )
    assert abs((gap / reference).mean().item()) < 0.02
