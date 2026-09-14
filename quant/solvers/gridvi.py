"""Rung 2: backward induction on a grid, for a deliberately reduced model.

Every solver so far has been a gradient method on the same objective, which
means they can all be wrong in the same way. This one shares only the
*dynamics* -- it calls the same `StandardDynamics.step` the rollouts do -- and
otherwise has nothing in common with them: no gradients, no policy
parameterization, no Adam. Where it agrees with the pathwise solvers, the
agreement is evidence; where it does not, one of them is wrong and the
disagreement localizes to a state.

**Backward induction, not fixed-point iteration.** The environment is
finite-horizon, so the grid solves the same problem by stepping backwards from
the terminal value rather than iterating to a stationary policy. That keeps the
two exactly comparable: a stationary solution would be answering a different
question and would disagree for a reason that is not a bug.

**The reduced model, named and versioned.** A grid over the full state is not
the point and not affordable. The job of this rung is to be *right*, not
general, so it solves a restriction:

    state    equity, and a single GRC stock held equal across the three
             families -- 2 dimensions
    action   total spend (split equally) and investment -- 2 dimensions
    fixed    payout held at a constant, wind-down switched off, and the deposit
             base pinned to the capacity the node's equity supports

The binding constraint on a grid solver here is the action space, not the
state space: a 5-dimensional continuous action at ten points each would be
100,000 evaluations per node. Restricting the action is what makes the rung
affordable, and it is a restriction rather than an approximation -- the
pathwise solvers are compared against it *on the same restriction*.

**The deposit restriction is the one to watch.** Deposits adjust partially
toward capacity in the real dynamics, so the base can sit away from where the
firm's capital says it belongs -- which is the entire reason it is a state
variable rather than a formula in equity. This solver pins it to capacity at
every node, because carrying it properly would add a third grid dimension and
the rung's job is to be right, not general. The cost is that the grid cannot
see a firm that is over-levered relative to the capital it now has, which is
exactly the state a run produces. Read a disagreement about a post-shock state
as this restriction before reading it as a learner bug.
"""

from dataclasses import dataclass, replace

import torch

from quant.env.actions import FirmAction
from quant.env.env import FirmEnv
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.env.state import N_FAMILIES, FirmState
from quant.params import DEFAULTS


@dataclass(frozen=True)
class ReducedSpec:
    """The grid, and the restriction it solves on."""

    n_equity: int = 28
    n_stock: int = 14
    n_spend: int = 11
    n_investment: int = 11
    # Ranges track the calibrated scale and are easy to get silently wrong: a
    # grid whose action range does not cover the optimum reports a confident
    # answer to a different problem. Before the first recalibration these read
    # 16.0, 3.0 and 18.0 -- a stock ceiling twenty times the steady state and an
    # investment ceiling *below* what the firm wants to deploy. The liability
    # side moved them again, and by more: the book went from 25 to 80, so an
    # investment ceiling of 30 would have confined the grid to a firm a third
    # the size of the one every other solver is scoring.
    equity_max: float = 64.0
    stock_max: float = 5.0         # per family; the steady state is 1.17
    spend_max: float = 1.2         # total per quarter; the optimum is ~0.38
    investment_max: float = 110.0  # I* is 97.6, and the funding cap binds near 80
    payout: float = 0.0           # the level the constant-policy solver settles on
    # 48 was the first value tried and it is badly too few: the severity
    # distributions are heavy-tailed, so a small sample misses the tail and the
    # grid overstates value. Measured against an honest 4096-path rollout of
    # the grid's own policy, the gap runs 7.7% at 48 shocks, 3.6% at 768 and
    # 2.0% at 3072 -- monotone, so it is sampling rather than disagreement.
    n_shocks: int = 1024
    seed: int = 17

    def equity_grid(self, profile):
        return profile.tensor(
            torch.linspace(0.0, self.equity_max, self.n_equity).tolist()
        )

    def stock_grid(self, profile):
        return profile.tensor(torch.linspace(0.0, self.stock_max, self.n_stock).tolist())

    def actions(self, profile):
        spend = torch.linspace(0.0, self.spend_max, self.n_spend).tolist()
        invest = torch.linspace(0.0, self.investment_max, self.n_investment).tolist()
        return [(s, i) for s in spend for i in invest]


class GridSolver:
    """Backward induction over (equity, GRC stock) for the reduced model."""

    def __init__(self, env: FirmEnv, spec: ReducedSpec = ReducedSpec()):
        self.env, self.spec = env, spec
        self.profile = env.config.profile
        self.equities = spec.equity_grid(self.profile)
        self.stocks = spec.stock_grid(self.profile)
        self.action_list = spec.actions(self.profile)

        # One shock set, drawn once and reused at every node, action and step.
        # The same common-random-numbers discipline the rollouts use, and for
        # the same reason: it removes sampling noise from every comparison
        # made against this solver.
        crn = CommonRandomNumbers(spec.seed, 1, spec.n_shocks, self.profile)
        self.shock = MonteCarloSampler(DEFAULTS.sampler)(crn.at(0), self.profile)

        grid_e, grid_g = torch.meshgrid(self.equities, self.stocks, indexing="ij")
        self.node_equity = grid_e.reshape(-1)
        self.node_stock = grid_g.reshape(-1)

    def n_nodes(self) -> int:
        return self.node_equity.numel()

    def interpolate(self, values, equity, stock):
        """Bilinear interpolation of V on the grid, clamped at the edges.

        Clamping means the solver treats anything past the grid as if it were
        at the boundary, which understates value for a firm that has grown out
        of the grid. `edge_mass` reports how often that happens so the
        truncation is visible rather than silent.
        """
        table = values.reshape(self.spec.n_equity, self.spec.n_stock)
        e_index, e_frac = _locate(self.equities, equity)
        g_index, g_frac = _locate(self.stocks, stock)

        low = table[e_index, g_index] * (1 - g_frac) + table[e_index, g_index + 1] * g_frac
        high = (
            table[e_index + 1, g_index] * (1 - g_frac)
            + table[e_index + 1, g_index + 1] * g_frac
        )
        return low * (1 - e_frac) + high * e_frac

    def _action(self, spend: float, investment: float, batch: int) -> FirmAction:
        per_family = self.profile.full((batch, N_FAMILIES), spend / N_FAMILIES)
        return FirmAction(
            grc=per_family,
            investment=self.profile.full((batch,), investment),
            abandon=self.profile.full((batch,), 0.0),
            payout=self.profile.full((batch,), self.spec.payout),
        )

    def q_at(self, continuation, equity, stock, action: FirmAction, t: int):
        """Q(s, a) for arbitrary states and arbitrary actions.

        The same one-step backup the grid uses on its own action set, opened up
        so that a policy's chosen action can be scored against the grid's value
        function. That is what the agreement metric needs: comparing V^grid
        with V^rollout conflates the grid's discretization error, the learner's
        optimization error and Monte Carlo noise, while Q(s, a_policy) - V(s)
        takes both terms from the same value function and isolates the policy.
        """
        draws = self.spec.n_shocks
        points = equity.numel()
        batch = points * draws

        state = FirmState(
            equity=equity.repeat_interleave(draws),
            grc_stock=stock.repeat_interleave(draws)
            .unsqueeze(-1)
            .expand(batch, N_FAMILIES)
            .contiguous(),
            alive=torch.ones(batch, dtype=torch.bool, device=self.profile.device),
            deposits=self._deposits(equity.repeat_interleave(draws)),
            t=t,
        )
        wide = FirmAction(
            grc=action.grc.repeat_interleave(draws, dim=0),
            investment=action.investment.repeat_interleave(draws),
            abandon=action.abandon.repeat_interleave(draws),
            payout=action.payout.repeat_interleave(draws),
        )
        result = self.env.dynamics.step(state, wide, self._tile(points))
        survival = torch.exp(result.log_survival)
        future = self.interpolate(
            continuation, result.state.equity, result.state.grc_stock[..., 0]
        )
        value = result.reward + self.env.config.discount * (
            survival * future + (1.0 - survival) * result.failure_value
        )
        weight = result.weight.reshape(points, draws)
        return (value.reshape(points, draws) * weight).sum(dim=1) / weight.sum(dim=1)

    def _deposits(self, equity: torch.Tensor) -> torch.Tensor | None:
        """The grid's deposit restriction: the base capacity supports, or None
        when the environment has no liability side. See the module docstring."""
        if self.env.config.funding is None:
            return None
        return self.env.dynamics.deposit_capacity(equity)

    def _tile(self, repeats: int):
        """The same shock draw, repeated once per node.

        Every tensor field is tiled by reflection rather than by name. The
        hand-written version listed eight fields, and adding a ninth to `Shock`
        left it at the original batch size while everything else was 128 times
        larger -- a shape error rather than a silent wrong answer, which is the
        only reason it was cheap to find. Reflection means a new channel cannot
        be forgotten here at all.
        """
        tiled = {
            name: value.repeat(repeats)
            for name, value in vars(self.shock).items()
            if isinstance(value, torch.Tensor)
        }
        return replace(self.shock, **tiled)

    def q_values(self, continuation, spend: float, investment: float, t: int):
        """E[ reward + discount * continuation ] at every node, for one action.

        Vectorized over nodes and shocks together; the action grid is walked in
        a Python loop, which keeps the peak tensor at nodes x shocks rather
        than nodes x actions x shocks.
        """
        nodes, draws = self.n_nodes(), self.spec.n_shocks
        batch = nodes * draws

        state = FirmState(
            equity=self.node_equity.repeat_interleave(draws),
            grc_stock=self.node_stock.repeat_interleave(draws)
            .unsqueeze(-1)
            .expand(batch, N_FAMILIES)
            .contiguous(),
            alive=torch.ones(batch, dtype=torch.bool, device=self.profile.device),
            deposits=self._deposits(self.node_equity.repeat_interleave(draws)),
            t=t,
        )
        tiled = self._tile(nodes)

        result = self.env.dynamics.step(state, self._action(spend, investment, batch), tiled)
        survival = torch.exp(result.log_survival)
        future = self.interpolate(
            continuation, result.state.equity, result.state.grc_stock[..., 0]
        )
        value = result.reward + self.env.config.discount * (
            survival * future + (1.0 - survival) * result.failure_value
        )
        # The reweighting the compliance channel uses, applied within each node.
        weight = result.weight.reshape(nodes, draws)
        return (value.reshape(nodes, draws) * weight).sum(dim=1) / weight.sum(dim=1)

    def solve(self, horizon: int | None = None):
        """Backward induction. Returns (values per step, greedy action indices)."""
        horizon = horizon or self.env.config.horizon
        terminal = FirmState(
            equity=self.node_equity,
            grc_stock=self.node_stock.unsqueeze(-1).expand(self.n_nodes(), N_FAMILIES),
            alive=torch.ones(self.n_nodes(), dtype=torch.bool, device=self.profile.device),
            deposits=self._deposits(self.node_equity),
            t=horizon,
        )
        values = [self.env.terminal_value(terminal)]
        policies = []

        for step in reversed(range(horizon)):
            best = None
            argbest = None
            for index, (spend, investment) in enumerate(self.action_list):
                q = self.q_values(values[0], spend, investment, step)
                if best is None:
                    best, argbest = q, torch.zeros_like(q, dtype=torch.long)
                else:
                    improved = q > best
                    best = torch.where(improved, q, best)
                    argbest = torch.where(improved, index, argbest)
            values.insert(0, best)
            policies.insert(0, argbest)
        return values, policies

    def edge_mass(self, equity, stock) -> float:
        """Share of queried points sitting outside the grid, where interpolation
        is clamped and the answer is a lower bound rather than a value."""
        outside = (equity > self.equities[-1]) | (stock > self.stocks[-1])
        return outside.to(self.profile.dtype).mean().item()


def _locate(grid, query):
    """Lower index and fractional position of `query` within `grid`."""
    index = torch.searchsorted(grid, query.detach().contiguous()) - 1
    index = index.clamp(0, grid.numel() - 2)
    low, high = grid[index], grid[index + 1]
    frac = ((query - low) / (high - low)).clamp(0.0, 1.0)
    return index, frac


class TabularPolicy:
    """The grid's greedy policy, usable anywhere a Policy is."""

    def __init__(self, solver: GridSolver, policies, spec: ReducedSpec):
        self.solver, self.policies, self.spec = solver, policies, spec
        self.actions = solver.action_list

    def __call__(self, state: FirmState) -> FirmAction:
        step = min(state.t, len(self.policies) - 1)
        table = self.policies[step].reshape(self.spec.n_equity, self.spec.n_stock)
        e_index, _ = _locate(self.solver.equities, state.equity)
        g_index, _ = _locate(self.solver.stocks, state.grc_stock[..., 0])
        chosen = table[e_index, g_index]

        lookup = self.solver.profile.tensor([[s, i] for s, i in self.actions])
        picked = lookup[chosen]
        profile = self.solver.profile
        batch = state.batch()
        return FirmAction(
            grc=(picked[..., 0] / N_FAMILIES).unsqueeze(-1).expand(batch, N_FAMILIES),
            investment=picked[..., 1],
            abandon=profile.full((batch,), 0.0),
            payout=profile.full((batch,), self.spec.payout),
        )


def policy_improvement_gap(solver: GridSolver, values, policy, trajectory, step: int = 0):
    """Q^grid(s, a_policy(s)) - V^grid(s), at states the policy actually visits.

    Sampled from the policy's own occupancy rather than from the grid, because
    accuracy matters where the policy goes. A negative gap is value the policy
    leaves on the table against the grid's best action; a positive one means
    the policy found something the grid's action set cannot express, which is
    a statement about the discretization rather than about the policy.

    Both terms come from the same value function, so the grid's own
    discretization error largely cancels and what is left is the policy.
    """
    state = trajectory.states[step]
    equity, stock = state.equity, state.grc_stock[..., 0]
    action = policy(state)

    baseline = solver.interpolate(values[step], equity, stock)
    chosen = solver.q_at(values[step + 1], equity, stock, action, step)
    return chosen - baseline
