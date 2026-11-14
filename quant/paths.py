"""The firm over several years, under a rule the user sets (docs/plan-tool.md
step 5; design.md section 3.2).

`simulate` carries the firm forward period by period through the one-period
accounting (`quant.integrated.integrated_step`), on many paths of randomly
drawn scenarios. It shows the range of where the firm may be: its equity, how
much it pays out, how often it has to raise equity and at what cost, how
often it fails, and the size of its vaults.

**The rule** (`Rule`) is what the firm does every period. A fixed policy
cannot be carried forward: the firm's equity and vaults change, and what it
may keep changes with them. So equity follows a rule, and the rest is held
fixed:

- above the capital it needs plus a buffer, the firm pays the excess out;
- between the two it does nothing;
- below what it needs, it raises equity back to exactly that. Raising is
  expensive, and dearer straight after BTC fell or an incident happened
  (`quant.core.issuance_cost_after`). If the raise needed is more than
  shareholders will put in (`largest_raise`), **the firm fails** on that
  path: they walk away, and receive nothing more.

**Random scenarios** (Monte Carlo). Each path draws, each period: the
Student-t shocks to operating losses and the BTC strategies (one shared tail
factor), BTC's smooth move (normal), whether BTC crashes and by how much, and
whether there is an incident, where, and how large -- the same distributions
the read-out averages over with weights. The draws come from a seeded
generator on the CPU in double precision (CLAUDE.md), so the same seed gives
the same paths. Figures carry sampling error: about one over the square root
of the number of paths, and more for rare events.

Over several periods **loan demand moves with BTC's price** (`demand_btc`):
borrowers post BTC, and what they can borrow against it moves with what it is
worth.

Money is in dollars. The assumptions are illustrative until calibrated.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import gammaincinv

from quant.core import issuance_cost_after, jump_probability, next_grc
from quant.firm import crash_probability
from quant.integrated import (
    IntegratedControls,
    integrated_btc_held,
    integrated_capital_floor,
    integrated_correlation,
    integrated_step,
)
from quant.numerics import REFERENCE, NumericsProfile
from quant.params import Params
from quant.policy import ALLOCATION, Policy
from quant.profile import ModelFirm
from quant.quadrature import crash_scale

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True, kw_only=True)
class Rule:
    """What the firm does every period. Dollars, and rates per year."""

    equity_buffer: float               # equity held above the capital needed; more than that is paid out
    largest_raise: float               # the most shareholders will put in at once; needing more, the firm fails
    grc_spend: float                   # a year
    own_btc: float                     # the firm's own BTC, valued in dollars, restored every period
    hedge_ratio: float                 # share of the BTC the firm owns that is hedged, 0 to 1
    borrower_rate: float               # a year
    ltv: float
    allocation: tuple[float, float, float, float]   # the BTC vault's shares, in the order of quant.policy.ALLOCATION


def _controls(rule: Rule, model: ModelFirm, profile: NumericsProfile):
    """The rule's fixed decisions for one period, in model units."""
    scale, n = model.scale, model.params.periods_per_year
    g = profile.tensor(rule.grc_spend / scale / n)
    q = profile.tensor(rule.own_btc / scale)
    rate = profile.tensor((1.0 + rule.borrower_rate) ** (1.0 / n) - 1.0)
    return g, q, rate, profile.tensor(rule.ltv), profile.tensor([list(rule.allocation)])


def capital_needed(rule: Rule, model: ModelFirm, profile: NumericsProfile = REFERENCE) -> float:
    """The least equity the firm may keep today under this rule, in dollars."""
    p = model.params.per_period()
    g, q, _, _, _ = _controls(rule, model, profile)
    A_L, A_B = profile.tensor(model.state.A_L), profile.tensor(model.state.A_B)
    return float(integrated_capital_floor(p, A_L, A_B, g, q)) * model.scale


def problems(rule: Rule, model: ModelFirm) -> list[str]:
    """Everything wrong with a rule for this firm, in plain words."""
    p = model.params
    found = []
    for name, value in (("the equity buffer", rule.equity_buffer), ("the largest raise", rule.largest_raise),
                        ("GRC spend", rule.grc_spend), ("the firm's own BTC", rule.own_btc),
                        ("the borrower rate", rule.borrower_rate)):
        if value < 0:
            found.append(f"{name} cannot be negative")
    if not 0.0 <= rule.hedge_ratio <= 1.0:
        found.append("the hedge ratio must be between 0 and 1: short perps cover at most the BTC the firm owns")
    if len(rule.allocation) != len(ALLOCATION):
        found.append(f"the BTC vault's allocation needs {len(ALLOCATION)} shares: {', '.join(ALLOCATION)}")
    elif min(rule.allocation) < 0:
        found.append("the BTC vault's allocation cannot have a negative share")
    elif abs(sum(rule.allocation) - 1.0) > 1e-6:
        found.append(f"the BTC vault's allocation must sum to 100% (it sums to {100 * sum(rule.allocation):.1f}%)")
    if not p.ltv_min - 1e-9 <= rule.ltv <= p.ltv_max + 1e-9:
        found.append(
            f"the loan-to-value of {100 * rule.ltv:.0f}% is outside the range the firm may offer, "
            f"{100 * p.ltv_min:.0f}% to {100 * p.ltv_max:.0f}%"
        )
    return found


def rule_from(policy: Policy, model: ModelFirm, largest_raise: float | None = None) -> Rule:
    """The rule that starts as `policy` does: the same decisions, the buffer
    this policy keeps above the capital it needs today, and the share of its
    BTC it hedges. `largest_raise` defaults to the capital needed today:
    shareholders will rebuild the firm's capital once from nothing, not more."""
    p = model.params
    held = policy.own_btc + p.btc_co_invest * model.state.A_B * model.scale
    rule = Rule(
        equity_buffer=0.0, largest_raise=0.0, grc_spend=policy.grc_spend, own_btc=policy.own_btc,
        hedge_ratio=policy.hedged / held if held > 0 else 0.0, borrower_rate=policy.borrower_rate,
        ltv=policy.ltv, allocation=tuple(policy.allocation),
    )
    needed = capital_needed(rule, model)
    return Rule(**{
        **rule.__dict__,
        "equity_buffer": max(policy.equity_kept - needed, 0.0),
        "largest_raise": needed if largest_raise is None else largest_raise,
    })


# -- random scenarios -----------------------------------------------------------------


@dataclass(frozen=True)
class Draws:
    """Every path's scenarios, period by period: each tensor is (periods, paths)."""

    xi_l: torch.Tensor          # the lending system's operating-loss shock
    xi_1: torch.Tensor          # the two active BTC strategies' yield shocks
    xi_2: torch.Tensor
    eps: torch.Tensor           # BTC's smooth move
    crash: torch.Tensor         # the share of BTC's price a crash takes; zero without one
    incident: torch.Tensor      # uniform: decides whether there is an incident, and where
    severity: torch.Tensor      # its size, as a share


def draw(
    per_period: Params, periods: int, paths: int, seed: int, profile: NumericsProfile = REFERENCE
) -> Draws:
    """Random scenarios from the model's own distributions (design.md sections
    3.4, 4.2 and 4.3), drawn on the CPU in double precision and cast last."""
    p = per_period
    generator = profile.generator(seed)

    def uniform() -> torch.Tensor:
        # kept off 0 and 1, where the inverse distributions are infinite
        return profile.draw_uniform((periods, paths), generator).clamp(1e-15, 1.0 - 1e-15)

    def normal() -> torch.Tensor:
        return torch.special.ndtri(uniform())

    # The vault's shocks: correlated normals over one shared scale (a Student-t), unit variance.
    chol = torch.linalg.cholesky(torch.as_tensor(integrated_correlation(p)[:3, :3], dtype=torch.float64))
    z = torch.stack([normal(), normal(), normal()], dim=-1) @ chol.T
    if math.isinf(p.nu):
        scale = torch.ones((periods, paths), dtype=torch.float64)
    else:
        w = 2.0 * gammaincinv(p.nu / 2.0, uniform().numpy())             # chi-squared, nu degrees of freedom
        scale = torch.sqrt((p.nu - 2.0) / torch.as_tensor(np.asarray(w), dtype=torch.float64))
    xi = z * scale.unsqueeze(-1)

    eps = normal()
    if p.btc_crash_rate > 0.0:
        size = crash_scale(p.btc_crash_mean, p.btc_crash_tail) * uniform() ** (-1.0 / p.btc_crash_tail)
        crash = torch.where(uniform() < crash_probability(p), 1.0 - torch.exp(-size), torch.zeros_like(size))
    else:
        crash = torch.zeros_like(eps)
    pareto = p.jump_mean * (p.jump_tail - 1.0) / p.jump_tail              # as quant.quadrature.pareto_jump_rule
    severity = torch.clamp(pareto * uniform() ** (-1.0 / p.jump_tail), max=1.0)
    return Draws(
        xi_l=profile.to(xi[..., 0]), xi_1=profile.to(xi[..., 1]), xi_2=profile.to(xi[..., 2]),
        eps=profile.to(eps), crash=profile.to(crash), incident=profile.to(uniform()),
        severity=profile.to(severity),
    )


# -- the simulation --------------------------------------------------------------------


@dataclass(frozen=True)
class PathResult:
    """Many paths of one firm under one rule. Dollars. Tensors indexed by
    period have `periods + 1` entries where they describe a moment (index 0 is
    today, the last is the end) and `periods` where they describe a period."""

    rule: Rule
    decisions_per_year: int
    periods: int
    paths: int
    seed: int
    # -- per path, per moment: (periods + 1, paths) --------------------------------------
    equity: torch.Tensor         # before that period's payout or raise; zero once the firm has failed
    dollar_vault: torch.Tensor
    btc_vault: torch.Tensor
    btc_price: torch.Tensor      # today = 1
    failed: torch.Tensor         # whether the firm has failed by then
    # -- per path, per period: (periods, paths) ---------------------------------------------
    payout: torch.Tensor
    raised: torch.Tensor
    raise_cost: torch.Tensor
    bad_debt: torch.Tensor
    dollar_return: torch.Tensor  # what outside investors received over the period
    btc_return: torch.Tensor     # in BTC
    # -- per path -----------------------------------------------------------------------------
    present_value: torch.Tensor  # see `value`
    discount: float              # the discount factor of one period

    @property
    def years(self) -> float:
        return self.periods / self.decisions_per_year

    def quantiles(self, series: torch.Tensor, alive_only: bool = False) -> torch.Tensor:
        """The 5%, 25%, 50%, 75% and 95% points of a per-moment series, at
        each moment: (moments, 5). With `alive_only`, over the paths on which
        the firm has not failed by then."""
        q = torch.tensor(QUANTILES, dtype=series.dtype, device=series.device)
        if not alive_only:
            return torch.quantile(series, q, dim=1).T
        rows = []
        for t in range(series.shape[0]):
            alive = series[t][~self.failed[t]]
            rows.append(torch.quantile(alive, q) if alive.numel() else torch.full_like(q, float("nan")))
        return torch.stack(rows)

    def chance_failed(self) -> torch.Tensor:
        """The chance the firm has failed by each moment: (periods + 1,)."""
        return self.failed.to(self.equity.dtype).mean(dim=1)

    def chance_raised(self) -> torch.Tensor:
        """The chance it has raised equity at least once by the end of each period: (periods,)."""
        return (torch.cumsum(self.raised, dim=0) > 0).to(self.equity.dtype).mean(dim=1)

    def value(self) -> tuple[float, float]:
        """What shareholders take over the horizon, per path on average, and
        its sampling error: payouts less equity raised and its cost, each
        discounted at their required return, plus the firm's equity at the
        end counted at book (nothing if it failed)."""
        v = self.present_value
        return float(v.mean()), float(v.std() / math.sqrt(v.shape[0]))


def simulate(
    model: ModelFirm, rule: Rule, years: float = 5.0, paths: int = 20000, seed: int = 0,
    profile: NumericsProfile = REFERENCE,
) -> PathResult:
    """Carry the firm forward `years` under `rule`, on `paths` random paths.
    An unsound rule is refused with the reasons."""
    found = problems(rule, model)
    if found:
        raise ValueError("this rule cannot be run: " + "; ".join(found))
    n, scale = model.params.periods_per_year, model.scale
    periods = round(years * n)
    if periods < 1:
        raise ValueError("the horizon must be at least one period")
    p = model.params.per_period()
    shocks = draw(p, periods, paths, seed, profile)
    g, q, rate, ltv, alloc = _controls(rule, model, profile)
    buffer, largest = rule.equity_buffer / scale, rule.largest_raise / scale
    beta = p.beta

    def full(x: float) -> torch.Tensor:
        return profile.full((paths,), x)

    w, S = full(model.state.w), full(model.state.S)
    A_L, A_B = full(model.state.A_L), full(model.state.A_B)
    index = full(1.0)                                   # BTC's price, today = 1
    fall, incident = full(0.0), full(0.0)               # last period's, for the cost of raising
    failed = torch.zeros(paths, dtype=torch.bool, device=w.device)
    value = full(0.0)
    moments = {k: [] for k in ("equity", "dollar_vault", "btc_vault", "btc_price", "failed")}
    flows = {k: [] for k in ("payout", "raised", "raise_cost", "bad_debt", "dollar_return", "btc_return")}

    def record(now_failed: torch.Tensor) -> None:
        moments["equity"].append(torch.where(now_failed, torch.zeros_like(w), w) * scale)
        moments["dollar_vault"].append(A_L * scale)
        moments["btc_vault"].append(A_B * scale)
        moments["btc_price"].append(index)
        moments["failed"].append(now_failed)

    with torch.no_grad():
        for t in range(periods):
            record(failed)
            # -- the equity rule ---------------------------------------------------------
            needed = integrated_capital_floor(p, A_L, A_B, g, q)
            target = needed + buffer
            payout = torch.relu(w - target)
            raised = torch.relu(needed - w)
            failed = failed | (raised > largest)
            alive = (~failed).to(w.dtype)
            payout, raised = payout * alive, raised * alive
            cost = issuance_cost_after(p, raised, fall, incident)
            K = torch.maximum(torch.minimum(w, target), needed)
            value = value + beta**t * (payout - raised - cost)
            # -- the period ----------------------------------------------------------------
            c = IntegratedControls(
                K=K, g=g, q=q, H=rule.hedge_ratio * integrated_btc_held(p, A_B, q), rate=rate, ltv=ltv, alloc=alloc,
            )
            S = next_grc(p, S, g)
            chance = jump_probability(p, S) if p.jump_rate > 0.0 else torch.zeros_like(S)
            in_lending = shocks.incident[t] < chance
            in_btc_vault = (shocks.incident[t] >= chance) & (shocks.incident[t] < 2.0 * chance)
            zero = torch.zeros_like(w)
            step = integrated_step(
                p, A_L, A_B, c, S, shocks.xi_l[t], shocks.xi_1[t], shocks.xi_2[t], shocks.eps[t], shocks.crash[t],
                torch.where(in_lending, shocks.severity[t], zero), torch.where(in_btc_vault, shocks.severity[t], zero),
                demand_scale=index ** p.demand_btc,
            )
            for name, x in (("payout", payout * scale), ("raised", raised * scale), ("raise_cost", cost * scale),
                            ("bad_debt", step.bad_debt * alive * scale), ("dollar_return", step.received_l),
                            ("btc_return", step.received_b)):
                flows[name].append(x.expand(paths))
            w, A_L, A_B = step.wealth, step.aum_l, step.aum_b
            index = index * step.price
            fall, incident = 1.0 - step.price, (in_lending | in_btc_vault).to(w.dtype)
        record(failed)
        value = value + beta**periods * torch.relu(w) * (~failed).to(w.dtype)

    return PathResult(
        rule=rule, decisions_per_year=n, periods=periods, paths=paths, seed=seed,
        **{k: torch.stack(v) for k, v in moments.items()},
        **{k: torch.stack(v) for k, v in flows.items()},
        present_value=value * scale, discount=beta,
    )


def report(r: PathResult) -> str:
    """The paths year by year, in millions of dollars."""
    n = r.decisions_per_year
    marks = [t for t in range(0, r.periods + 1, n)]
    if marks[-1] != r.periods:
        marks.append(r.periods)
    equity, dollar, btc = r.quantiles(r.equity), r.quantiles(r.dollar_vault, True), r.quantiles(r.btc_vault, True)
    price = r.quantiles(r.btc_price)
    failed, raised = r.chance_failed(), r.chance_raised()
    paid = torch.cumsum(r.payout.mean(dim=1), dim=0)
    put_in = torch.cumsum((r.raised + r.raise_cost).mean(dim=1), dim=0)

    def m(x) -> str:
        return f"{float(x) / 1e6:7.1f}"

    def pct(x) -> str:
        return f"{100 * float(x):6.1f}%"

    value, error = r.value()
    head = (f"{'year':>4} | {'equity: 5%':>10} {'median':>7} {'95%':>7} | {'failed':>7} {'raised':>7} | "
            f"{'paid out':>8} {'put in':>7} | {'$ vault':>7} {'B vault':>7} | {'BTC: 5%':>7} {'median':>7} {'95%':>7}")
    lines = [
        f"{r.paths:,} random paths over {r.years:g} years ({n} decisions a year), seed {r.seed}. Dollars in millions.",
        "'failed': the chance shareholders have walked away by then. 'raised': the chance of at least one equity raise.",
        "'paid out' and 'put in' (raises and their cost) are averages over all paths, to date. Vaults are medians",
        "over the paths still running. BTC's price is today's = 1.",
        "",
        head,
        "-" * len(head),
    ]
    for t in marks:
        done = t - 1
        lines.append(
            f"{t / n:4.1f} | {m(equity[t, 0]):>10} {m(equity[t, 2])} {m(equity[t, 4])} | {pct(failed[t])} "
            f"{pct(raised[done]) if t else pct(0.0)} | {m(paid[done]) if t else m(0.0):>8} "
            f"{m(put_in[done]) if t else m(0.0)} | {m(dollar[t, 2])} {m(btc[t, 2])} | "
            f"{float(price[t, 0]):7.2f} {float(price[t, 2]):7.2f} {float(price[t, 4]):7.2f}"
        )
    lines += [
        "",
        f"What shareholders take over the {r.years:g} years, on average: {value / 1e6:.2f} (give or take {error / 1e6:.2f} "
        "from sampling):",
        "payouts less equity raised and its cost, discounted at their required return, plus equity at the end at book.",
    ]
    return "\n".join(lines)
