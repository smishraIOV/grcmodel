"""Two policies for the same firm, side by side (docs/plan-tool.md step 6).

The tool does not search for the best policy. What it offers in its place is
this: the same firm, the same scenarios, two policies of the user's choosing,
and what changing from one to the other buys and costs -- one period ahead
(`quant.readout`), under each stress (`quant.stress`), and over several years
(`quant.paths`).

Over several years both policies are run on **the same random paths** (the
same seed), so the difference between them is not blurred by one having drawn
luckier paths than the other: it is measured path by path, and its sampling
error is that of the differences (common random numbers, a standard device).
"""

import math
from dataclasses import dataclass, fields

from quant.numerics import REFERENCE, NumericsProfile
from quant.paths import PathResult, rule_from, simulate
from quant.policy import ALLOCATION, Policy, problems
from quant.profile import ModelFirm
from quant.readout import ReadOut, ScenarioSet, read_out, scenario_set
from quant.stress import Stress, StressResult, run_all


@dataclass(frozen=True)
class Comparison:
    """Policy `a` and policy `b` for one firm. `paths` is None when no
    horizon was asked for."""

    names: tuple[str, str]
    policies: tuple[Policy, Policy]
    read_outs: tuple[ReadOut, ReadOut]
    stresses: tuple[tuple[StressResult, ...], tuple[StressResult, ...]]
    paths: tuple[PathResult, PathResult] | None

    def value_gain(self) -> tuple[float, float]:
        """What shareholders take over the horizon under `b` less under `a`,
        on average, and its sampling error, path by path on the same paths."""
        if self.paths is None:
            raise ValueError("no horizon was asked for: pass `years` to compare")
        gap = self.paths[1].present_value - self.paths[0].present_value
        return float(gap.mean()), float(gap.std() / math.sqrt(gap.shape[0]))


def compare(
    model: ModelFirm, a: Policy, b: Policy, names: tuple[str, str] = ("A", "B"),
    stresses: tuple[Stress, ...] | None = None, years: float | None = None, paths: int = 20000,
    seed: int = 0, largest_raise: float | None = None, scenarios: ScenarioSet | None = None,
    profile: NumericsProfile = REFERENCE,
) -> Comparison:
    """Read both policies out, stress both, and with `years` carry both
    forward on the same paths. Each policy becomes a rule as
    `quant.paths.rule_from` makes one. An unsound policy is refused, named."""
    for name, policy in zip(names, (a, b), strict=True):
        found = problems(policy, model)
        if found:
            raise ValueError(f"policy {name} cannot be run: " + "; ".join(found))
    if scenarios is None:
        scenarios = scenario_set(model.params.per_period(), profile)
    both = (a, b)
    return Comparison(
        names=names, policies=both,
        read_outs=tuple(read_out(model, x, scenarios, profile) for x in both),
        stresses=tuple(tuple(run_all(model, x, stresses, profile)) for x in both),
        paths=None if years is None else tuple(
            simulate(model, rule_from(x, model, largest_raise), years=years, paths=paths, seed=seed, profile=profile)
            for x in both
        ),
    )


# -- the report ------------------------------------------------------------------------

_MONEY, _SHARE = "money", "share"
_READ_OUT_ROWS = (
    ("Capital", None, None),
    ("equity kept", "equity_kept", _MONEY),
    ("capital needed", "capital_needed", _MONEY),
    ("paid out now", "payout", _MONEY),
    ("raised now", "raised", _MONEY),
    ("stablecoins", "stablecoins", _MONEY),
    ("GRC and the loan book", None, None),
    ("chance of an incident in a year", "incident_chance", _SHARE),
    ("loans", "loans", _MONEY),
    ("expected bad debt a year", "expected_bad_debt", _MONEY),
    ("Expected over a year", None, None),
    ("interest share", "interest_income", _MONEY),
    ("BTC vault fees", "btc_vault_fees", _MONEY),
    ("repaid to investors", "compensation", _MONEY),
    ("Investors, expected a year", None, None),
    ("dollar vault", "dollar_yield", _SHARE),
    ("BTC vault, in BTC", "btc_yield", _SHARE),
    ("How the next period may end", None, None),
    ("expected equity", "expected_equity", _MONEY),
    ("chance of ending with less", "chance_of_loss", _SHARE),
    ("chance of ending below needs", "chance_below_needed", _SHARE),
    ("expected cost of refilling", "expected_cost_of_refilling", _MONEY),
)


def _cells(kind: str, x: float, y: float) -> str:
    if kind == _MONEY:
        return f"{x / 1e6:10.2f} {y / 1e6:10.2f} {(y - x) / 1e6:+10.2f}"
    return f"{100 * x:9.2f}% {100 * y:9.2f}% {100 * (y - x):+9.2f}pt"


def report(c: Comparison) -> str:
    """The two policies side by side, with the change from the first to the
    second. Dollars in millions; 'pt' is percentage points."""
    na, nb = c.names
    a, b = c.policies
    head = f"{'':<34} {na[:10]:>10} {nb[:10]:>10} {'change':>10}"
    lines = ["Dollars in millions; 'pt' is percentage points. 'change' is the second policy less the first.",
             "", "The two policies", head, "-" * len(head)]
    for f in fields(Policy):
        x, y = getattr(a, f.name), getattr(b, f.name)
        label = f.name.replace("_", " ")
        if f.name == "allocation":
            for name, xs, ys in zip(ALLOCATION, x, y, strict=True):
                lines.append(f"  {('to ' + name):<32} {_cells(_SHARE, xs, ys)}")
        elif f.name in ("borrower_rate", "ltv"):
            lines.append(f"  {label:<32} {_cells(_SHARE, x, y)}")
        else:
            lines.append(f"  {label:<32} {_cells(_MONEY, x, y)}")

    ra, rb = c.read_outs
    lines += ["", "One period ahead", head, "-" * len(head)]
    for label, name, kind in _READ_OUT_ROWS:
        if name is None:
            lines.append(label)
        else:
            lines.append(f"  {label:<32} {_cells(kind, getattr(ra, name), getattr(rb, name))}")

    width = max(len(s.stress.name) for s in c.stresses[0])
    head = (f"{'':<{width}} | {'equity after':^32} | {'short of capital needed':^32}\n"
            f"{'':<{width}} | {na[:10]:>10} {nb[:10]:>10} {'change':>10} | {na[:10]:>10} {nb[:10]:>10} {'change':>10}")
    lines += ["", "Under stress, after one period", head, "-" * (width + 70)]
    for sa, sb in zip(*c.stresses, strict=True):
        lines.append(f"{sa.stress.name:<{width}} | {_cells(_MONEY, sa.equity, sb.equity)} | "
                     f"{_cells(_MONEY, sa.shortfall, sb.shortfall)}")

    if c.paths is not None:
        pa, pb = c.paths
        head = f"{'':<34} {na[:10]:>10} {nb[:10]:>10} {'change':>10}"
        lines += ["", f"Over {pa.years:g} years, on the same {pa.paths:,} random paths (seed {pa.seed})", head, "-" * len(head)]
        rows = (
            ("chance of having failed", _SHARE, lambda r: float(r.chance_failed()[-1])),
            ("chance of having raised equity", _SHARE, lambda r: float(r.chance_raised()[-1])),
            ("paid out, on average", _MONEY, lambda r: float(r.payout.sum(dim=0).mean())),
            ("raised, on average", _MONEY, lambda r: float(r.raised.sum(dim=0).mean())),
            ("cost of raising, on average", _MONEY, lambda r: float(r.raise_cost.sum(dim=0).mean())),
            ("equity at the end: 5%", _MONEY, lambda r: float(r.quantiles(r.equity)[-1, 0])),
            ("equity at the end: median", _MONEY, lambda r: float(r.quantiles(r.equity)[-1, 2])),
            ("equity at the end: 95%", _MONEY, lambda r: float(r.quantiles(r.equity)[-1, 4])),
            ("what shareholders take", _MONEY, lambda r: r.value()[0]),
        )
        for label, kind, pick in rows:
            lines.append(f"  {label:<32} {_cells(kind, pick(pa), pick(pb))}")
        gain, error = c.value_gain()
        verdict = (
            "too small to tell from sampling error" if abs(gain) <= 2 * error
            else f"{nb} is ahead" if gain > 0 else f"{na} is ahead"
        )
        lines += [
            "",
            f"What shareholders take changes by {gain / 1e6:+.2f}, give or take {error / 1e6:.2f} from sampling: {verdict}.",
            "It is payouts less equity raised and its cost, discounted, plus equity at the end at book; it does not count risk.",
        ]
    return "\n".join(lines)
