"""Assumptions shown as assumptions (docs/plan-tool.md step 7).

Much of what a profile assumes cannot be observed: how well GRC works, how
often incidents happen and how bad they are, how often BTC crashes, what a
liquidation loses, what investors do after an incident. A `Dial` is one such
assumption with a range. `sensitivity` turns each dial to the low and the
high end of its range, one at a time, and records what every output does --
so each figure the tool gives can be read with how much it leans on which
assumption, and which assumptions it does not depend on at all.

With `versus`, the outputs are the **change from one policy to another**.
That is how the tool answers "what is more GRC spend worth": the answer is
not a number but a range, set by how well GRC is assumed to work.

The ranges of `standard_dials` are wide on purpose. They are not estimates:
they say "if this assumption were half or twice what was entered", so that
the assumptions worth pinning down stand out from those that do not matter.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

from quant.numerics import REFERENCE, NumericsProfile
from quant.paths import PathResult, rule_from, simulate
from quant.policy import Policy, check
from quant.profile import FirmProfile, GrcEffect, LoanPoint, to_model
from quant.readout import ReadOut, read_out, scenario_set
from quant.stress import Stress, StressResult, run

GRC_EFFECTS = ("grc_on_losses", "grc_on_spread", "grc_on_incidents", "grc_on_sale_discount")
# Fewer scenarios than the read-out's own default: a sensitivity is many read-outs, and what
# matters is how the figures move, not their last digit.
SCENARIO_COUNTS = dict(n_ops=2, n_strategy=4, n_btc=101, n_mixing=8, n_crash=24, n_jump=8)
STRESSES = (
    Stress(name="BTC crashes 50%", btc_crash=0.50),
    Stress(name="incident: 10% of the lending system", lending_incident=0.10),
)


@dataclass(frozen=True)
class Dial:
    """One assumption and the range to try it over. `apply` returns the
    profile with the assumption set to a value; `base` is its value in the
    profile as entered. `unit` says how to print it: 'share', 'a year' or 'x'
    (a multiple of what was entered)."""

    name: str
    base: float
    low: float
    high: float
    unit: str
    apply: Callable[[FirmProfile, float], FirmProfile]


def _risk(profile: FirmProfile, **changes) -> FirmProfile:
    return replace(profile, risk=replace(profile.risk, **changes))


def _market(profile: FirmProfile, **changes) -> FirmProfile:
    return replace(profile, market=replace(profile.market, **changes))


def _grc_works(profile: FirmProfile, factor: float) -> FirmProfile:
    """Every GRC statement's reduction times `factor`, kept below its limit."""
    changes = {}
    for name in GRC_EFFECTS:
        e: GrcEffect = getattr(profile.risk, name)
        changes[name] = replace(e, reduction=min(factor * e.reduction, 0.98 * e.limit))
    return _risk(profile, **changes)


def standard_dials(profile: FirmProfile) -> tuple[Dial, ...]:
    """The assumptions a user cannot look up, each over a wide range around
    what the profile enters."""
    m, k, f = profile.market, profile.risk, profile.facts
    room = 1.0 - f.liquidation_threshold          # the sale discount must stay below this

    def scaled(name: str, unit: str, base: float, low: float, high: float, apply, cap: float | None = None) -> Dial:
        top = base * high if cap is None else min(base * high, cap)
        return Dial(name=name, base=base, low=base * low, high=top, unit=unit, apply=apply)

    return (
        Dial(name="how well GRC works (0: not at all)", base=1.0, low=0.0, high=1.5, unit="x", apply=_grc_works),
        scaled("incidents a year, with no GRC", "a year", k.incidents_per_year, 0.5, 2.0,
               lambda p, v: _risk(p, incidents_per_year=v)),
        scaled("average loss in an incident", "share", k.incident_loss, 0.5, 2.0,
               lambda p, v: _risk(p, incident_loss=v), cap=0.9),
        scaled("everyday operating losses", "x", 1.0, 0.5, 2.0,
               lambda p, v: _risk(p, lending_loss=v * p.risk.lending_loss,
                                  strategy_loss=tuple(v * x for x in p.risk.strategy_loss))),
        scaled("sale discount in a liquidation", "share", k.sale_discount, 0.5, 1.5,
               lambda p, v: _risk(p, sale_discount=v), cap=0.95 * room),
        scaled("slippage in a smooth fall", "share", k.liquidation_slippage, 0.0, 2.0,
               lambda p, v: _risk(p, liquidation_slippage=v), cap=1.0),
        scaled("investors leaving after an incident", "x", 1.0, 0.5, 2.0,
               lambda p, v: _risk(p, outflow_after_incident=min(v * p.risk.outflow_after_incident, 1.0),
                                  outflow_from_other_vault=min(v * p.risk.outflow_from_other_vault, 1.0))),
        scaled("BTC volatility", "share", m.btc_volatility, 2 / 3, 4 / 3, lambda p, v: _market(p, btc_volatility=v)),
        scaled("BTC crashes a year", "a year", m.btc_crashes_per_year, 0.5, 2.0,
               lambda p, v: _market(p, btc_crashes_per_year=v)),
        scaled("average BTC crash", "share", m.btc_crash_size, 0.6, 1.4,
               lambda p, v: _market(p, btc_crash_size=v), cap=0.9),
        scaled("loan demand", "x", 1.0, 0.7, 1.3,
               lambda p, v: _market(p, loan_demand=LoanPoint(v * p.market.loan_demand.loans, p.market.loan_demand.rate))),
        scaled("how strongly investors follow yield", "x", 1.0, 0.5, 2.0,
               lambda p, v: _market(p, dollar_flow_response=v * p.market.dollar_flow_response,
                                    btc_flow_response=v * p.market.btc_flow_response)),
        scaled("extra cost of raising after a bad period", "x", 1.0, 0.0, 2.0,
               lambda p, v: _market(p, raise_cost_after_btc_halves=1 + v * (p.market.raise_cost_after_btc_halves - 1),
                                    raise_cost_after_incident=1 + v * (p.market.raise_cost_after_incident - 1))),
    )


@dataclass(frozen=True)
class Outlook:
    """Everything the tool says about one firm under one policy."""

    read_out: ReadOut
    stresses: tuple[StressResult, ...]
    paths: PathResult | None


@dataclass(frozen=True)
class Output:
    """One figure to follow. `kind` is 'money' (dollars) or 'share'."""

    name: str
    kind: str
    pick: Callable[[Outlook], float]
    needs_paths: bool = False


def standard_outputs() -> tuple[Output, ...]:
    return (
        Output("revenue a year", "money", lambda o: o.read_out.revenue),
        Output("expected bad debt a year", "money", lambda o: o.read_out.expected_bad_debt),
        Output("chance of an incident in a year", "share", lambda o: o.read_out.incident_chance),
        Output("dollar investors' expected yield", "share", lambda o: o.read_out.dollar_yield),
        Output("BTC investors' expected yield, in BTC", "share", lambda o: o.read_out.btc_yield),
        Output("expected equity after one period", "money", lambda o: o.read_out.expected_equity),
        Output("chance of ending the period below needs", "share", lambda o: o.read_out.chance_below_needed),
        Output("expected cost of refilling capital", "money", lambda o: o.read_out.expected_cost_of_refilling),
        *(Output(f"equity after: {s.name}", "money", lambda o, i=i: o.stresses[i].equity) for i, s in enumerate(STRESSES)),
        Output("chance of having failed, over the horizon", "share",
               lambda o: float(o.paths.chance_failed()[-1]), needs_paths=True),
        Output("chance of having raised equity, over the horizon", "share",
               lambda o: float(o.paths.chance_raised()[-1]), needs_paths=True),
        Output("what shareholders take, over the horizon", "money", lambda o: o.paths.value()[0], needs_paths=True),
    )


def outlook(
    profile: FirmProfile, policy: Policy, years: float | None = None, paths: int = 20000, seed: int = 0,
    largest_raise: float | None = None, counts: dict | None = None, numerics: NumericsProfile = REFERENCE,
) -> Outlook:
    """The read-out, the stresses of `STRESSES` and, with `years`, the paths."""
    model = to_model(profile)
    check(policy, model)
    scenarios = scenario_set(model.params.per_period(), numerics, **(SCENARIO_COUNTS if counts is None else counts))
    return Outlook(
        read_out=read_out(model, policy, scenarios, numerics),
        stresses=tuple(run(model, policy, s, numerics) for s in STRESSES),
        paths=None if years is None else simulate(
            model, rule_from(policy, model, largest_raise), years=years, paths=paths, seed=seed, profile=numerics
        ),
    )


@dataclass(frozen=True)
class Sensitivity:
    """What each output is as entered (`base`) and with each dial at the low
    and at the high end of its range (`low`, `high`): each is
    {output name: value} or {dial name: {output name: value}}. With `versus`
    the values are changes from the policy to the other."""

    dials: tuple[Dial, ...]
    outputs: tuple[Output, ...]
    base: dict[str, float]
    low: dict[str, dict[str, float]]
    high: dict[str, dict[str, float]]
    versus: bool

    def swings(self, output: str) -> list[tuple[Dial, float, float]]:
        """The dials that move `output`, the one that moves it most first,
        each with the output's value at the dial's low and high end."""
        moved = [(d, self.low[d.name][output], self.high[d.name][output]) for d in self.dials]
        scale = max(abs(self.base[output]), *(max(abs(lo), abs(hi)) for _, lo, hi in moved), 1e-300)
        moved = [(d, lo, hi) for d, lo, hi in moved if abs(hi - lo) > 1e-9 * scale or abs(lo - self.base[output]) > 1e-9 * scale]
        return sorted(moved, key=lambda x: -abs(x[2] - x[1]))

    def unaffected(self, output: str) -> list[str]:
        """The dials `output` does not depend on."""
        moving = {d.name for d, _, _ in self.swings(output)}
        return [d.name for d in self.dials if d.name not in moving]


def sensitivity(
    profile: FirmProfile, policy: Policy, dials: tuple[Dial, ...] | None = None,
    outputs: tuple[Output, ...] | None = None, versus: Policy | None = None, years: float | None = None,
    paths: int = 20000, seed: int = 0, largest_raise: float | None = None, counts: dict | None = None,
    numerics: NumericsProfile = REFERENCE,
) -> Sensitivity:
    """Each dial to the low and the high end of its range, one at a time, the
    rest as entered. Outputs that need paths are left out unless `years` is
    given; with it, every run uses the same random paths."""
    dials = standard_dials(profile) if dials is None else dials
    outputs = standard_outputs() if outputs is None else outputs
    outputs = tuple(o for o in outputs if years is not None or not o.needs_paths)

    def values(p: FirmProfile) -> dict[str, float]:
        def of(x: Policy) -> Outlook:
            return outlook(p, x, years, paths, seed, largest_raise, counts, numerics)

        first = of(policy)
        if versus is None:
            return {o.name: o.pick(first) for o in outputs}
        second = of(versus)
        return {o.name: o.pick(second) - o.pick(first) for o in outputs}

    return Sensitivity(
        dials=dials, outputs=outputs, base=values(profile),
        low={d.name: values(d.apply(profile, d.low)) for d in dials},
        high={d.name: values(d.apply(profile, d.high)) for d in dials},
        versus=versus is not None,
    )


def report(s: Sensitivity, top: int = 5) -> str:
    """For each output: its value as entered, then the assumptions that move
    it most, with its value at each end of their range. Dollars in millions."""
    def value(kind: str, x: float) -> str:
        sign = "+" if s.versus else ""
        return f"{x / 1e6:{sign}.2f}" if kind == "money" else f"{100 * x:{sign}.2f}{'pt' if s.versus else '%'}"

    def setting(d: Dial, x: float) -> str:
        return f"{x:.2f}x" if d.unit == "x" else f"{x:.2f} a year" if d.unit == "a year" else f"{100 * x:.0f}%"

    lines = [
        "Each assumption is turned to the low and the high end of a wide range, one at a time; the rest stay as entered.",
        "The ranges are not estimates. Dollars in millions."
        + (" Every figure is the change from the first policy to the second." if s.versus else ""),
    ]
    for o in s.outputs:
        swings = s.swings(o.name)
        lines += ["", f"{o.name}: {value(o.kind, s.base[o.name])} as entered"]
        for d, lo, hi in swings[:top]:
            lines.append(
                f"    {d.name:<42} {setting(d, d.low):>12} -> {value(o.kind, lo):>10}   "
                f"{setting(d, d.high):>12} -> {value(o.kind, hi):>10}"
            )
        rest = len(swings) - top
        still = len(s.unaffected(o.name))
        tail = []
        if rest > 0:
            tail.append(f"{rest} more move it less")
        if still:
            tail.append(f"{still} do not move it at all")
        if tail:
            lines.append("    (" + "; ".join(tail) + ")")
    return "\n".join(lines)
