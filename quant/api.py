"""The tool's interface for other programs: plain data in, plain data out
(docs/plan-tool.md step 9).

`respond` takes a **request** -- a dictionary of the kind JSON carries -- and
returns a **response** of the same kind. A dashboard, a notebook or another
program can drive the whole tool through it without knowing how the model is
built. Nothing here computes anything new: it reads the request into a
profile (`quant.profile`) and a policy (`quant.policy`), calls the read-out,
the stresses, the paths, the comparison and the sensitivity, and writes what
they return as plain data.

A request:

    {
      "firm":   {...},            the firm's facts; all of them (see `example_request`)
      "market": {...},            optional: only the assumptions to change
      "risk":   {...},            optional: likewise
      "policy": {...},            optional: only the decisions to change from the starting policy
      "second_policy": {...},     optional: changes from "policy", to compare the two
      "want": ["read_out", "stress", "paths", "compare", "sensitivity"],
      "years": 5, "paths": 20000, "seed": 0,       for "paths", and for the others' horizon
      "equity_buffer": ..., "largest_raise": ...,   optional: the rule over several years
      "stresses": [{"name": ..., "btc_crash": 0.5}, ...]      optional: in place of the standard set
    }

Assumptions not given are the model's illustrative defaults, sized to the
firm. Money is in dollars and rates are per year, as in a profile.

A response has `"ok": true`, the inputs as the tool understood them (every
default filled in, so a form can show them), and one entry per thing wanted.
If the request cannot be run it has `"ok": false` and `"problems"`: a list of
plain sentences. A mistake in a request never raises.

`VERSION` changes when a request or a response changes shape.
"""

import dataclasses
import math
from dataclasses import fields, is_dataclass, replace

import torch

from quant import compare as compare_
from quant import paths as paths_
from quant import policy as policy_
from quant import readout, sensitivity as sensitivity_
from quant import stress as stress_
from quant.numerics import REFERENCE, NumericsProfile
from quant.profile import (
    FirmFacts,
    FirmProfile,
    GrcEffect,
    LoanPoint,
    MarketAssumptions,
    ModelFirm,
    RaiseCost,
    RiskAssumptions,
    example_facts,
    illustrative,
    to_model,
)

VERSION = 1
WANTS = ("read_out", "stress", "paths", "compare", "sensitivity")
_KEYS = {"firm", "market", "risk", "policy", "second_policy", "want", "years", "paths", "seed",
         "equity_buffer", "largest_raise", "stresses"}
_NESTED = {"small_raise": RaiseCost, "large_raise": RaiseCost, "loan_demand": LoanPoint,
           **{name: GrcEffect for name in sensitivity_.GRC_EFFECTS}}


class _Problems(Exception):
    """A request that cannot be run, with the reasons."""

    def __init__(self, found: list[str]):
        super().__init__("; ".join(found))
        self.found = found


# -- plain data out -----------------------------------------------------------------------


def plain(x):
    """A result as data JSON can carry: dataclasses become dictionaries (with
    their computed properties), tuples lists, tensors lists of numbers, and
    numbers that are not finite become null."""
    if isinstance(x, torch.Tensor):
        return plain(x.tolist())
    if is_dataclass(x) and not isinstance(x, type):
        out = {f.name: plain(getattr(x, f.name)) for f in fields(x)}
        for name in dir(type(x)):
            if isinstance(getattr(type(x), name), property):
                out[name] = plain(getattr(x, name))
        return out
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _paths(r: paths_.PathResult) -> dict:
    """Paths as what a chart needs: by period, the 5%, 25%, 50%, 75% and 95%
    points and the chances, not every path."""
    n = r.decisions_per_year
    value, error = r.value()
    cumulative = lambda x: [0.0] + torch.cumsum(x.mean(dim=1), dim=0).tolist()   # noqa: E731
    return {
        "rule": plain(r.rule), "years": r.years, "decisions_per_year": n, "paths": r.paths, "seed": r.seed,
        "quantiles": list(paths_.QUANTILES),
        "year": [t / n for t in range(r.periods + 1)],
        "equity": plain(r.quantiles(r.equity)),
        "dollar_vault": plain(r.quantiles(r.dollar_vault, alive_only=True)),
        "btc_vault": plain(r.quantiles(r.btc_vault, alive_only=True)),
        "btc_price": plain(r.quantiles(r.btc_price)),
        "chance_failed": plain(r.chance_failed()),
        "chance_raised": [0.0] + plain(r.chance_raised()),
        "paid_out": cumulative(r.payout),
        "raised": cumulative(r.raised),
        "cost_of_raising": cumulative(r.raise_cost),
        "bad_debt": cumulative(r.bad_debt),
        "shareholders_take": value, "shareholders_take_error": error,
    }


def _sensitivity(s: sensitivity_.Sensitivity) -> dict:
    return {
        "versus": s.versus,
        "dials": [{"name": d.name, "base": d.base, "low": d.low, "high": d.high, "unit": d.unit} for d in s.dials],
        "outputs": [{"name": o.name, "kind": o.kind} for o in s.outputs],
        "base": plain(s.base), "low": plain(s.low), "high": plain(s.high),
        "moved_by": {o.name: [d.name for d, _, _ in s.swings(o.name)] for o in s.outputs},
    }


# -- plain data in ----------------------------------------------------------------------------


def _build(cls, given, where: str, base=None):
    """A dataclass from a dictionary: every field if there is no `base`, else
    `base` with the fields given changed. Unknown names are problems."""
    if not isinstance(given, dict):
        raise _Problems([f"'{where}' must be a set of named values"])
    names = {f.name for f in fields(cls)}
    found = [f"'{where}' has no input called '{k}'" for k in given if k not in names]
    if base is None:
        required = [f.name for f in fields(cls) if f.default is dataclasses.MISSING]
        found += [f"'{where}' needs '{k}'" for k in required if k not in given]
    if found:
        raise _Problems(found)
    values = {}
    for key, value in given.items():
        nested = _NESTED.get(key) if cls is not policy_.Policy else None
        if nested is not None:
            values[key] = _build(nested, value, f"{where}.{key}")
        elif isinstance(value, list):
            values[key] = tuple(value)
        else:
            values[key] = value
    try:
        return cls(**values) if base is None else replace(base, **values)
    except TypeError as error:
        raise _Problems([f"'{where}': {error}"]) from None


def read_profile(request: dict) -> FirmProfile:
    """The request's firm, with the assumptions it gives and the model's
    illustrative defaults for the rest."""
    if "firm" not in request:
        raise _Problems(["the request needs a 'firm': the firm's own facts"])
    facts = _build(FirmFacts, request["firm"], "firm")
    try:
        base = illustrative(facts)
    except (ValueError, ZeroDivisionError, TypeError) as error:
        raise _Problems([f"the firm's facts cannot be used: {error}"]) from None
    return FirmProfile(
        facts=facts,
        market=_build(MarketAssumptions, request.get("market", {}), "market", base.market),
        risk=_build(RiskAssumptions, request.get("risk", {}), "risk", base.risk),
    )


def read_policy(given: dict, model: ModelFirm, where: str, base: policy_.Policy | None = None) -> policy_.Policy:
    """A policy: the starting policy (or `base`) with the decisions given
    changed. Unsound, it is refused with every reason."""
    policy = _build(policy_.Policy, given, where, policy_.starting_policy(model) if base is None else base)
    found = policy_.problems(policy, model)
    if found:
        raise _Problems([f"{where}: {x}" for x in found])
    return policy


# -- the interface --------------------------------------------------------------------------------


def respond(request: dict, numerics: NumericsProfile = REFERENCE) -> dict:
    """Answer a request (module docstring). Never raises on a mistaken
    request: it answers `{"ok": false, "problems": [...]}`."""
    try:
        return {"ok": True, "version": VERSION, **_answer(request, numerics)}
    except _Problems as error:
        return {"ok": False, "version": VERSION, "problems": error.found}
    except ValueError as error:           # the model's own refusals, already in plain words
        return {"ok": False, "version": VERSION, "problems": [str(error)]}


def _answer(request: dict, numerics: NumericsProfile) -> dict:
    if not isinstance(request, dict):
        raise _Problems(["a request must be a set of named values"])
    found = [f"the request has no entry called '{k}'" for k in request if k not in _KEYS]
    want = request.get("want", ["read_out"])
    if not isinstance(want, list) or not want:
        found.append("'want' must be a list of what to compute")
    else:
        found += [f"'want' can list {', '.join(WANTS)}; not '{w}'" for w in want if w not in WANTS]
    if found:
        raise _Problems(found)

    profile = read_profile(request)
    model = to_model(profile)
    policy = read_policy(request.get("policy", {}), model, "policy")
    second = None
    if "second_policy" in request:
        second = read_policy(request["second_policy"], model, "second_policy", policy)
    if "compare" in want and second is None:
        raise _Problems(["'compare' needs a 'second_policy'"])

    years = request.get("years")
    n_paths, seed = int(request.get("paths", 20000)), int(request.get("seed", 0))
    largest_raise = request.get("largest_raise")
    if "paths" in want and years is None:
        raise _Problems(["'paths' needs 'years': how far ahead to look"])
    if years is not None and not (isinstance(years, (int, float)) and years > 0):
        raise _Problems(["'years' must be a positive number"])
    if n_paths < 2:
        raise _Problems(["'paths' must be at least 2"])
    stresses = None
    if "stresses" in request:
        if not isinstance(request["stresses"], list) or not request["stresses"]:
            raise _Problems(["'stresses' must be a list of stresses"])
        stresses = tuple(_build(stress_.Stress, s, "stresses") for s in request["stresses"])

    def rule(x: policy_.Policy) -> paths_.Rule:
        made = paths_.rule_from(x, model, largest_raise)
        return made if request.get("equity_buffer") is None else replace(made, equity_buffer=request["equity_buffer"])

    answer = {
        "inputs": {"firm": plain(profile.facts), "market": plain(profile.market), "risk": plain(profile.risk),
                   "policy": plain(policy), "position": plain(policy_.position(policy, model)),
                   **({} if second is None else {"second_policy": plain(second)})},
        "illustrative": True,     # until the assumptions are calibrated (docs/plan-tool.md step 8)
    }
    if "read_out" in want:
        answer["read_out"] = plain(readout.read_out(model, policy, profile=numerics))
    if "stress" in want:
        answer["stress"] = plain(stress_.run_all(model, policy, stresses, numerics))
    if "paths" in want:
        answer["paths"] = _paths(paths_.simulate(model, rule(policy), years, n_paths, seed, numerics))
    if "compare" in want:
        c = compare_.compare(model, policy, second, stresses=stresses, profile=numerics)
        answer["compare"] = {"read_out": plain(c.read_outs), "stress": plain(c.stresses)}
        if years is not None:
            a, b = (paths_.simulate(model, rule(x), years, n_paths, seed, numerics) for x in (policy, second))
            gap = b.present_value - a.present_value
            answer["compare"]["paths"] = [_paths(a), _paths(b)]
            answer["compare"]["shareholders_take_gain"] = float(gap.mean())
            answer["compare"]["shareholders_take_gain_error"] = float(gap.std() / math.sqrt(gap.shape[0]))
    if "sensitivity" in want:
        answer["sensitivity"] = _sensitivity(sensitivity_.sensitivity(
            profile, policy, versus=second, years=years, paths=n_paths, seed=seed,
            largest_raise=largest_raise, numerics=numerics,
        ))
    return answer


def example_request() -> dict:
    """A complete request for the made-up example firm: every fact, one
    assumption changed, a policy, and everything wanted but the sensitivity."""
    return {
        "firm": plain(example_facts()),
        "risk": {"incidents_per_year": 0.15},
        "policy": {"borrower_rate": 0.09, "ltv": 0.6},
        "second_policy": {"ltv": 0.7, "allocation": [0.25, 0.25, 0.25, 0.25]},
        "want": ["read_out", "stress", "paths", "compare"],
        "years": 5, "paths": 20000, "seed": 0,
    }


def inputs() -> dict:
    """The names of every input a request can carry, by part, with the
    example firm's values: enough to lay out a form."""
    profile = illustrative(example_facts())
    model = to_model(profile)
    return {
        "version": VERSION,
        "firm": plain(profile.facts), "market": plain(profile.market), "risk": plain(profile.risk),
        "policy": plain(policy_.starting_policy(model)),
        "stress": plain(stress_.Stress(name="a name")),
        "want": list(WANTS),
        "allocation_order": list(policy_.ALLOCATION),
    }
