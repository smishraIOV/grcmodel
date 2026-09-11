#!/usr/bin/env python3
"""Multi-period model: what changes when GRC persists (docs/quant-model.md 5.2).

The static model charged GRC as a per-period expense that bought exactly one
period of protection. Over a horizon that is the wrong shape: a control
installed this quarter still works next quarter. Making GRC a depreciating
capital stock is the smallest change that turns the repeated static problem
into an intertemporal one, and it is what gives pre-emptive spend option value.

Three solvers, all scored by the same evaluate():

  constant    the best state-independent action. The floor a learner must beat.
  neural      state feedback, trained by backpropagation through the rollout.
  PI bound    every control chosen per path with the future known. Not
              implementable -- it is an upper bound on what any policy can do.

The sandwich V(constant) <= V(neural) <= V_PI is the cheapest correctness
check available: a learner above the bound has an information leak, a
mis-signed discount or a death that failed to absorb, none of which shows up
in the value on its own.

Usage: uv run python scripts/run_dynamic_model.py [--quarters 8] [--paths 2048]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace

from quant.cli import print_header
from quant.env.env import EnvConfig, FirmEnv, evaluate
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.numerics import DEFAULT_PROFILE, PROFILES, get_profile
from quant.params import DEFAULTS
from quant.solvers.neural import train_pathwise
from quant.solvers.pathwise import optimize_constant, perfect_information_bound


def annual_death(survival_rate: float, quarters: int) -> float:
    """Survival over the horizon, re-expressed as a probability per year."""
    return 1.0 - survival_rate ** (4.0 / quarters)


def survival_channel(env, naive_env, crn, steps: int, quarters: int) -> None:
    """What it costs to budget as though GRC only reduced expected loss.

    Both policies are scored in the same world -- the one where GRC also
    reduces failure intensity. Comparing them in their own worlds would be
    meaningless, since the loss-only world is simply less dangerous.
    """
    naive = optimize_constant(naive_env, crn, n_steps=steps)[0]
    aware, aware_result = optimize_constant(env, crn, n_steps=steps)
    naive_result = evaluate(naive, env, crn)

    print("\nWhere GRC acts: budgeting for losses only, vs for losses and survival")
    header = f"  {'budget set for':>16} | {'spend/qtr':>10} | {'annual death':>13} | {'value':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, r in (("expected loss", naive_result), ("loss + survival", aware_result)):
        print(
            f"  {label:>16} | {r.total_grc:>10.4f} | {annual_death(r.survival_rate, quarters):>12.2%} | "
            f"{r.value:>8.4f}"
        )
    print(
        f"  Ignoring the survival channel costs {1 - naive_result.value / aware_result.value:.1%} of firm "
        f"value and raises\n  the annual failure probability by "
        f"{annual_death(naive_result.survival_rate, quarters) - annual_death(aware_result.survival_rate, quarters):.2%}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default=DEFAULT_PROFILE.name, choices=sorted(PROFILES))
    parser.add_argument("--quarters", type=int, default=8)
    parser.add_argument("--paths", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    print_header(profile)

    firm = DEFAULTS.firm
    sampler = MonteCarloSampler(DEFAULTS.sampler)
    crn = CommonRandomNumbers(0, args.quarters, args.paths, profile)

    print(
        f"{args.quarters} quarters, {args.paths} paths, discount {firm.discount():.4f}/quarter, "
        f"insolvency barrier at 0"
    )
    print(
        "GRC decay 1.000 charges spend as a one-period expense (the static model's\n"
        f"assumption); {firm.grc_depreciation():.3f} is the quarterly rate implied by "
        f"{firm.annual_grc_depreciation:.0%} a year.\n"
    )

    header = f"{'GRC decay':>11} | {'solver':>9} | {'value':>8} | {'spend/qtr':>10} | {'end stock':>10} | {'survives':>9}"
    print(header)
    print("-" * len(header))

    summary = {}
    for delta in (1.0, firm.grc_depreciation()):
        env = FirmEnv(
            EnvConfig.quarterly(args.quarters, grc_depreciation=delta), sampler, 
        )
        results = {
            "constant": optimize_constant(env, crn, n_steps=args.steps)[1],
            "neural": train_pathwise(env, crn, n_steps=max(1, args.steps // 2)).evaluation,
            "PI bound": perfect_information_bound(env, crn, n_steps=args.steps)[1],
        }
        for name, r in results.items():
            print(
                f"{delta:>11.3f} | {name:>9} | {r.value:>8.4f} | {r.total_grc:>10.4f} | "
                f"{sum(r.grc_stock):>10.4f} | {r.survival_rate:>9.3f}"
            )
        bound = results["PI bound"].value
        for name in ("constant", "neural"):
            assert results[name].value <= bound + 1e-6, f"{name} broke the bound"
        print(f"{'':>11} | EVPI = {bound - results['neural'].value:.4f}")
        print("-" * len(header))
        summary[delta] = results["neural"]

    env = FirmEnv(EnvConfig.quarterly(args.quarters), sampler)
    naive_env = FirmEnv(
        EnvConfig.quarterly(
            args.quarters,
            hazard=replace(DEFAULTS.hazard, annual_operational_rate=0.0, annual_licence_rate=0.0),
        ),
        sampler,
    )
    survival_channel(env, naive_env, crn, args.steps, args.quarters)

    flow, stock = summary[1.0], summary[firm.grc_depreciation()]
    print(
        f"\nPersistence is worth {stock.value / flow.value - 1:+.0%} of firm value "
        f"({flow.value:.2f} -> {stock.value:.2f}) and takes survival from "
        f"{flow.survival_rate:.0%} to {stock.survival_rate:.0%}."
    )
    print(
        f"The firm also spends MORE per quarter ({flow.total_grc:.2f} -> {stock.total_grc:.2f}), "
        "not less:\neach unit now protects every later quarter too, so more of it is worth buying."
    )
    print(
        "\nParameters are illustrative. Read the direction and the mechanism, not the\n"
        "magnitudes (docs/quant-model.md section 8)."
    )


if __name__ == "__main__":
    main()
