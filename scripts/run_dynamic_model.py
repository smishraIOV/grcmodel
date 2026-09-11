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

from quant.cli import print_header
from quant.env.env import EnvConfig, FirmEnv
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.numerics import DEFAULT_PROFILE, PROFILES, get_profile
from quant.params import DEFAULTS
from quant.solvers.neural import train_pathwise
from quant.solvers.pathwise import optimize_constant, perfect_information_bound


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
