#!/usr/bin/env python3
"""Break-evens: what the model says instead of quoting a budget.

Nothing here can calibrate alpha, so an optimal budget computed from an
invented one implies a precision that does not exist. These invert the question
into claims a reader can agree or disagree with.

Usage: uv run python scripts/run_breakevens.py [--alpha-sweep] [--paths 1024]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.cli import print_header
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.solvers.analytic import family_exposures
from quant.numerics import DEFAULT_PROFILE, PROFILES, get_profile
from quant.params import DEFAULTS, MONTHLY, steady_state_firm
from quant.studies.breakeven import (
    solve,
    FAMILIES,
    is_material,
    alpha_breakeven,
    capitalization_band,
    franchise_breakeven,
    hazard_breakeven,
    value_curvature,
)

FREQUENCIES = {"quarterly": DEFAULTS, "monthly": MONTHLY}


def exposures(profile, book: float, params=DEFAULTS):
    """Base-probability-weighted expected loss per family, **per period**.

    Computed rather than hardcoded. They were written in as 4.0 / 3.5 / 1.5,
    which were the static model's magnitudes; after the recalibration the true
    values are about twenty-five times smaller, so the risk-neutral break-even
    1/X was being compared against the wrong denominator entirely.

    Drawn from `params.sampler` rather than `DEFAULTS.sampler`, so the exposures
    are per period *at the frequency being studied*. Pinned to DEFAULTS they
    would have been quarterly magnitudes divided into a monthly optimum, which
    moves every risk-neutral break-even by the frequency ratio.
    """
    crn = CommonRandomNumbers(0, 1, 1 << 16, profile)
    shock = MonteCarloSampler(params.sampler)(crn.at(0), profile)
    return family_exposures(shock, book=book)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default=DEFAULT_PROFILE.name, choices=sorted(PROFILES))
    parser.add_argument(
        "--frequency", default="monthly", choices=sorted(FREQUENCIES),
        help="how often the firm decides (default: monthly, the main line)",
    )
    parser.add_argument(
        "--periods", type=int, default=None,
        help="horizon in periods (default: five years at the chosen frequency)",
    )
    parser.add_argument("--paths", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--alpha-sweep", action="store_true",
        help="bisect the per-family effectiveness break-even (slow: ~2 minutes)",
    )
    args = parser.parse_args()

    profile = get_profile(args.profile)
    print_header(profile)

    params = FREQUENCIES[args.frequency]
    periods = args.periods or 5 * params.firm.periods_per_year
    # The opening GRC stock is a fixed point of the *configuration*, not of the
    # firm, so it is looked up rather than inherited. An unsolved combination
    # raises here instead of silently using whichever default is nearest --
    # which would hand the firm a control stock it would never have built and
    # report the result as a budget.
    firm = steady_state_firm(params, periods)
    print(
        f"{args.frequency} decisions, {periods} periods "
        f"({periods / params.firm.periods_per_year:.0f} years), "
        f"opening GRC stock {firm.initial_grc_stock}"
    )
    crn = CommonRandomNumbers(0, periods, args.paths, profile)
    kw = dict(periods=periods, steps=args.steps, params=params)

    print("1. HAZARD BREAK-EVEN -- the headline, and the only one with no alpha in it")
    print("   " + hazard_breakeven(firm, crn, **kw).sentence())

    print("\n2. CAPITALIZATION BAND -- over what range is a programme a decision?")
    print(f"   {'equity':>8} | {'spend/prd':>10} | {'annual death':>13} | {'going-concern':>14} | {'verdict':>18}")
    for equity, r in capitalization_band(firm, crn, [4.0, 8.0, 16.0, 32.0, 64.0], **kw):
        verdict = "worth running" if is_material(r, firm) else "uneconomic"
        print(f"   {equity:>8.1f} | {r.total_grc:>10.4f} | {r.annual_death_probability:>12.2%} | "
              f"{r.going_concern_share:>14.3f} | {verdict:>18}")
    print("   Spend peaks in the middle. Well capitalized, the hazard is too small to be")
    print("   worth buying down; thinly capitalized, there is too little franchise left to protect.")

    print("\n3. FRANCHISE BREAK-EVEN -- how much business must be at stake?")
    print(f"   {'going concern':>14} | {'spend/prd':>10} | {'firm value':>11} | {'verdict':>18}")
    for franchise, r in franchise_breakeven(firm, crn, [0.0, 3.0, 8.0, 15.0], **kw):
        verdict = "worth running" if is_material(r, firm) else "uneconomic"
        print(f"   {franchise:>14.1f} | {r.total_grc:>10.4f} | {r.value:>11.3f} | {verdict:>18}")

    print("\n4. VALUE CURVATURE -- the gambling-for-resurrection check")
    equities = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    values, curvature = value_curvature(firm, crn, equities, **kw)
    print(f"   {'equity':>8} | {'value':>9} | {'d2V/dE2':>10} | region")
    shape = dict(curvature)
    for equity, value in zip(equities, values):
        second = shape.get(equity)
        label = "" if second is None else (
            "CONVEX -- risk-loving" if second > 0 else "concave -- risk-averse"
        )
        cell = "" if second is None else f"{second:.5f}"
        print(f"   {equity:>8.1f} | {value:>9.3f} | {cell:>10} | {label}")
    worst = max((v for _, v in curvature), default=0.0)
    print(f"   Largest positive second difference: {worst:.5f}. A convex region would mean the")
    print("   firm is risk-loving near failure, and a learner would find it and recommend")
    print("   cutting GRC in a crisis -- correct inside the model, indefensible outside it.")

    if args.alpha_sweep:
        print("\n5. EFFECTIVENESS BREAK-EVEN -- the original question, per family")
        print(f"   {'family':>12} | {'risk-neutral 1/X':>17} | {'with survival':>14} | {'ratio':>7}")
        # Credit exposure is a rate on the book, so the comparison needs the
        # book the firm actually funds, not a bare draw.
        exposure = exposures(profile, book=solve(firm, crn, **kw).book, params=params)
        for family in FAMILIES:
            alpha = alpha_breakeven(firm, crn, family, low=0.0005, iterations=8, **kw)
            neutral = 1.0 / exposure[family]
            cell = "none" if alpha is None else f"{alpha:.4f}"
            ratio = "-" if alpha is None else f"{neutral / alpha:.0f}x"
            print(f"   {family:>12} | {neutral:>17.3f} | {cell:>14} | {ratio:>7}")
        print("   The gap is the Froot-Stein premium as a threshold. Under the survival framing")
        print("   a programme pays at effectiveness an expected-loss calculation rejects outright.")

    print("\nParameters are illustrative. These are thresholds, not forecasts"
          " (docs/quant-model.md section 8).")


if __name__ == "__main__":
    main()
