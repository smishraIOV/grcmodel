#!/usr/bin/env python3
"""Solve the opening GRC stock a configuration supports, and print it.

`FirmParams.initial_grc_stock` is not a property of the firm. It is a fixed
point of the whole model -- the level at which the firm's own optimal
maintenance spend exactly replaces depreciation:

    G = g*(G) / delta

where `g*(G)` is the per-family spend a firm opening at `G` chooses. It moves
with the hazards, with the loss channels *and* with the horizon, which is why
`quant/params.GRC_STEADY_STATE` is keyed by `(periods_per_year, horizon)` and
why `steady_state_firm` raises on a combination nobody has solved.

Until now the iteration existed only as a sentence in a comment in
`quant/params.py`, and CLAUDE.md, `docs/calibration.md` and
`docs/quant-model.md` all make re-solving *mandatory* after any change that
moves optimal spend. A mandatory step with no tooling is a step that gets
skipped: it already was once, and the firm then opened six times above its own
steady state, spent the horizon running the stock down, and reported a budget
78% lower as a headline finding. Re-solved, the fall was 5%.

This prints the passes and the line to paste. It deliberately does **not**
write to `quant/params.py`: the value is load-bearing enough that a human
should see the regime diagnostics next to it before it lands.

**What the map actually looks like**, measured at eight points on one draw at
monthly/five years, 1024 paths and 2500 steps:

    G     0.600  0.762  1.050  1.120  1.200  1.300  1.577  1.687
    f(G)  1.687  1.577  1.374  1.117  1.061  0.997  0.833  0.762

Three things follow, and each sets a default here.

**The slope is about -0.9.** So plain iteration does converge, but it
oscillates and contracts by only a tenth a pass. Hence `--damping 0.5`, which
puts the effective slope near 0.05.

**A single draw cannot locate the root.** Regressing those eight points leaves
a residual RMS of 0.066 in stock units -- about 6% of spend -- against a root
that sits somewhere near 1.17 and moves by only ~0.05 between plausible
answers. Two *adjacent* points carried residuals of +0.098 and -0.096, which
between them fake a kink in the map and a factor-of-four change in its slope.
Hence `--seeds`, and hence the warning when it is 1.

**The stopping rule has to know about that noise.** The first run of this
script stopped after a single pass, on a move of 0.23%, and reported the value
it started from -- which is what a 1% tolerance does against 6% noise. It is
not a converged answer, it is a draw that happened to land nearby. Convergence
is now judged against `max(tolerance, seed spread)` and the binding one is
named.

Usage:

    uv run python scripts/solve_steady_state.py --seeds 3            # monthly, 5y
    uv run python scripts/solve_steady_state.py --frequency quarterly --periods 8
    uv run python scripts/solve_steady_state.py --from 1.16 --seeds 3 --steps 4000
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.cli import print_header
from quant.env.env import EnvConfig, FirmEnv
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.numerics import DEFAULT_PROFILE, PROFILES, get_profile
from quant.params import DEFAULTS, GRC_STEADY_STATE, MONTHLY
from quant.solvers.pathwise import optimize_constant

FREQUENCIES = {"quarterly": DEFAULTS, "monthly": MONTHLY}

# The band outside which the survival comparative statics stop meaning anything
# (docs/quant-model.md appendix A6). Checked at the fixed point rather than
# assumed, because both neighbouring regimes are degenerate in opposite ways: a
# firm rebuilding from nothing dies of tail events while it does, and one handed
# too much capital spends nothing and the model has no budget question left.
DEATH_BAND = (0.002, 0.15)

# Below this, treat the run as a smoke test rather than a solve. `optimize_constant`
# starts every control at `softplus(0)` and Adam moves a raw parameter by at most
# `lr` per step, so a short budget does not converge -- it stops somewhere and
# reports where it stopped as though the firm had chosen it (CLAUDE.md). That
# failure is quiet: the pass table fills in, a fixed point is printed, and only
# the regime rows look odd. This script's whole output is a number someone pastes
# into `quant/params.py`, so it says so loudly instead.
MINIMUM_STEPS = 1500


def map_at(params, periods, profile, paths, steps, stock, seeds):
    """`f(G) = g*(G) / delta`, averaged over scenario draws.

    Averaged because a single draw is not accurate enough to locate this root,
    which was measured rather than assumed. Evaluating the map at eight points
    on one draw and regressing gives a residual RMS of 0.066 in stock units --
    about 6% of spend -- against a root that moves by only ~0.05 between
    plausible answers. Two adjacent points carried residuals of +0.098 and
    -0.096, which is enough to fake a kink in the map and a factor-of-four
    change in its slope.

    Returns `(mean, spread, last result)`. The spread is what the caller needs
    to know whether a pass actually moved.
    """
    sampler = MonteCarloSampler(params.sampler)
    depreciation = params.firm.grc_depreciation()
    firm = replace(params.firm, initial_grc_stock=stock)
    env = FirmEnv(EnvConfig.at_frequency(params, periods, firm=firm), sampler)

    implied, result = [], None
    for seed in seeds:
        crn = CommonRandomNumbers(seed, periods, paths, profile)
        result = optimize_constant(env, crn, n_steps=steps)[1]
        implied.append(implied_stock(result, depreciation))

    mean = sum(implied) / len(implied)
    spread = max(implied) - min(implied) if len(implied) > 1 else 0.0
    return mean, spread, result


def implied_stock(result, depreciation: float) -> float:
    """`g*(G) / delta`, the stock this much maintenance spend sustains.

    Per family, because `initial_grc_stock` is per family -- `FirmEnv.reset`
    opens every family at it. The three families do not spend equally, so this
    is the stock the *average* family's maintenance supports and the fixed point
    is a statement about one scalar rather than three.
    """
    return (sum(result.grc) / len(result.grc)) / depreciation


def solve(params, periods, profile, paths, steps, start, passes, damping,
          tolerance, seeds):
    """Damped fixed-point iteration, printing every pass.

    Damped because the measured slope is about -0.9: plain iteration converges,
    but it oscillates and contracts by only ~10% a pass, so from a start half a
    unit away it takes longer than anyone will wait. At `damping = 0.5` the
    effective slope is `(1 - d) + d*f'` = 0.05 and two passes are enough.

    **The stopping rule is the part that needs care.** A pass that moves the
    stock less than the seed spread has not converged -- it has drawn a scenario
    set that happened to land near where it started. Measured on one draw, the
    noise in `f` is about 0.066 against a 1% tolerance of 0.011, so a
    single-seed run stops on luck roughly whenever it feels like it. It did
    exactly that the first time this script was run: it declared convergence
    after one pass, on a move of 0.23%, at a value the regression later put
    ~4% away. So convergence is judged against `max(tolerance, spread)` and the
    reason it stopped is printed.
    """
    depreciation = params.firm.grc_depreciation()
    multi = len(seeds) > 1

    header = (
        f"  {'pass':>4} | {'opening G':>10} | {'spend/prd':>10} | {'implied G':>10} | "
        f"{'spread':>8} | {'move':>8} | {'annual death':>13} | {'going concern':>14}"
    )
    print(f"\nper-period depreciation {depreciation:.5f}, "
          f"so a stock of G needs {depreciation:.5f} x G a period to hold it")
    print(f"map evaluated over {len(seeds)} seed(s): {list(seeds)}\n")
    print(header)
    print("  " + "-" * (len(header) - 2))

    stock, last, stopped = start, None, None
    for index in range(passes):
        implied, spread, result = map_at(
            params, periods, profile, paths, steps, stock, seeds
        )
        move = implied - stock
        print(
            f"  {index:>4} | {stock:>10.4f} | {result.total_grc:>10.4f} | "
            f"{implied:>10.4f} | {spread if multi else float('nan'):>8.4f} | "
            f"{move:>+8.4f} | {result.annual_death_probability:>12.2%} | "
            f"{result.going_concern_share:>14.3f}"
        )
        last = result

        floor = max(tolerance * max(stock, 1e-9), spread)
        if abs(move) < floor:
            stopped = "the seed spread" if spread > tolerance * stock else "the tolerance"
            stock = implied
            break
        stock = stock + damping * move

    return stock, last, stopped, multi


def report(stock, result, params, periods, steps, stopped, multi):
    """The fixed point, the regime check, and the line to paste."""
    if stopped is None:
        print(
            f"\nDID NOT CONVERGE: {stock:.4f} is where the passes ran out, not a\n"
            "  fixed point. Re-run with --from that value and more --passes."
        )
    else:
        print(f"\nFixed point: {stock:.4f}   (stopped on {stopped})")

    if not multi:
        print(
            "\n  ONE SCENARIO DRAW. The noise between draws is comparable to the\n"
            "  distance between plausible answers here -- measured, about 6% of\n"
            "  spend against a root that moves by ~4%. Re-run with --seeds 3\n"
            "  before pasting anything."
        )
    if stopped == "the seed spread":
        print(
            "\n  STOPPED INSIDE THE NOISE: the last pass moved the stock by less\n"
            "  than the spread across seeds, so this is where the iteration ran\n"
            "  out of resolution rather than where it converged. Raise --steps or\n"
            "  --paths to sharpen the map before believing the last digit."
        )
    if steps < MINIMUM_STEPS:
        print(
            f"\n  NOT A SOLVE: {steps} optimizer steps, against a floor of "
            f"{MINIMUM_STEPS}.\n"
            "  The controls start at softplus(0) and Adam moves them by at most\n"
            "  `lr` a step, so this reports where the optimizer stopped rather\n"
            "  than what the firm would choose. Do not paste it."
        )

    death = result.annual_death_probability
    low, high = DEATH_BAND
    checks = [
        ("annual death probability", f"{death:.2%}", low < death < high,
         f"inside {low:.1%}-{high:.0%}"),
        ("going-concern share", f"{result.going_concern_share:.3f}",
         0.0 < result.going_concern_share < 1.0, "off 0 and 1"),
        ("underinvestment fraction", f"{result.underinvestment_fraction:.3f}",
         result.underinvestment_fraction < 1.0, "not pinned at 1"),
        ("reserve ratio", f"{result.reserve_ratio:.1%}",
         0.0 < result.reserve_ratio < 1.0, "off 0 and 1"),
        ("leverage", f"{result.leverage:.2f}", result.leverage >= 1.0, "at least 1"),
    ]
    print("\nRegime check at the fixed point (docs/quant-model.md appendix A6):")
    for name, value, passed, expectation in checks:
        print(f"  {'ok  ' if passed else 'WARN'} {name:>26} = {value:>8}   ({expectation})")
    if not all(passed for _, _, passed, _ in checks):
        print(
            "\n  A failed row does not make the fixed point wrong -- it makes the\n"
            "  configuration one whose comparative statics should not be quoted."
        )

    key = (params.firm.periods_per_year, periods)
    known = GRC_STEADY_STATE.get(key)
    print(f"\nPaste into GRC_STEADY_STATE in quant/params.py:\n")
    print(f"    {key}: {stock:.2f},")
    if known is not None:
        print(
            f"\n  (currently {known}; this run says {stock:.2f}, "
            f"a move of {stock - known:+.2f})"
        )


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
    parser.add_argument(
        "--from", dest="start", type=float, default=None,
        help="opening guess (default: the stored value for this configuration, "
             "or the dataclass default if there is none)",
    )
    parser.add_argument("--passes", type=int, default=6)
    parser.add_argument(
        "--seed", type=int, default=0, help="base scenario seed"
    )
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="how many consecutive seeds to average the map over. One draw is\n"
             "not enough to locate this root -- see map_at()",
    )
    parser.add_argument(
        "--damping", type=float, default=0.5,
        help="fraction of each step to take. The measured slope is about -0.9, "
             "so 0.5 gives an effective slope near 0.05 and 1.0 oscillates",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.01,
        help="stop when a pass moves the stock by less than this, relatively",
    )
    parser.add_argument("--paths", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2500)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    print_header(profile)

    params = FREQUENCIES[args.frequency]
    periods = args.periods or 5 * params.firm.periods_per_year
    # Deliberately not `steady_state_firm`: that looks the value up and raises
    # on a configuration nobody has solved, which is the right behaviour
    # everywhere except here, where solving it is the job.
    start = args.start
    if start is None:
        start = GRC_STEADY_STATE.get(
            (params.firm.periods_per_year, periods), params.firm.initial_grc_stock
        )

    print(
        f"{periods} {args.frequency} periods "
        f"({periods / params.firm.periods_per_year:.0f} years), {args.paths} paths, "
        f"{args.steps} steps, starting from {start}"
    )

    seeds = tuple(range(args.seed, args.seed + max(1, args.seeds)))
    stock, result, stopped, multi = solve(
        params, periods, profile, args.paths, args.steps,
        start, args.passes, args.damping, args.tolerance, seeds,
    )
    report(stock, result, params, periods, args.steps, stopped, multi)

    print(
        "\nThis is a fixed point of the whole model, so it is invalidated by any\n"
        "change that moves optimal spend -- a hazard, a loss channel, or the\n"
        "horizon. A stale value does not error; it produces a firm that spends\n"
        "the horizon running down controls it would never have built, and\n"
        "reports that as a budget (CLAUDE.md)."
    )


if __name__ == "__main__":
    main()
