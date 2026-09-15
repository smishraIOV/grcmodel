#!/usr/bin/env python3
"""Does state feedback earn its place? Measured across seeds, scored out of sample.

Every other number this project publishes comes from one draw of the scenarios
and one optimizer start. This reports a median, a confidence interval and the
range, so a difference between solvers can be told apart from a difference
between runs.

Usage: uv run python scripts/run_seed_study.py [--frequency monthly] [--seeds 5]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.cli import print_header
from quant.env.env import EnvConfig, FirmEnv
from quant.env.shocks import MonteCarloSampler
from quant.numerics import DEFAULT_PROFILE, PROFILES, get_profile
from quant.params import DEFAULTS, MONTHLY, steady_state_firm
# Decision frequencies with a solved GRC steady state.
FREQUENCIES = {"quarterly": DEFAULTS, "monthly": MONTHLY}

from quant.studies.seeds import edge, header, run_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default=DEFAULT_PROFILE.name, choices=sorted(PROFILES))
    parser.add_argument(
        "--frequency", default="monthly", choices=sorted(FREQUENCIES),
        help="how often the firm decides (default: monthly, the main line)",
    )
    parser.add_argument(
        "--periods", type=int, nargs="+", default=None,
        help="horizons in periods (default: the solved configuration)",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-paths", type=int, default=512)
    parser.add_argument("--test-paths", type=int, default=4096)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    print_header(profile)
    sampler = MonteCarloSampler(FREQUENCIES[args.frequency].sampler)
    seeds = tuple(range(args.seeds))

    params = FREQUENCIES[args.frequency]
    horizons = args.periods or [5 * params.firm.periods_per_year]
    for periods in horizons:
        firm = steady_state_firm(params, periods)
        env = FirmEnv(EnvConfig.at_frequency(params, periods, firm=firm), sampler)
        studies = run_seeds(
            env, seeds=seeds, train_paths=args.train_paths, test_paths=args.test_paths
        )
        print(
            f"\n{periods} {args.frequency} periods, {len(seeds)} seeds, trained on {args.train_paths} paths "
            f"and scored on {args.test_paths} held out"
        )
        print("  " + header())
        for study in studies.values():
            print("  " + study.line())
        advantage = edge(studies)
        # Compared against the spread this run actually measured, not against a
        # fixed 2%. The constant against which "clear of seed noise" means
        # anything is the noise, and it is not a constant: at thirty-two
        # quarters the neural policy's interval reached +/-37% of its own median
        # because one seed collapsed to 0.9, and a fixed threshold called a
        # +4.8% median difference clear of it.
        noise = max(s.half_width() / abs(s.median()) for s in studies.values())
        verdict = (
            "within seed noise" if abs(advantage) < max(noise, 0.02)
            else "clear of seed noise"
        )
        print(
            f"  state feedback: {advantage:+.2%} at the median, against seed noise of "
            f"{noise:.2%} -- {verdict}"
        )
        worst = min(min(s.held_out) for s in studies.values())
        if worst < 0.5 * max(max(s.held_out) for s in studies.values()):
            print(
                f"  WARNING: a seed landed at {worst:.3f}, less than half the "
                "best. One solve failed rather than merely varied; read the "
                "range, not the median."
            )

    print(
        "\nThe interval is what decides whether a solver difference is real. A single"
        "\nrun cannot tell a better policy from a luckier seed."
    )


if __name__ == "__main__":
    main()
