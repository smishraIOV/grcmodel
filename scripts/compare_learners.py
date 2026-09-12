#!/usr/bin/env python3
"""Full backpropagation versus truncated BPTT with a critic.

The plan called for SVG(K) and stage 5 declined to build it on the evidence
available then. This runs both so the choice rests on a measurement rather than
on either argument.

K is a dial, not a different algorithm: at K = horizon the windowed objective
is bitwise identical to the ordinary rollout and the critic is never consulted.
Smaller K cuts the gradient chain more often and leans harder on the critic.

Usage: uv run python scripts/compare_learners.py [--quarters 8] [--windows 1 2 4]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.cli import print_header
from quant.env.env import EnvConfig, FirmEnv, evaluate
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.numerics import DEFAULT_PROFILE, PROFILES, get_profile
from quant.params import DEFAULTS
from quant.solvers.neural import train_pathwise
from quant.solvers.pathwise import optimize_constant
from quant.solvers.svg import train_svg

HELD_OUT = 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default=DEFAULT_PROFILE.name, choices=sorted(PROFILES))
    parser.add_argument("--quarters", type=int, default=8)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--train-paths", type=int, default=512)
    parser.add_argument("--test-paths", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    print_header(profile)

    quarters = args.quarters
    env = FirmEnv(EnvConfig.quarterly(quarters), MonteCarloSampler(DEFAULTS.sampler))
    train = CommonRandomNumbers(0, quarters, args.train_paths, profile)
    test = CommonRandomNumbers(HELD_OUT, quarters, args.test_paths, profile)

    print(f"{quarters} quarters, trained on {args.train_paths} paths, scored on "
          f"{args.test_paths} held out")
    header = f"  {'solver':>14} | {'held-out':>9} | {'in-sample':>10} | notes"
    print(header)
    print("  " + "-" * (len(header) - 2))

    constant = optimize_constant(env, train, n_steps=int(args.steps * 1.5))[0]
    print(f"  {'constant':>14} | {evaluate(constant, env, test).value:>9.4f} | "
          f"{evaluate(constant, env, train).value:>10.4f} | the floor")

    full = train_pathwise(env, train, n_steps=args.steps).policy
    print(f"  {'full BPTT':>14} | {evaluate(full, env, test).value:>9.4f} | "
          f"{evaluate(full, env, train).value:>10.4f} | K = horizon, no critic")

    for window in args.windows:
        result = train_svg(env, train, window=window, n_steps=args.steps)
        note = f"critic loss {result.critic_loss[0]:.2f} -> {result.critic_loss[-1]:.3f}"
        print(f"  {'SVG(K=%d)' % window:>14} | {evaluate(result.policy, env, test).value:>9.4f} | "
              f"{result.evaluation.value:>10.4f} | {note}")

    print(
        "\nA value near 0.7 x opening equity is the firm winding down in the first"
        "\nquarter, not a policy. Both learners fall into it past about twelve"
        "\nquarters -- see docs/quant-model.md section 6."
    )


if __name__ == "__main__":
    main()
