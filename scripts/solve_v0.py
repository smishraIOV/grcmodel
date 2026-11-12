"""Solve the v0 bank and print what it says (design.md section 5, v0).

    uv run python scripts/solve_v0.py [--profile reference] [--periods-per-year 1]

Prints the walk-away and payout thresholds, the effective risk aversion and
the product's hurdle across equity, the target GRC stock, and what a GRC
program is worth to the bank.

The parameters are the solver tests' interior configuration (GRC worth buying,
the product not free money), not a calibration.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from quant.cli import parse_profile, print_header  # noqa: E402
from quant.params import DEFAULTS  # noqa: E402
from quant.results import grc_value, risk_map, target_stock, thresholds  # noqa: E402
from quant.solver import GridSpec, solve  # noqa: E402

PARAMS = replace(DEFAULTS, m=0.02, s_P=0.03, a_A=100.0, a_B=100.0, pi=0.002, jump_rate=0.0)
# Annual by default for quick exploration; pass --periods-per-year 4 for the
# quarterly model, which takes longer to converge.
DEFAULT_PERIODS = 1
SPEC = GridSpec(n_w=257, n_s=9, s_hi=0.08, w_lo=-0.4, w_hi=0.6, n_hermite=5, n_mixing=8)
SHOW = (-0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.03, 0.06, 0.1, 0.3)


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--periods-per-year", type=int, default=DEFAULT_PERIODS)
    known, rest = pre.parse_known_args(argv)
    profile = parse_profile(__doc__.splitlines()[0], rest)
    print_header(profile)
    params = replace(PARAMS, periods_per_year=known.periods_per_year)
    print(f"periods per year: {params.periods_per_year}")
    sol = solve(params, profile, SPEC)
    print(f"converged in {sol.iterations} iterations, residual {sol.residual:.1e}")
    print(f"grid reaches walk-away: {sol.lower_edge_is_walk_away}; reaches payout: {sol.upper_edge_pays_out}")

    th = thresholds(sol)
    print("\nthresholds by GRC stock")
    for s, wa, pay in zip(sol.grid.s.tolist(), th.walk_away.tolist(), th.payout.tolist()):
        print(f"  S={s:.3f}  walk-away below w={wa:+.3f}  pays out above w={pay:+.3f}")

    rm = risk_map(sol)
    c = sol.controls()
    w, s = sol.grid.states()
    print("\nat zero GRC stock")
    print("        w       G   hurdle_N   alpha       I       K")
    for x in SHOW:
        r = int(torch.argmin((w - x).abs() + 1e3 * s))
        print(
            f"  {float(w[r]):+.3f}  {float(rm.G[r]):6.3f}  {float(rm.hurdle_N[r]):+.5f}"
            f"  {float(c.alpha[r]):6.3f}  {float(c.I[r]):6.3f}  {float(c.K[r]):6.4f}"
        )
    print(f"\ntarget GRC stock: {target_stock(sol):.4f}")

    gv = grc_value(params, profile, SPEC)
    print("\nGRC program at w = 0.1, starting from none")
    print(f"  value to the bank  {gv.gain:.5f}")
    print(f"  target stock       {gv.stock:.4f}")

if __name__ == "__main__":
    main()
