"""Solve the v0 bank and print what it says (design.md section 6, v0).

    uv run python scripts/solve_v0.py [--profile reference]

Prints the walk-away and payout thresholds, the effective risk aversion and
the product's hurdle across equity, the target GRC stock, and GRC's value
split into expected-loss reduction and the Froot-Stein premium.

The parameters are the solver tests' interior configuration (GRC worth buying,
the product not free money), not a calibration.
"""

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
SPEC = GridSpec(n_w=257, n_s=9, s_hi=0.08, w_lo=-0.4, w_hi=0.6, n_hermite=5, n_mixing=8)
SHOW = (-0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.03, 0.06, 0.1, 0.3)


def main(argv=None):
    profile = parse_profile(__doc__.splitlines()[0], argv)
    print_header(profile)
    sol = solve(PARAMS, profile, SPEC)
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

    gv = grc_value(PARAMS, profile, SPEC)
    print("\nGRC value at w = 0.1, starting from no program")
    print(f"  to this bank                    {gv.gain:.5f}   target stock {gv.stock:.4f}")
    print(f"  to a bank that raises equity free {gv.gain_frictionless:.5f}   target stock {gv.stock_frictionless:.4f}")
    print(f"  Froot-Stein premium             {gv.premium:.5f}   ({gv.premium_share:.1%} of GRC's value)")

if __name__ == "__main__":
    main()
