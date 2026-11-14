"""Solve the vault form and print what it says (design.md sections 2 and 7).

    uv run python scripts/solve_vault.py [--profile reference] [--periods-per-year 1]
                                         [--btc-view 0.0]

Prints whether the grid reached walk-away and payout, the firm's controls and
effective risk aversion across equity and AUM, the vault's allocation, the
target GRC stock, and what a GRC program is worth to the firm.

`--btc-view` is the firm's believed excess return on BTC a year. It defaults to
0: at the default 10% the firm treats BTC as a positive-NPV project, because
shareholders' valuation does not price BTC's risk, and the solve degenerates
into a BTC holding company (design.md section 11). Numbers printed here are
exploratory and are not quoted in docs/.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from quant.cli import parse_profile, print_header  # noqa: E402
from quant.params import DEFAULTS  # noqa: E402
from quant.vault_results import grc_value, risk_map, target_stock, thresholds  # noqa: E402
from quant.vault_solver import VaultSpec, solve_vault  # noqa: E402

SPEC = VaultSpec()
SHOW_W = (-0.2, -0.1, 0.0, 0.1, 0.3)
SHOW_A = (0.45, 0.8, 1.15)


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--periods-per-year", type=int, default=1)
    pre.add_argument("--btc-view", type=float, default=0.0)
    known, rest = pre.parse_known_args(argv)
    profile = parse_profile(__doc__.splitlines()[0], rest)
    print_header(profile)
    params = replace(DEFAULTS, periods_per_year=known.periods_per_year, mu_BTC=known.btc_view)
    print(f"periods per year: {params.periods_per_year}; believed BTC excess return: {params.mu_BTC:.1%}")

    sol = solve_vault(params, profile, SPEC)
    print(f"converged in {sol.iterations} iterations, residual {sol.residual:.1e}")
    print(f"grid reaches walk-away: {sol.lower_edge_is_walk_away}; reaches payout: {sol.upper_edge_pays_out}")

    rm = risk_map(sol)
    c = sol.controls()
    w, s, A = sol.states()
    print("\nat zero GRC stock")
    print("      w      A       G       K       g       q       H   alloc (lend, neutral, treasuries)")
    for a in SHOW_A:
        for x in SHOW_W:
            r = int(torch.argmin((w - x).abs() + 1e3 * s + (A - a).abs()))
            al = ", ".join(f"{v:.2f}" for v in c.alloc[r].tolist())
            print(
                f"  {float(w[r]):+.2f}  {float(A[r]):.2f}  {float(rm.G[r]):6.2f}  {float(c.K[r]):.4f}"
                f"  {float(c.g[r]):.4f}  {float(c.q[r]):.4f}  {float(c.H[r]):.4f}   ({al})"
            )
    th = thresholds(sol)
    print("\nat zero GRC stock, by AUM")
    for k, a in enumerate(sol.grid.axes[2].tolist()):
        print(f"  A={a:.2f}  walk away below w={float(th.walk_away[0, k]):+.3f}  pay out above w={float(th.payout[0, k]):+.3f}")
    print(f"\ntarget GRC stock at capacity: {target_stock(sol):.4f}")

    gv = grc_value(params, profile, SPEC)
    print(f"GRC program at w = 0.1, A = capacity: value to the firm {gv.gain:.5f}, target stock {gv.stock:.4f}")


if __name__ == "__main__":
    main()
