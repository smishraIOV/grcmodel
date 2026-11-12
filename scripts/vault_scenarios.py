"""Compare ways of putting the firm's own capital behind the vault (design.md section 4).

    uv run python scripts/vault_scenarios.py [--profile reference] [--periods-per-year 1]

Four scenarios: neither, co-investment only, a guarantee only, both. For each,
prints convergence, where the firm walks away and pays out at full capacity, its
operating capital, effective risk aversion and allocation, and what a GRC
program is worth to it. No BTC view (design.md section 8). Exploratory: numbers
printed here are not quoted in docs/.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from quant.cli import parse_profile, print_header  # noqa: E402
from quant.params import DEFAULTS  # noqa: E402
from quant.vault_results import grc_value, risk_map, thresholds  # noqa: E402
from quant.vault_solver import VaultSpec, solve_vault  # noqa: E402

SPEC = VaultSpec()
CO_INVEST = 0.10      # the firm keeps 10% of AUM of its own money in the vault
FLOOR = 0.0           # depositors' principal is protected each period

SCENARIOS = {
    "neither": dict(),
    "co-investment": dict(co_invest=CO_INVEST),
    "guarantee": dict(guarantee=FLOOR),
    "both": dict(co_invest=CO_INVEST, guarantee=FLOOR),
}


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--periods-per-year", type=int, default=1)
    known, rest = pre.parse_known_args(argv)
    profile = parse_profile(__doc__.splitlines()[0], rest)
    print_header(profile)
    base = replace(DEFAULTS, periods_per_year=known.periods_per_year, mu_BTC=0.0)
    print(f"co-investment {CO_INVEST:.0%} of AUM; guarantee floor {FLOOR:.0%} a year\n")

    for name, change in SCENARIOS.items():
        p = replace(base, **change)
        sol = solve_vault(p, profile, SPEC)
        w, s, A = sol.states()
        k = int(torch.argmin((sol.grid.axes[2] - p.capacity).abs()))
        th = thresholds(sol)
        rm = risk_map(sol)
        c = sol.controls()
        r = int(torch.argmin((w - 0.1).abs() + 1e3 * s + (A - sol.grid.axes[2][k]).abs()))
        alloc = ", ".join(f"{v:.2f}" for v in c.alloc[r].tolist())
        gv = grc_value(p, profile, SPEC)
        print(
            f"{name:14s} residual {sol.residual:.0e} | at capacity: walk away below "
            f"{float(th.walk_away[0, k]):+.3f}, pay out above {float(th.payout[0, k]):+.3f}, "
            f"capital {float(c.K[r]):.4f}, G {float(rm.G[r]):.2f}, alloc ({alloc}) | "
            f"GRC value {gv.gain:+.5f}, target stock {gv.stock:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
