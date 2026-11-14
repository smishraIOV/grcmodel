"""Solve the integrated firm on the coarse 4-D grid and print what it says
at one state (design.md sections 2-3; docs/plan-integrated.md steps 4-5).

    uv run python -m scripts.integrated --periods 1
    uv run python -m scripts.integrated --periods 1 --equity 0.3 --dollar-vault 1 --btc-vault 2

Annual (`--periods 1`) is for exploring; quarterly needs a finer equity grid
than the coarse one here (design.md section 4.4). The grid is coarse in every
direction: read shapes and signs, not levels.
"""

import time
from dataclasses import replace

from quant.cli import print_header, profile_parser
from quant.integrated_results import OPERATING_RANGE, PAYROLL_FLOOR, STRATEGIES, ReadOut, readout
from quant.integrated_solver import IntegratedSpec, solve_integrated
from quant.numerics import get_profile
from quant.params import INTEGRATED_DEFAULTS


def report(r: ReadOut) -> str:
    """The read-out as plain text. Money in the model's unit (the dollar
    vault's capacity = 1); rates and flows per year."""
    def pct(x):
        return f"{100 * x:6.2f}%"

    def money(x):
        return f"{x:8.4f}"

    lo, hi = OPERATING_RANGE
    lines = [
        f"State: equity {r.equity:.3f}, GRC stock {r.grc_stock:.3f}, "
        f"dollar vault {r.dollar_vault:.2f}, BTC vault {r.btc_vault:.2f}",
        f"Value to shareholders            {money(r.value)}",
        "",
        "Capital",
        f"  equity kept                    {money(r.equity_kept)}   (floor {r.capital_floor:.4f})",
        f"  paid out / raised now          {money(r.dividend)} / {r.issue:.4f}",
        f"  stakes in the two vaults       {money(r.stakes)}",
        f"  own BTC                        {money(r.own_btc)}   (hedged {r.hedged:.4f})",
        f"  stablecoins                    {money(r.stablecoins)}",
        "",
        "GRC",
        f"  spend a year                   {money(r.grc_spend)}",
        f"  chance of an incident a year   {pct(r.incident_chance)}",
        "",
        "The loan book",
        f"  borrower rate                  {pct(r.borrower_rate)}",
        f"  loan-to-value                  {pct(r.ltv)}",
        f"  loans                          {money(r.loans)}   ({100 * r.share_lent:.0f}% of the stablecoins supplied)",
        f"  expected bad debt a year       {money(r.expected_bad_debt)}",
        "",
        "The BTC vault's allocation",
        *(f"  {name:<30} {pct(share)}" for name, share in zip(STRATEGIES, r.allocation, strict=True)),
        "",
        "Expected over a year",
        f"  interest share                 {money(r.interest_income)}",
        f"  BTC vault fees                 {money(r.btc_vault_fees)}",
        f"  operating costs                {money(r.operating_costs)}   "
        f"({100 * r.operating_ratio:.0f}% of revenue; agreed range {100 * lo:.0f}-{100 * hi:.0f}%)",
        f"  payroll                        {money(r.payroll_share * r.operating_costs)}   "
        f"({100 * r.payroll_ratio:.0f}% of revenue; at least {100 * PAYROLL_FLOOR:.0f}%)",
        f"  repaid to investors            {money(r.compensation)}",
        "",
        "Investors, expected a year",
        f"  dollar vault                   {pct(r.dollar_yield)}   (elsewhere {100 * r.dollar_outside:.2f}%)",
        f"  BTC vault, in BTC              {pct(r.btc_yield)}   (elsewhere {100 * r.btc_outside:.2f}%)",
        "",
        "How the period may end",
        f"  equity below the capital floor {pct(r.chance_below_floor)}   (expected gap {r.expected_shortfall:.4f})",
        f"  shareholders walk away         {pct(r.chance_walk_away)}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = profile_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--periods", type=int, default=1, help="decisions a year (default: %(default)s)")
    parser.add_argument("--bias", type=float, default=0.0, help="the firm's BTC bias a year")
    parser.add_argument("--opex", type=float, default=None, help="operating costs a year")
    parser.add_argument("--equity", type=float, default=0.3)
    parser.add_argument("--dollar-vault", type=float, default=INTEGRATED_DEFAULTS.lend_capacity)
    parser.add_argument("--btc-vault", type=float, default=INTEGRATED_DEFAULTS.btc_capacity)
    parser.add_argument("--grc-stock", type=float, default=0.0)
    args = parser.parse_args(argv)
    profile = get_profile(args.profile)
    print_header(profile)

    p = replace(INTEGRATED_DEFAULTS, periods_per_year=args.periods, btc_bias=args.bias)
    if args.opex is not None:
        p = replace(p, opex=args.opex)
    start = time.time()
    sol = solve_integrated(p, profile, IntegratedSpec())
    print(f"solved in {time.time() - start:.0f}s: {sol.iterations} iterations, residual {sol.residual:.1e}\n")
    print(report(readout(sol, args.equity, args.dollar_vault, args.btc_vault, args.grc_stock)))


if __name__ == "__main__":
    main()
