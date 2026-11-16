"""Print what one period holds for a firm under a policy, what a set of
stress scenarios would leave, and where several years may take it; or two
policies side by side; or how much any of it leans on what cannot be observed
(docs/plan-tool.md steps 3 to 7).

    uv run python -m scripts.whatif
    uv run python -m scripts.whatif --periods 1 --rate 0.08 --ltv 0.6 --sleeve 0.2
    uv run python -m scripts.whatif --ltv 0.7 --stress
    uv run python -m scripts.whatif --years 5 --buffer 8e5
    uv run python -m scripts.whatif --ltv 0.5 --b-ltv 0.7 --b-sleeve 0.2 --years 5
    uv run python -m scripts.whatif --sensitivity
    uv run python -m scripts.whatif --b-grc-spend 1.5e5 --years 5 --sensitivity

For now the firm is the made-up example of `quant.profile.example_facts`,
with illustrative assumptions, and the policy is `quant.policy.starting_policy`
with the options below changed. A firm and a policy of the user's own come
with the tool's input format (step 9).
"""

from dataclasses import replace

from quant.cli import print_header, profile_parser
from quant.compare import compare
from quant.compare import report as compare_report
from quant.numerics import get_profile
from quant.paths import report as path_report
from quant.paths import rule_from, simulate
from quant.policy import problems, starting_policy
from quant.profile import example_facts, illustrative, to_model
from quant.readout import read_out, report
from quant.sensitivity import report as sensitivity_report
from quant.sensitivity import sensitivity
from quant.stress import report as stress_report
from quant.stress import run_all


def main(argv: list[str] | None = None) -> None:
    parser = profile_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--periods", type=int, default=4, help="decisions a year (default: %(default)s)")
    parser.add_argument("--equity-kept", type=float, help="dollars")
    parser.add_argument("--grc-spend", type=float, help="dollars a year")
    parser.add_argument("--own-btc", type=float, help="dollars")
    parser.add_argument("--hedged", type=float, help="dollars")
    parser.add_argument("--rate", type=float, help="borrower rate a year, e.g. 0.08")
    parser.add_argument("--ltv", type=float, help="loan-to-value, e.g. 0.6")
    parser.add_argument("--sleeve", type=float, help="share of the BTC vault lent as stablecoins; "
                                                     "the rest is split evenly over its three strategies")
    parser.add_argument("--stress", action="store_true", help="also work a set of stress scenarios through")
    parser.add_argument("--years", type=float, help="also carry the firm forward this many years on random paths")
    parser.add_argument("--paths", type=int, default=20000, help="how many paths (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer", type=float, help="equity held above the capital needed, dollars "
                                                     "(default: what the policy keeps today)")
    parser.add_argument("--largest-raise", type=float, help="the most shareholders will put in at once, dollars "
                                                            "(default: the capital needed today)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="show how much each figure leans on each assumption that cannot be observed "
                             "(with a second policy: how much the change between the two does)")
    second = parser.add_argument_group(
        "a second policy to compare with", "the first policy with these changed; giving any prints the two side by side")
    second.add_argument("--b-equity-kept", type=float)
    second.add_argument("--b-grc-spend", type=float)
    second.add_argument("--b-own-btc", type=float)
    second.add_argument("--b-hedged", type=float)
    second.add_argument("--b-rate", type=float)
    second.add_argument("--b-ltv", type=float)
    second.add_argument("--b-sleeve", type=float)
    args = parser.parse_args(argv)
    profile = get_profile(args.profile)
    print_header(profile)

    firm = illustrative(replace(example_facts(), decisions_per_year=args.periods))
    model = to_model(firm)
    policy = starting_policy(model)
    changes = {
        "equity_kept": args.equity_kept, "grc_spend": args.grc_spend, "own_btc": args.own_btc,
        "hedged": args.hedged, "borrower_rate": args.rate, "ltv": args.ltv,
    }
    policy = replace(policy, **{k: v for k, v in changes.items() if v is not None})
    if args.sleeve is not None:
        rest = (1.0 - args.sleeve) / 3.0
        policy = replace(policy, allocation=(rest, rest, rest, args.sleeve))
    found = problems(policy, model)
    if found:
        print("This policy cannot be run:")
        for line in found:
            print(f"  - {line}")
        return
    other = {
        "equity_kept": args.b_equity_kept, "grc_spend": args.b_grc_spend, "own_btc": args.b_own_btc,
        "hedged": args.b_hedged, "borrower_rate": args.b_rate, "ltv": args.b_ltv,
    }
    if args.b_sleeve is not None or any(v is not None for v in other.values()):
        second_policy = replace(policy, **{k: v for k, v in other.items() if v is not None})
        if args.b_sleeve is not None:
            rest = (1.0 - args.b_sleeve) / 3.0
            second_policy = replace(second_policy, allocation=(rest, rest, rest, args.b_sleeve))
        found = problems(second_policy, model)
        if found:
            print("The second policy cannot be run:")
            for line in found:
                print(f"  - {line}")
            return
        print("An example firm with illustrative assumptions: read shapes, not levels.\n")
        if args.sensitivity:
            print(sensitivity_report(sensitivity(
                firm, policy, versus=second_policy, years=args.years, paths=args.paths, seed=args.seed,
                largest_raise=args.largest_raise, numerics=profile,
            )))
            return
        print(compare_report(compare(
            model, policy, second_policy, names=("first", "second"), years=args.years, paths=args.paths,
            seed=args.seed, largest_raise=args.largest_raise, profile=profile,
        )))
        return
    print("An example firm with illustrative assumptions: read shapes, not levels.\n")
    if args.sensitivity:
        print(sensitivity_report(sensitivity(
            firm, policy, years=args.years, paths=args.paths, seed=args.seed,
            largest_raise=args.largest_raise, numerics=profile,
        )))
        return
    print(report(read_out(model, policy, profile=profile)))
    if args.stress:
        print("\n" + stress_report(run_all(model, policy, profile=profile)))
    if args.years is not None:
        rule = rule_from(policy, model, largest_raise=args.largest_raise)
        if args.buffer is not None:
            rule = replace(rule, equity_buffer=args.buffer)
        print("\n" + path_report(simulate(model, rule, years=args.years, paths=args.paths, seed=args.seed, profile=profile)))


if __name__ == "__main__":
    main()
