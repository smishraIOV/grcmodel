#!/usr/bin/env python3
"""Multi-period model: what changes when GRC persists (docs/quant-model.md 5.2).

The static model charged GRC as a per-period expense that bought exactly one
period of protection. Over a horizon that is the wrong shape: a control
installed this quarter still works next quarter. Making GRC a depreciating
capital stock is the smallest change that turns the repeated static problem
into an intertemporal one, and it is what gives pre-emptive spend option value.

Three solvers, all scored by the same evaluate():

  constant    the best state-independent action. The floor a learner must beat.
  neural      state feedback, trained by backpropagation through the rollout.
  PI bound    every control chosen per path with the future known. Not
              implementable -- it is an upper bound on what any policy can do.

The sandwich V(constant) <= V(neural) <= V_PI is the cheapest correctness
check available: a learner above the bound has an information leak, a
mis-signed discount or a death that failed to absorb, none of which shows up
in the value on its own.

Usage: uv run python scripts/run_dynamic_model.py [--frequency monthly] [--paths 2048]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace

from quant.cli import print_header
from quant.env.env import EnvConfig, FirmEnv, evaluate
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.numerics import DEFAULT_PROFILE, PROFILES, get_profile
from quant.params import DEFAULTS, MONTHLY, steady_state_firm
from quant.solvers.neural import train_pathwise
from quant.solvers.pathwise import optimize_constant, perfect_information_bound

# The decision frequencies with a solved GRC steady state. Monthly is the
# main line; quarterly is kept so the published two-year numbers stay
# reproducible.
FREQUENCIES = {"quarterly": DEFAULTS, "monthly": MONTHLY}


def annual_death(survival_rate: float, periods: int, per_year: int = 4) -> float:
    """Survival over the horizon, re-expressed as a probability per year.

    `per_year` is a parameter rather than a literal 4 because it was a literal
    4, and at any other decision frequency that silently reports the wrong
    number everywhere it is printed -- the horizon is in periods, so treating
    60 monthly periods as 60 quarters annualizes over fifteen years instead of
    five. `EvalResult` computes its own version correctly from
    `firm.periods_per_year` (quant/env/env.py); this is the script-local one.
    """
    return 1.0 - survival_rate ** (per_year / periods)


def survival_channel(env, naive_env, crn, steps: int, periods: int, params) -> None:
    """What it costs to budget as though GRC only reduced expected loss.

    Both policies are scored in the same world -- the one where GRC also
    reduces failure intensity. Comparing them in their own worlds would be
    meaningless, since the loss-only world is simply less dangerous.
    """
    naive = optimize_constant(naive_env, crn, n_steps=steps)[0]
    aware, aware_result = optimize_constant(env, crn, n_steps=steps)
    naive_result = evaluate(naive, env, crn)

    print("\nWhere GRC acts: budgeting for losses only, vs for losses and survival")
    header = f"  {'budget set for':>16} | {'spend/prd':>10} | {'annual death':>13} | {'value':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, r in (("expected loss", naive_result), ("loss + survival", aware_result)):
        print(
            f"  {label:>16} | {r.total_grc:>10.4f} | {annual_death(r.survival_rate, periods, params.firm.periods_per_year):>12.2%} | "
            f"{r.value:>8.4f}"
        )
    print(
        f"  Ignoring the survival channel costs {1 - naive_result.value / aware_result.value:.1%} of firm "
        f"value and raises\n  the annual failure probability by "
        f"{annual_death(naive_result.survival_rate, periods, params.firm.periods_per_year) - annual_death(aware_result.survival_rate, periods, params.firm.periods_per_year):.2%}."
    )


def liquidity_defence(sampler, crn, steps: int, periods: int, params) -> None:
    """Controls or cash? What the firm buys when a run is a mechanism.

    The run replaced a hazard rate that asserted an incident kills the firm.
    Once depositors actually leave and the firm can hold reserves against them,
    it has two defences rather than one, and the comparison below is what it
    chooses. Sweeping the run rate is the test: if controls were the answer, a
    likelier run would buy more of them.
    """
    from quant.env.env import EnvConfig, FirmEnv

    firm = steady_state_firm(params, periods)
    print("\nDefending against a run: controls, or cash?")
    header = (
        f"  {'runs/year':>10} | {'GRC/prd':>8} {'operational':>12} | {'reserves':>9} "
        f"{'book':>7} | {'p(run)/prd':>11} {'annual death':>13} | {'value':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = []
    for rate in (0.0, 0.45, 1.0, 2.0):
        funding = replace(params.funding, annual_run_rate=rate)
        env = FirmEnv(
            EnvConfig.at_frequency(params, periods, firm=firm, funding=funding), sampler
        )
        policy, r = optimize_constant(env, crn, n_steps=steps)
        trajectory = env.rollout(policy, crn, differentiable=False)
        runs = sum(
            float(info["run_probability"].mean()) for info in trajectory.infos
        ) / len(trajectory.infos)
        rows.append((rate, r))
        print(
            f"  {rate:>10.2f} | {r.total_grc:>8.4f} {r.grc[1]:>12.4f} | "
            f"{r.reserve_ratio:>8.1%} {r.book:>7.2f} | {runs:>10.3%} "
            f"{annual_death(r.survival_rate, periods, params.firm.periods_per_year):>12.2%} | {r.value:>8.3f}"
        )

    # Read off the measured rows rather than asserted. An earlier version of
    # this line said flatly that a likelier run buys cash and not controls,
    # which was true of the first calibration it was written against and stopped
    # being true at the next one. The repo has made this mistake before, in this
    # same script, about which way per-quarter spend moves.
    quiet, first = rows[0][1], rows[1][1]
    riskiest = rows[-1][1]
    print(
        f"  Introducing a run at all takes operational GRC from {quiet.grc[1]:.4f} to "
        f"{first.grc[1]:.4f}\n  and reserves from {quiet.reserve_ratio:.0%} to "
        f"{first.reserve_ratio:.0%}: the firm buys both defences.\n"
        f"  Past that it substitutes -- at {rows[-1][0]:.1f} runs a year reserves reach "
        f"{riskiest.reserve_ratio:.0%} while\n  operational GRC falls back to "
        f"{riskiest.grc[1]:.4f}. Cash is the certain defence and controls\n"
        f"  are the probabilistic one, so the cheaper certainty wins at the margin."
    )


def recovery_sweep(sampler, crn, steps: int, periods: int, params) -> None:
    """What the firm does as failure becomes less destructive.

    The cleanest comparative static the survival channel produces, and the one
    an executive can argue with: the less a failure destroys, the less a
    programme to avoid it is worth. GRC spend here is bought entirely by the
    franchise at risk, since recovery is the part of firm value that survives
    failure.
    """
    print("\nHow much of the firm survives failure, and what that does to the budget")
    header = f"  {'recovery':>9} | {'spend/prd':>10} | {'annual death':>13} | {'value':>8} | {'going-concern':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for recovery in (0.0, 0.4, 0.8):
        firm = replace(steady_state_firm(params, periods), failure_recovery=recovery)
        env = FirmEnv(EnvConfig.at_frequency(params, periods, firm=firm), sampler)
        r = optimize_constant(env, crn, n_steps=steps)[1]
        print(
            f"  {recovery:>9.1f} | {r.total_grc:>10.4f} | {annual_death(r.survival_rate, periods, params.firm.periods_per_year):>12.2%} | "
            f"{r.value:>8.3f} | {r.going_concern_share:>14.3f}"
        )


def balance_sheet(sampler, crn, steps: int, periods: int, params) -> None:
    """What distributing earnings does to a firm whose funding is scarce.

    Retaining everything, capital compounds until the funding constraint stops
    mattering and the balance sheet is decorative. Distributing holds it flat,
    which keeps funding scarce, keeps the underinvestment channel alive, and
    is the first thing in this model to make the discount rate matter -- until
    there were dividends, every reward was zero and beta was a scalar
    multiplier on a terminal value.
    """
    firm = steady_state_firm(params, periods)
    print("\nRetaining earnings versus distributing them")
    header = (f"  {'payout':>10} | {'value':>8} | {'spend/prd':>10} | {'underinvest':>12} | "
              f"{'annual death':>13} | {'end equity':>11} | {'div share':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, allow in (("retain all", False), ("optimized", True)):
        env = FirmEnv(
            EnvConfig.at_frequency(params, periods, firm=firm, allow_payout=allow), sampler
        )
        policy, r = optimize_constant(env, crn, n_steps=steps)
        final = env.rollout(policy, crn, False).states[-1].equity.median().item()
        print(
            f"  {label:>10} | {r.value:>8.3f} | {r.total_grc:>10.4f} | {r.underinvestment_fraction:>12.3f} | "
            f"{annual_death(r.survival_rate, periods, params.firm.periods_per_year):>12.2%} | {final:>11.2f} | {r.payout_share:>10.3f}"
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
    parser.add_argument("--paths", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--recovery-sweep", action="store_true",
        help="also sweep how much of the firm survives failure (slower)",
    )
    args = parser.parse_args()

    profile = get_profile(args.profile)
    print_header(profile)

    params = FREQUENCIES[args.frequency]
    periods = args.periods or 5 * params.firm.periods_per_year
    # A fixed point of the configuration, not of the firm -- looked up, and an
    # unsolved combination raises rather than inheriting a stale default.
    firm = steady_state_firm(params, periods)
    sampler = MonteCarloSampler(params.sampler)
    crn = CommonRandomNumbers(0, periods, args.paths, profile)

    per_year = params.firm.periods_per_year
    print(
        f"{periods} {args.frequency} periods ({periods / per_year:.0f} years), "
        f"{args.paths} paths, discount {firm.discount():.4f}/period, "
        f"opening GRC stock {firm.initial_grc_stock}, insolvency barrier at 0"
    )
    print(
        "GRC decay 1.000 charges spend as a one-period expense (the static model's\n"
        f"assumption); {firm.grc_depreciation():.3f} is the per-period rate implied by "
        f"{firm.annual_grc_depreciation:.0%} a year.\n"
    )

    header = f"{'GRC decay':>11} | {'solver':>9} | {'value':>8} | {'spend/prd':>10} | {'end stock':>10} | {'survives':>9}"
    print(header)
    print("-" * len(header))

    summary = {}
    for delta in (1.0, firm.grc_depreciation()):
        env = FirmEnv(
            EnvConfig.at_frequency(params, periods, firm=firm, grc_depreciation=delta), sampler
        )
        results = {
            "constant": optimize_constant(env, crn, n_steps=args.steps)[1],
            "neural": train_pathwise(env, crn, n_steps=max(1, args.steps // 2)).evaluation,
            "PI bound": perfect_information_bound(env, crn, n_steps=args.steps)[1],
        }
        for name, r in results.items():
            print(
                f"{delta:>11.3f} | {name:>9} | {r.value:>8.4f} | {r.total_grc:>10.4f} | "
                f"{sum(r.grc_stock):>10.4f} | {r.survival_rate:>9.3f}"
            )
        bound = results["PI bound"].value
        for name in ("constant", "neural"):
            assert results[name].value <= bound + 1e-6, f"{name} broke the bound"

        # Flag a solver that has fallen into the absorbing wind-down rather
        # than letting it be averaged into a headline. The exit action is
        # absorbing, so once the probability mass has left there is nothing
        # still operating to generate a gradient for staying, and the solve
        # lands on exactly `orderly_recovery * initial_equity` with survival at
        # zero. That is an optimizer trap and not a valuation
        # (docs/static-model-debug-notes.md section 8).
        collapsed = firm.orderly_recovery * firm.initial_equity
        for name, r in results.items():
            if abs(r.value - collapsed) < 1e-2 and r.survival_rate < 1e-3:
                print(
                    f"{'':>11} | WARNING: {name} collapsed into the wind-down "
                    f"({collapsed:.3f}); its row is an optimizer trap, not a value"
                )

        # The best *implementable* policy, not a fixed one of the two. The
        # persistence headline below used to take the neural row unconditionally
        # and printed "+203%" off a collapsed solve. The sandwich only claims
        # V(constant) <= V*, so whichever of the two is higher is the better
        # estimate -- and taking the maximum is also what stops one solver
        # failing from rewriting a result about the model.
        best = max(("constant", "neural"), key=lambda n: results[n].value)
        print(f"{'':>11} | EVPI = {bound - results[best].value:.4f} (over {best})")
        print("-" * len(header))
        summary[delta] = results[best]

    env = FirmEnv(EnvConfig.at_frequency(params, periods, firm=firm), sampler)
    naive_env = FirmEnv(
        EnvConfig.at_frequency(
            params, periods, firm=firm,
            hazard=replace(params.hazard, annual_operational_rate=0.0, annual_licence_rate=0.0),
        ),
        sampler,
    )
    survival_channel(env, naive_env, crn, args.steps, periods, params)

    balance_sheet(sampler, crn, args.steps, periods, params)

    liquidity_defence(sampler, crn, args.steps, periods, params)

    if args.recovery_sweep:
        recovery_sweep(sampler, crn, args.steps, periods, params)

    flow, stock = summary[1.0], summary[firm.grc_depreciation()]
    print(
        f"\nPersistence is worth {stock.value / flow.value - 1:+.0%} of firm value "
        f"({flow.value:.2f} -> {stock.value:.2f}) and takes survival from "
        f"{flow.survival_rate:.0%} to {stock.survival_rate:.0%}."
    )
    # State the measured direction rather than assert one. This line used to
    # read "spends MORE per quarter" unconditionally; after credit loss was made
    # to scale with the book it printed that over a pair of numbers that had
    # gone the other way.
    rose = stock.total_grc > flow.total_grc
    print(
        f"Per-period spend {'rises' if rose else 'falls'} "
        f"({flow.total_grc:.3f} -> {stock.total_grc:.3f}). "
        + (
            "Each unit now protects every later quarter too, so more of it is worth buying."
            if rose
            else "A persisting stock buys the same protection for less flow."
        )
    )
    print(
        "\nParameters are illustrative. Read the direction and the mechanism, not the\n"
        "magnitudes (docs/quant-model.md section 8)."
    )


if __name__ == "__main__":
    main()
