"""Named stress scenarios: one thing that could happen in the next period,
worked through the accounting (docs/plan-tool.md step 4; design.md sections
2.2-2.5 and 4.3).

A `Stress` says what happens: where BTC's price goes by a smooth move, whether
a crash then takes a further share at once, and whether there is an incident
in the lending system or in the BTC vault. `run` gives what the firm, its two
vaults and their investors are left with, in dollars, under the user's own
policy. Nothing is averaged and no chances are attached: a stress answers
"what if this happens", not "how likely is it".

The difference between a smooth fall and a crash is the model's (design.md
section 2.2): a smooth fall is caught at the liquidation threshold and leaves
no bad debt; a crash jumps past it, and the collateral sells at the price
after it. Everyday operating losses are at their average.

The model allows at most one incident a period (design.md section 4.3), so a
stress may have one in the lending system or one in the BTC vault, not both.
"""

import math
from dataclasses import dataclass, replace

import torch

from quant.core import issuance_cost, next_grc
from quant.firm import btc_prices
from quant.integrated import integrated_capital_floor, integrated_outcomes
from quant.numerics import REFERENCE, NumericsProfile
from quant.policy import Policy, to_controls
from quant.profile import ModelFirm
from quant.quadrature import JumpRule, ShockRule


@dataclass(frozen=True, kw_only=True)
class Stress:
    """One period's events. Shares are of what is there at the start."""

    name: str
    btc_move: float = 0.0             # BTC's smooth move over the period: -0.3 is a 30% fall
    btc_crash: float = 0.0            # then a sudden crash taking this share of the price
    lending_incident: float = 0.0     # share of the stablecoins in the lending system lost
    btc_vault_incident: float = 0.0   # severity of an incident in the BTC vault's own strategies;
                                      # each loses this times its exposure

    @property
    def btc_price(self) -> float:
        """BTC's price at the end, starting from 1."""
        return (1.0 + self.btc_move) * (1.0 - self.btc_crash)


def standard_stresses() -> tuple[Stress, ...]:
    """A first set to look at. The sizes are round numbers, not estimates."""
    return (
        Stress(name="calm: BTC flat, no incident"),
        Stress(name="BTC falls 30%, smoothly", btc_move=-0.30),
        Stress(name="BTC falls 50%, smoothly", btc_move=-0.50),
        Stress(name="BTC crashes 30%", btc_crash=0.30),
        Stress(name="BTC crashes 50%", btc_crash=0.50),
        Stress(name="BTC crashes 70%", btc_crash=0.70),
        Stress(name="incident: 10% of the lending system", lending_incident=0.10),
        Stress(name="incident: 10% in the BTC vault", btc_vault_incident=0.10),
        Stress(name="BTC crashes 50% and a lending incident of 10%", btc_crash=0.50, lending_incident=0.10),
        Stress(name="BTC rises 50%", btc_move=0.50),
    )


@dataclass(frozen=True)
class StressResult:
    """What one stress leaves, at the end of the period. Dollars."""

    stress: Stress
    btc_price: float                 # at the end, starting from 1
    # -- the firm -------------------------------------------------------------------
    equity_kept: float               # at the start, after any payout or raise
    equity: float                    # at the end
    capital_needed: float            # this period's, which the firm must get back to
    shortfall: float                 # equity below it, to raise or to make up
    cost_of_raising: float           # the issuance cost of raising that shortfall at once
    interest_income: float           # the period's
    btc_vault_fees: float            # the period's, at the closing BTC price
    repaid_to_investors: float       # compensation for incident losses and bad debt
    # -- the loan book and the vaults --------------------------------------------------
    bad_debt: float
    dollar_vault: float              # the vaults' sizes at the end, withdrawals included
    btc_vault: float
    # -- outside investors, over the period ------------------------------------------------
    dollar_return: float             # what a dollar provider receives, compensation included
    btc_return: float                # what a BTC investor receives, in BTC
    dollar_investors_gain: float     # in dollars; a loss is negative
    btc_investors_gain: float        # in dollars, BTC's price move included

    @property
    def equity_change(self) -> float:
        return self.equity - self.equity_kept


def problems(stress: Stress, model: ModelFirm) -> list[str]:
    """What is wrong with a stress, in plain words; empty if it can be run."""
    found = []
    if not stress.btc_move > -1.0:
        found.append("BTC's smooth move must leave some of the price: a fall of less than 100%")
    for name, value in (("the crash", stress.btc_crash), ("the lending incident", stress.lending_incident),
                        ("the BTC vault incident", stress.btc_vault_incident)):
        if not 0.0 <= value < 1.0:
            found.append(f"{name} must take a share between 0 and 1, below all of it")
    if stress.lending_incident > 0 and stress.btc_vault_incident > 0:
        found.append("the model allows one incident a period: in the lending system or in the BTC vault, not both")
    if stress.btc_move != 0.0 and model.params.s_BTC == 0.0:
        found.append("BTC's volatility is zero in this profile, so its price cannot move smoothly")
    return found


def run(
    model: ModelFirm, policy: Policy, stress: Stress, profile: NumericsProfile = REFERENCE
) -> StressResult:
    """One stress under `policy`. An unsound policy or stress is refused with
    the reasons."""
    found = problems(stress, model)
    if found:
        raise ValueError(f"the stress '{stress.name}' cannot be run: " + "; ".join(found))
    scale = model.scale
    p = model.params.per_period()
    c = to_controls(policy, model, profile)

    def one(x: float) -> torch.Tensor:
        return profile.tensor([x])

    # The shock to BTC's smooth move that takes the price to 1 + btc_move (quant.firm.btc_prices).
    if p.s_BTC > 0.0:
        _, at_zero = btc_prices(p, one(0.0), one(0.0), p.btc_bias)
        eps = (math.log(1.0 + stress.btc_move) - math.log(float(at_zero))) / p.s_BTC
    else:
        eps = 0.0
    rule = ShockRule(nodes=profile.tensor([[0.0, 0.0, 0.0, eps, stress.btc_crash]]), weights=profile.ones(1))

    # Outcomes come as: no incident, an incident in the lending system, one in the BTC vault.
    severity = max(stress.lending_incident, stress.btc_vault_incident)
    if severity > 0.0:
        jumps = JumpRule(severity=one(severity), weights=profile.ones(1))
        branch = 1 if stress.lending_incident > 0.0 else 2
        p_run = p if p.jump_rate > 0.0 else _with_incidents(p)
    else:
        jumps, branch, p_run = None, 0, p

    A_L, A_B = one(model.state.A_L), one(model.state.A_B)
    with torch.no_grad():
        s_next = next_grc(p, one(model.state.S), c.g)
        out = integrated_outcomes(p_run, A_L, A_B, c, s_next, rule, jumps)
        needed = float(integrated_capital_floor(p, A_L, A_B, c.g, c.q))

        def at(x: torch.Tensor) -> float:
            return float(x.reshape(-1)[branch])

        price = stress.btc_price
        equity = at(out.wealth)
        short = max(needed - equity, 0.0)
        outside_l = (1.0 - p.lend_co_invest) * model.state.A_L
        outside_b = (1.0 - p.btc_co_invest) * model.state.A_B
        return StressResult(
            stress=stress, btc_price=price,
            equity_kept=policy.equity_kept, equity=equity * scale, capital_needed=needed * scale,
            shortfall=short * scale, cost_of_raising=float(issuance_cost(p, one(short))) * scale,
            interest_income=at(out.interest) * scale, btc_vault_fees=at(out.fees_b) * scale,
            repaid_to_investors=at(out.cover) * scale,
            bad_debt=at(out.bad_debt) * scale,
            dollar_vault=at(out.aum_l) * scale, btc_vault=at(out.aum_b) * scale,
            dollar_return=at(out.received_l), btc_return=at(out.received_b),
            dollar_investors_gain=outside_l * at(out.received_l) * scale,
            btc_investors_gain=outside_b * ((1.0 + at(out.received_b)) * price - 1.0) * scale,
        )


def _with_incidents(p):
    """The accounting lays out incident outcomes only when incidents can
    happen. A stress imposes one whatever its chance, so a profile with none
    is given a nominal rate; a single outcome does not depend on it."""
    return replace(p, jump_rate=1e-12)


def run_all(
    model: ModelFirm, policy: Policy, stresses: tuple[Stress, ...] | None = None,
    profile: NumericsProfile = REFERENCE,
) -> list[StressResult]:
    return [run(model, policy, s, profile) for s in (standard_stresses() if stresses is None else stresses)]


def report(results: list[StressResult]) -> str:
    """The stresses side by side, in millions of dollars."""
    def m(x: float) -> str:
        return f"{x / 1e6:8.2f}"

    def pct(x: float) -> str:
        return f"{100 * x:7.1f}%"

    first = results[0]
    width = max(len(r.stress.name) for r in results)
    head = (f"{'':<{width}}  {'BTC':>6} | {'equity':>8} {'change':>8} {'short':>8} {'to raise':>8} | "
            f"{'bad debt':>8} {'repaid':>8} | {'$ vault':>8} {'B vault':>8} | {'$ inv.':>8} {'BTC inv.':>8}")
    lines = [
        f"One period under this policy. Equity kept {first.equity_kept / 1e6:.2f}, "
        f"capital needed {first.capital_needed / 1e6:.2f}. Dollars in millions.",
        "'short' is equity below the capital needed; 'to raise' is the cost of raising it at once.",
        "Investors' columns are what outside investors receive over the period: dollar providers, and BTC investors in BTC.",
        "",
        head,
        "-" * len(head),
    ]
    for r in results:
        lines.append(
            f"{r.stress.name:<{width}}  {r.btc_price:6.2f} | {m(r.equity)} {m(r.equity_change)} {m(r.shortfall)} "
            f"{m(r.cost_of_raising)} | {m(r.bad_debt)} {m(r.repaid_to_investors)} | {m(r.dollar_vault)} "
            f"{m(r.btc_vault)} | {pct(r.dollar_return)} {pct(r.btc_return)}"
        )
    return "\n".join(lines)
