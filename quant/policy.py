"""The firm's policy as an input (design.md section 3.2; docs/plan-tool.md
step 2).

A `Policy` is what the user's firm does this period, in the units a profile
uses: money in **dollars**, rates **per year**. It is given, not searched
for. `problems` says in plain words what is wrong with one; `to_controls`
turns a sound one into the decision the one-period accounting takes
(`quant.integrated.IntegratedControls`), and `to_policy` maps back.

The constraints are the model's own (design.md sections 2.6, 3.3, 3.4):

- **own funds**: equity kept is at least the required share of the two
  vaults' size;
- **runway**: after this period's GRC spend, the firm's own BTC and its
  stakes in the vaults, what is left in stablecoins covers the runway of
  operating costs;
- **hedge**: short perps cover at most the BTC the firm owns -- its own, and
  its stake in the BTC vault;
- the loan-to-value is inside the range the firm may offer, the allocation's
  shares are non-negative and sum to one, and nothing is negative.
"""

from dataclasses import dataclass

import torch

from quant.integrated import IntegratedControls
from quant.numerics import REFERENCE, NumericsProfile
from quant.profile import ModelFirm

ALLOCATION = ("lending BTC", "basis trading", "cold custody", "stablecoin lending sleeve")
_SLACK = 1e-9   # relative; rounding in what a user types is not a breach


@dataclass(frozen=True, kw_only=True)
class Policy:
    """One period's decisions. Dollars, and rates per year."""

    equity_kept: float                 # after this period's payout or raise
    grc_spend: float                   # a year
    own_btc: float                     # the firm's own BTC, valued in dollars
    hedged: float                      # of the firm's BTC, hedged with short perps
    borrower_rate: float               # a year
    ltv: float                         # loan-to-value at origination
    allocation: tuple[float, float, float, float]   # the BTC vault's shares, in the order of ALLOCATION


@dataclass(frozen=True)
class Position:
    """What a policy means for the firm's books this period, in dollars."""

    payout: float              # paid to shareholders now
    raised: float              # new equity raised now
    stakes: float              # the firm's own money in the two vaults
    stablecoins: float         # what is left of equity kept
    runway_needed: float       # the stablecoins the runway rule asks for
    own_funds_needed: float    # the equity the own-funds requirement asks for
    btc_held: float            # the BTC the firm owns and may hedge


def _usd(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def position(policy: Policy, model: ModelFirm) -> Position:
    p, s, scale = model.params, model.state, model.scale
    n = p.periods_per_year
    stakes = (p.lend_co_invest * s.A_L + p.btc_co_invest * s.A_B) * scale
    equity = s.w * scale
    return Position(
        payout=max(equity - policy.equity_kept, 0.0),
        raised=max(policy.equity_kept - equity, 0.0),
        stakes=stakes,
        stablecoins=policy.equity_kept - policy.grc_spend / n - policy.own_btc - stakes,
        runway_needed=p.runway * p.opex * scale,
        own_funds_needed=p.k_A * (s.A_L + s.A_B) * scale,
        btc_held=policy.own_btc + p.btc_co_invest * s.A_B * scale,
    )


def problems(policy: Policy, model: ModelFirm) -> list[str]:
    """Everything wrong with a policy for this firm, in plain words; empty if
    it is sound."""
    p = model.params
    n = p.periods_per_year
    pos = position(policy, model)
    found = []

    for name, value in (("GRC spend", policy.grc_spend), ("the firm's own BTC", policy.own_btc),
                        ("the hedge", policy.hedged), ("the borrower rate", policy.borrower_rate)):
        if value < 0:
            found.append(f"{name} cannot be negative")

    if len(policy.allocation) != len(ALLOCATION):
        found.append(f"the BTC vault's allocation needs {len(ALLOCATION)} shares: {', '.join(ALLOCATION)}")
    else:
        if min(policy.allocation) < 0:
            found.append("the BTC vault's allocation cannot have a negative share")
        if abs(sum(policy.allocation) - 1.0) > 1e-6:
            found.append(f"the BTC vault's allocation must sum to 100% (it sums to {100 * sum(policy.allocation):.1f}%)")

    if not p.ltv_min - _SLACK <= policy.ltv <= p.ltv_max + _SLACK:
        found.append(
            f"the loan-to-value of {100 * policy.ltv:.0f}% is outside the range the firm may offer, "
            f"{100 * p.ltv_min:.0f}% to {100 * p.ltv_max:.0f}%"
        )

    if policy.hedged > pos.btc_held * (1 + _SLACK):
        found.append(
            f"the hedge of {_usd(policy.hedged)} is more than the BTC the firm owns, {_usd(pos.btc_held)} "
            "(its own BTC plus its stake in the BTC vault)"
        )

    tolerance = _SLACK * max(model.scale, abs(policy.equity_kept))
    if policy.equity_kept < pos.own_funds_needed - tolerance:
        found.append(
            f"equity kept, {_usd(policy.equity_kept)}, is below the own funds required, "
            f"{_usd(pos.own_funds_needed)} ({100 * p.k_A:.1f}% of the two vaults)"
        )
    if pos.stablecoins < pos.runway_needed - tolerance:
        found.append(
            f"equity kept, {_usd(policy.equity_kept)}, leaves {_usd(pos.stablecoins)} in stablecoins after "
            f"GRC spend ({_usd(policy.grc_spend / n)} this period), the firm's own BTC ({_usd(policy.own_btc)}) "
            f"and its stakes in the vaults ({_usd(pos.stakes)}); the runway needs {_usd(pos.runway_needed)} "
            f"({12 * p.runway:.0f} months of operating costs)"
        )
    return found


def check(policy: Policy, model: ModelFirm) -> None:
    """Refuse an unsound policy, saying everything that is wrong with it."""
    found = problems(policy, model)
    if found:
        raise ValueError("this policy cannot be run: " + "; ".join(found))


def to_controls(
    policy: Policy, model: ModelFirm, profile: NumericsProfile = REFERENCE
) -> IntegratedControls:
    """The policy as the accounting's decision for **one period**, in model
    units: GRC spend is a period's share of the year's, and the borrower rate
    is the period's (compounding to the year's). Each tensor has shape (1,)."""
    check(policy, model)
    scale, n = model.scale, model.params.periods_per_year

    def one(x: float) -> torch.Tensor:
        return profile.tensor([x])

    return IntegratedControls(
        K=one(policy.equity_kept / scale),
        g=one(policy.grc_spend / scale / n),
        q=one(policy.own_btc / scale),
        H=one(policy.hedged / scale),
        rate=one((1.0 + policy.borrower_rate) ** (1.0 / n) - 1.0),
        ltv=one(policy.ltv),
        alloc=profile.tensor([list(policy.allocation)]),
    )


def to_policy(controls: IntegratedControls, model: ModelFirm, index: int = 0) -> Policy:
    """One decision of the accounting as a policy in dollars and yearly
    rates: `to_policy(to_controls(x, m), m)` is x."""
    scale, n = model.scale, model.params.periods_per_year

    def at(x: torch.Tensor) -> float:
        return float(x.reshape(-1)[index])

    return Policy(
        equity_kept=at(controls.K) * scale,
        grc_spend=at(controls.g) * scale * n,
        own_btc=at(controls.q) * scale,
        hedged=at(controls.H) * scale,
        borrower_rate=(1.0 + at(controls.rate)) ** n - 1.0,
        ltv=at(controls.ltv),
        allocation=tuple(float(x) for x in controls.alloc.reshape(-1, len(ALLOCATION))[index]),
    )


def starting_policy(model: ModelFirm) -> Policy:
    """A sound policy to start from, not a recommendation: keep today's
    equity, or the least the constraints allow if that is more; keep the GRC
    budget that holds today's stock; hold no BTC beyond the stake and hedge
    nothing; lend at half the rate at which nobody borrows and at the
    reference loan-to-value, kept inside the range; split the BTC vault
    evenly across its three own strategies, with no lending sleeve."""
    p, s, scale = model.params, model.state, model.scale
    n = p.periods_per_year
    budget = s.S * (1.0 - (1.0 - p.delta) ** (1.0 / n)) * n * scale
    empty = Policy(equity_kept=0.0, grc_spend=budget, own_btc=0.0, hedged=0.0, borrower_rate=p.rate_max / 2.0,
                   ltv=min(max(p.ltv_ref, p.ltv_min), p.ltv_max), allocation=(1 / 3, 1 / 3, 1 / 3, 0.0))
    pos = position(empty, model)
    least = max(pos.own_funds_needed, pos.runway_needed + budget / n + pos.stakes)
    return Policy(**{**empty.__dict__, "equity_kept": max(s.w * scale, least)})

