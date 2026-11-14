"""What each input means, for a person filling in a form (docs/plan-tool.md
step 10).

For every input a profile, a policy or the multi-year rule takes: a **label**
to show, **what it means** in a sentence or two, and **what to enter** -- how
to read the number, the limits the model puts on it, and a rough range to
try. The text speaks in the units a form shows: money in millions of
dollars, shares and yearly rates as percentages.

The ranges are a guide to sensible entries, **not calibration**: where they
describe the world ("BTC's volatility has usually been...") they are rough
and from general knowledge, and the user's own figures should replace them.

`resolve` gives the entry for an input; `guide` gives all of them, laid out
as `quant.api.inputs` lays out the inputs. A test fails if an input has no
entry, so a new input cannot be added without saying what it means.
"""

from dataclasses import fields

from quant.policy import Policy
from quant.profile import FirmFacts, GrcEffect, LoanPoint, MarketAssumptions, RaiseCost, RiskAssumptions

# input -> (label, what it means, what to enter)
FIRM = {
    "dollar_vault": (
        "Dollar vault: size now",
        "The stablecoins investors have in the dollar lending vault today. This is what can be lent to borrowers.",
        "In millions of dollars. Usually at or below the vault's capacity.",
    ),
    "dollar_vault_capacity": (
        "Dollar vault: capacity",
        "The size the dollar vault grows toward when things go well. The model measures everything else against it.",
        "In millions of dollars. Must be above zero.",
    ),
    "btc_vault": (
        "BTC vault: size now",
        "The BTC investors have in the BTC yield vault today, valued in dollars at today's price.",
        "In millions of dollars. Zero if the firm runs no BTC vault.",
    ),
    "btc_vault_capacity": (
        "BTC vault: capacity",
        "The dollar value the BTC vault grows toward when things go well.",
        "In millions of dollars.",
    ),
    "equity": (
        "The firm's equity now",
        "The firm's own money today: what its owners would have left if everything were settled. It absorbs the firm's "
        "losses before anything else does.",
        "In millions of dollars. It is the starting point for the payout or raise the policy implies.",
    ),
    "grc_budget": (
        "GRC budget so far, a year",
        "What the firm has been spending a year on governance, risk and compliance: controls, audits, security, "
        "monitoring. The model takes it as steady up to now, to work out how much protection the firm has already built.",
        "In millions of dollars a year. Zero means no protection built up yet. The policy's GRC spend sets what happens from now on.",
    ),
    "operating_costs": (
        "Operating costs a year",
        "Payroll, rent, vendors and everything else it costs to run the firm for a year, not counting GRC spend. "
        "They do not fall when revenue does.",
        "In millions of dollars a year. Firms like this usually spend 50% to 60% of revenue on them; the page shows the ratio.",
    ),
    "payroll_share": (
        "Payroll, share of operating costs",
        "How much of the operating costs is pay. Used only to report payroll against revenue.",
        "A percentage, 0 to 100. Often around 60%.",
    ),
    "runway_months": (
        "Runway kept in stablecoins",
        "How many months of operating costs the firm keeps in stablecoins at all times, so it can pay its bills "
        "without selling BTC after a fall. It is one of the two things that set the least equity the firm may keep.",
        "In months. 12 is a year of costs. Zero switches the rule off.",
    ),
    "interest_share": (
        "The firm's share of loan interest",
        "Of the interest borrowers pay, the part the firm keeps. The rest goes to the investors whose money was lent.",
        "A percentage, 0 to 100. The example uses 40%; lending platforms often keep 10% to 20%.",
    ),
    "management_fee": (
        "BTC vault: management fee",
        "The yearly fee on the BTC vault, as a share of the BTC in it, paid in BTC whatever the vault earns.",
        "A percentage a year. Funds usually charge 0.5% to 2%.",
    ),
    "performance_fee": (
        "BTC vault: performance fee",
        "The firm's share of the BTC vault's gains, in BTC, in periods when it gains. Nothing is charged on a loss.",
        "A percentage, 0 to under 100. Funds usually charge 10% to 20%.",
    ),
    "dollar_vault_stake": (
        "The firm's own stake in the dollar vault",
        "The part of the dollar vault that is the firm's own money, invested alongside its investors. It earns and "
        "loses what they do.",
        "A percentage of the vault, 0 to under 100. The example uses 5%. It ties up the firm's equity.",
    ),
    "btc_vault_stake": (
        "The firm's own stake in the BTC vault",
        "The part of the BTC vault that is the firm's own BTC, invested alongside its investors. It moves with BTC's "
        "price unless the firm hedges it.",
        "A percentage of the vault, 0 to under 100. The example uses 5%.",
    ),
    "compensation": (
        "Investor losses the firm repays",
        "When investors lose money to an incident or to bad debt, the share the firm pays back out of its own pocket, "
        "to protect its name.",
        "A percentage, 0 to 100. Zero means investors bear all of it; 100 means the firm makes them whole.",
    ),
    "own_funds": (
        "Own funds required per dollar of vaults",
        "A capital rule: the firm's equity must be at least this share of the two vaults' combined size, as some "
        "regulators require.",
        "A percentage. The example uses 2%. Zero if no such rule applies.",
    ),
    "liquidation_threshold": (
        "Liquidation threshold",
        "A loan is liquidated when what the borrower owes reaches this share of the value of their BTC collateral.",
        "A percentage, above the highest loan-to-value offered and at most 100. Crypto lenders usually use 75% to 85%.",
    ),
    "liquidation_penalty": (
        "Liquidation penalty",
        "The extra a liquidated borrower pays, as a share of what they owe. It goes to the vault's investors.",
        "A percentage, 0 to 100. Usually 5% to 10%.",
    ),
    "ltv_min": (
        "Lowest loan-to-value the firm may offer",
        "The lower limit on the policy's loan-to-value.",
        "A percentage, above zero and no more than the highest.",
    ),
    "ltv_max": (
        "Highest loan-to-value the firm may offer",
        "The upper limit on the policy's loan-to-value.",
        "A percentage, below the liquidation threshold.",
    ),
    "btc_security_cost": (
        "Cost of securing the firm's own BTC",
        "What it costs each year to keep the firm's own BTC safe: custody, key management, insurance, monitoring. "
        "Paid whether the BTC is hedged or not.",
        "A percentage of the BTC's value a year. The example uses 1.5%.",
    ),
    "required_return": (
        "Shareholders' required return",
        "The yearly return the firm's owners expect on money they leave in it. Future payouts are discounted at this "
        "rate, and in this model the market expects the same return from BTC.",
        "A percentage a year, at least the riskless rate. The example uses 6%; a riskier business would be asked for more.",
    ),
    "small_raise": (
        "Raising equity: a small raise",
        "What it costs to raise a modest amount of new equity in normal times, everything included: fees, the "
        "discount to new investors, management's time.",
        "Give an amount and its all-in cost. The example: raising $5m costs 20% of it.",
    ),
    "small_raise.amount": ("Amount raised", "The size of the small raise.", "In millions of dollars. Smaller than the large raise."),
    "small_raise.cost": (
        "All-in cost",
        "What raising that amount costs, as a share of the amount.",
        "A percentage. 20% means raising $5m costs $1m.",
    ),
    "large_raise": (
        "Raising equity: a large raise",
        "The same for a large raise. Large raises cost more per dollar; the model draws a straight line through the "
        "two answers.",
        "Give an amount and its all-in cost. The example: raising $20m costs 50% of it.",
    ),
    "large_raise.amount": ("Amount raised", "The size of the large raise.", "In millions of dollars. Larger than the small raise."),
    "large_raise.cost": (
        "All-in cost",
        "What raising that amount costs, as a share of the amount.",
        "A percentage, at least the small raise's.",
    ),
    "decisions_per_year": (
        "How often the firm decides",
        "How often the firm resets its policy: pays out or raises, re-prices loans, rebalances. One period on this "
        "page is one such interval.",
        "Every quarter is the usual choice. Once a year is coarser: a year's worth of risk falls between decisions.",
    ),
}

MARKET = {
    "riskless_rate": (
        "Riskless rate",
        "What safe dollars earn in a year, for example on short government bills. Perp funding is set above it.",
        "A percentage a year, no more than the shareholders' required return.",
    ),
    "perp_premium": (
        "Perp funding above the riskless rate",
        "Perpetual futures charge a funding rate: those betting on BTC rising pay it, those betting on a fall receive "
        "it. This is how far that rate sits above the riskless rate. The firm receives it on its hedge; the BTC vault "
        "pays it on the hedge of its lending sleeve.",
        "A percentage a year. The example uses 3%. Higher makes hedging cheaper for the firm and the sleeve dearer.",
    ),
    "raise_cost_after_btc_halves": (
        "Raising equity after BTC halves",
        "How many times dearer it is to raise equity straight after BTC has fallen by half. Investors are hardest to "
        "find when the firm most needs them.",
        "A multiple, at least 1. 1 means no dearer; the example uses 3. Smaller falls scale in proportion.",
    ),
    "raise_cost_after_incident": (
        "Raising equity after an incident",
        "How many times dearer it is to raise equity straight after an incident.",
        "A multiple, at least 1. The example uses 3.",
    ),
    "btc_volatility": (
        "BTC volatility",
        "How much BTC's price swings in a year in ordinary times, not counting sudden crashes. Technically, the "
        "standard deviation of the yearly change in the logarithm of the price.",
        "A percentage a year. BTC's has usually been somewhere between 40% and 90%. The example uses 60%.",
    ),
    "btc_crashes_per_year": (
        "BTC crashes a year",
        "How many sudden crashes to expect in a year, on average: falls so fast that collateral cannot be sold on the "
        "way down. It is a rate, not a percentage.",
        "0.5 means one crash every two years; 1 means one a year; 0.1 means one every ten years. Zero switches crashes off.",
    ),
    "btc_crash_size": (
        "Average BTC crash",
        "When a crash happens, the share of BTC's price it takes on average. Individual crashes vary widely: most are "
        "smaller, a few far larger.",
        "A percentage, above 0 and below 100. The example uses 25%.",
    ),
    "btc_view": (
        "The firm's own view on BTC",
        "How much more BTC will return each year than the market expects, in the firm's opinion. The market's "
        "expectation here is the shareholders' required return. The averages and the multi-year paths are worked out "
        "as if the firm is right; the stress scenarios do not depend on it.",
        "A percentage a year. 0% means no view: plan with the market's expectation. 5% means plan as if BTC returns 5 "
        "points more a year. It can be negative.",
    ),
    "loan_demand": (
        "Loan demand",
        "How much borrowers would take at a given borrower rate, at the reference loan-to-value. Demand falls in a "
        "straight line as the rate rises.",
        "Give an amount and the rate it holds at. The example: about $72m at 10%.",
    ),
    "loan_demand.loans": ("Loans placeable", "The loans the firm could place at the rate below.", "In millions of dollars."),
    "loan_demand.rate": (
        "At a borrower rate of",
        "The rate at which that amount could be placed.",
        "A percentage a year, below the rate at which nobody borrows.",
    ),
    "rate_no_demand": (
        "Rate at which nobody borrows",
        "The borrower rate at which demand for loans disappears altogether.",
        "A percentage a year, above the rate loan demand is stated at. The example uses 20%.",
    ),
    "reference_ltv": (
        "Loan-to-value that demand is stated at",
        "The loan-to-value the loan demand above refers to. Offering more than this attracts more borrowers, less "
        "attracts fewer.",
        "A percentage. The example uses 50%.",
    ),
    "demand_per_10_ltv_points": (
        "Extra demand per 10 points of loan-to-value",
        "How much loan demand rises when the firm offers 10 points more loan-to-value.",
        "A percentage. 10% means 10 points more loan-to-value brings 10% more demand.",
    ),
    "demand_follows_btc": (
        "How far loan demand follows BTC's price",
        "Over several years, how loan demand moves with BTC's price. Borrowers post BTC, so what they can borrow moves "
        "with what it is worth. Used on the multi-year paths only.",
        "A number, usually 0 to 1. 0: demand ignores BTC. 1: demand halves when BTC halves. The example uses 0.5.",
    ),
    "dollar_outside_rate": (
        "What dollar investors can earn elsewhere",
        "The yearly return dollar investors could get on stablecoins somewhere else. When the vault pays less, money "
        "leaves; when it pays more, money comes.",
        "A percentage a year. The example uses 5%.",
    ),
    "btc_outside_yield": (
        "What BTC investors can earn elsewhere",
        "The yearly return, in BTC, that BTC investors could get somewhere else.",
        "A percentage a year. The example uses 2%.",
    ),
    "dollar_flow_response": (
        "How strongly dollar investors follow yield",
        "How much money moves into or out of the dollar vault for each point its return differs from what investors "
        "can earn elsewhere.",
        "A number. 2 means a return one point better over a period brings in 2% of the vault. Zero: investors do not react.",
    ),
    "btc_flow_response": (
        "How strongly BTC investors follow yield",
        "The same for the BTC vault, against what BTC investors can earn elsewhere.",
        "A number. The example uses 2.",
    ),
    "inflow_speed": (
        "How fast a vault fills toward capacity",
        "The share of the gap between a vault's size and its capacity that new money closes in a year, before any "
        "reaction to yield.",
        "A percentage a year, 0 to 100. 50% means half the gap closes each year.",
    ),
    "strategy_yield": (
        "BTC strategies: expected yield",
        "What each of the BTC vault's own strategies is expected to earn in a year, in BTC, before losses and fees.",
        "A percentage a year for each. The example: lending BTC 3%, basis trading 6%, cold custody 0%.",
    ),
    "strategy_volatility": (
        "BTC strategies: volatility",
        "How much each strategy's yearly yield varies in ordinary times.",
        "A percentage a year for each. Zero for cold custody, which earns nothing and risks little.",
    ),
    "strategy_crash_loss": (
        "BTC strategies: loss in a BTC crash",
        "What each strategy loses when BTC crashes, per unit of the crash's size.",
        "A number for each. 0.2 means a 50% crash costs the strategy 10% of its BTC. Zero for cold custody.",
    ),
    "strategy_correlation": (
        "How far lending BTC and basis trading move together",
        "Whether the two active strategies tend to have good and bad periods at the same time.",
        "A number from -1 to 1. 0: unrelated. 1: always together. The example uses 0.3.",
    ),
}

_GRC_HELP = (
    "A statement of how well GRC works on this risk: spending the amount below every year removes the stated share "
    "of the risk, and no amount of spending removes more than the limit. Nobody can observe this: try a range, and "
    "use \"What does this lean on?\" to see how much depends on it."
)
RISK = {
    "incidents_per_year": (
        "Incidents a year, with no GRC",
        "How many serious incidents to expect in a year in each vault if the firm spent nothing on GRC: an exploit, a "
        "custody failure, a compliance breach. It is a rate, not a percentage. GRC spend lowers it.",
        "0.1 means one every ten years in each vault; 1 means one a year. Zero switches incidents off.",
    ),
    "incident_loss": (
        "Average loss in an incident",
        "When an incident happens, the share of the assets exposed that is lost, on average. Most incidents are "
        "smaller, a few far larger.",
        "A percentage, above 0 and below 100. The example uses 5%.",
    ),
    "lending_loss": (
        "Everyday losses in lending",
        "The small operating losses of the lending business in a normal year, as a share of the money in it: errors, "
        "failed trades, minor fraud. GRC spend lowers them.",
        "A percentage a year. The example uses 0.2%.",
    ),
    "lending_loss_volatility": (
        "How much everyday lending losses vary",
        "How far those losses swing from year to year.",
        "A percentage a year. The example uses 0.5%.",
    ),
    "strategy_loss": (
        "BTC strategies: everyday losses",
        "The same small operating losses for each of the BTC vault's own strategies, in BTC.",
        "A percentage a year for each.",
    ),
    "strategy_incident_exposure": (
        "BTC strategies: exposure to incidents",
        "How much of an incident in the BTC vault each strategy takes.",
        "A number for each, 0 to 1. 1: the strategy takes the full loss. 0.2: a fifth of it.",
    ),
    "sale_discount": (
        "Sale discount in a liquidation",
        "How far below the market price liquidated collateral sells, because markets are thin and price feeds lag. "
        "GRC spend narrows it.",
        "A percentage, below 100% less the liquidation threshold (with an 80% threshold, below 20%). The example uses 10%.",
    ),
    "liquidation_slippage": (
        "Slippage in a smooth fall",
        "When BTC falls steadily through the liquidation threshold, how much of the fall beyond the threshold is lost "
        "before the collateral is sold. GRC spend narrows it.",
        "A percentage, 0 to 100. 0%: sold exactly at the threshold. 100%: sold only at the price the fall ended at. "
        "The example uses 25%.",
    ),
    "outflow_after_incident": (
        "Investors leaving the vault hit by an incident",
        "The share of a vault's money that investors withdraw after an incident in it, on top of the loss itself.",
        "A percentage, 0 to 100. The example uses 20%.",
    ),
    "outflow_from_other_vault": (
        "Investors leaving the other vault",
        "After an incident in one vault, the share of the other vault that would leave if the vault hit were the "
        "firm's whole business. The model scales it down by how large the vault hit is.",
        "A percentage, 0 to 100. Zero: an incident stays in the vault it hit. The example uses 10%.",
    ),
    "grc_wear_off": (
        "How fast GRC's effect wears off",
        "The share of the protection built by past GRC spend that is lost each year as people leave, systems change "
        "and threats evolve. It is why GRC has to be kept up.",
        "A percentage a year, above 0 and at most 100. The example uses 25%.",
    ),
    "grc_on_losses": ("What GRC does to everyday losses", _GRC_HELP, "Give a yearly spend, the share removed at it, and the limit."),
    "grc_on_spread": ("What GRC does to how much losses vary", _GRC_HELP, "Give a yearly spend, the share removed at it, and the limit."),
    "grc_on_incidents": ("What GRC does to how often incidents happen", _GRC_HELP, "Give a yearly spend, the share removed at it, and the limit."),
    "grc_on_sale_discount": (
        "What GRC does to liquidation losses",
        _GRC_HELP + " This one narrows both the sale discount and the slippage.",
        "Give a yearly spend, the share removed at it, and the limit.",
    ),
    "*.spend": (
        "Spending, a year",
        "The yearly GRC spend this statement is about, kept up year after year.",
        "In millions of dollars a year, above zero.",
    ),
    "*.reduction": (
        "Removes this share",
        "The share of the risk removed once that spend has been kept up.",
        "A percentage, from 0 to below the limit. 0% means GRC does nothing for this risk.",
    ),
    "*.limit": (
        "And no spend removes more than",
        "The most that any amount of GRC spend could remove. Some risk always remains.",
        "A percentage, above the share removed and at most 100. The example uses 70%.",
    ),
}

POLICY = {
    "equity_kept": (
        "Equity kept",
        "The equity the firm holds after this period's payout or raise. Less than its equity now means paying the "
        "difference out to shareholders; more means raising new equity, which is expensive.",
        "In millions of dollars. It must cover the own funds required and leave the runway in stablecoins after GRC "
        "spend, the firm's own BTC and its stakes; the page says so if it does not.",
    ),
    "grc_spend": (
        "GRC spend a year",
        "What the firm spends on governance, risk and compliance from now on. It builds protection that lowers "
        "incidents and losses, and wears off if not kept up.",
        "In millions of dollars a year. Compare with the GRC budget so far, under the firm's facts.",
    ),
    "own_btc": (
        "Own BTC",
        "BTC the firm holds for itself, bought with its equity, beyond its stake in the BTC vault. It costs the "
        "security cost to hold and moves with BTC's price unless hedged.",
        "In millions of dollars, zero or more. It comes out of the stablecoins the firm would otherwise hold.",
    ),
    "hedged": (
        "Of its BTC, hedged",
        "How much of the BTC the firm owns is protected against a fall, using short perpetual futures. Hedged BTC no "
        "longer gains or loses with the price, and earns the perp funding rate.",
        "In millions of dollars, from zero up to the BTC the firm owns: its own BTC plus its stake in the BTC vault.",
    ),
    "borrower_rate": (
        "Borrower rate",
        "The yearly interest rate the firm charges borrowers. Higher earns more per loan and attracts fewer borrowers.",
        "A percentage a year, from 0 up to the rate at which nobody borrows (under market assumptions).",
    ),
    "ltv": (
        "Loan-to-value",
        "How much the firm lends against each dollar of BTC collateral. Higher attracts more borrowers and leaves "
        "less room before a falling price causes losses.",
        "A percentage, within the range the firm may offer (under the firm's facts). Crypto lenders usually offer 30% to 70%.",
    ),
    "allocation": (
        "The BTC vault's allocation",
        "How the BTC vault is split across lending BTC, basis trading, cold custody, and the lending sleeve: BTC "
        "swapped into stablecoins and lent to the same borrowers as the dollar vault, hedged so it still tracks BTC.",
        "Four percentages that add up to 100.",
    ),
}

RULE = {
    "years": (
        "Years ahead",
        "How far the multi-year paths look.",
        "A number of years, above zero. Longer horizons take a little longer to work out.",
    ),
    "equity_buffer": (
        "Buffer above the capital needed",
        "Over several years, the equity the firm keeps above the least it may hold. Anything above the buffer is paid "
        "out; inside it, the firm does nothing. A bigger buffer means fewer costly raises and smaller payouts.",
        "In millions of dollars. Blank: the buffer the policy keeps today.",
    ),
    "largest_raise": (
        "Largest raise shareholders will make",
        "Over several years, the most new equity shareholders will put in at one time. If the firm needs more than "
        "this to get back to the capital it needs, it fails: shareholders walk away.",
        "In millions of dollars. Blank: the capital the firm needs today.",
    ),
}

GUIDE = {"firm": FIRM, "market": MARKET, "risk": RISK, "policy": POLICY, "rule": RULE}
_NESTED = {RaiseCost, LoanPoint, GrcEffect}
_PARTS = {"firm": FirmFacts, "market": MarketAssumptions, "risk": RiskAssumptions, "policy": Policy}


def resolve(part: str, name: str) -> tuple[str, str, str]:
    """The (label, meaning, what to enter) of an input. `name` is a field, or
    `field.subfield` for a statement's parts; a statement's parts fall back to
    the entry shared by all statements of its kind (`*.subfield`)."""
    entries = GUIDE[part]
    if name in entries:
        return entries[name]
    return entries["*." + name.split(".", 1)[1]]


def names(part: str) -> list[str]:
    """Every input of a part, statements' parts included as `field.subfield`."""
    if part == "rule":
        return list(RULE)
    out = []
    for f in fields(_PARTS[part]):
        out.append(f.name)
        if f.type in _NESTED:
            out += [f"{f.name}.{sub.name}" for sub in fields(f.type)]
    return out


def guide() -> dict:
    """Every input's entry, by part: {part: {name: {label, help, enter}}}."""
    return {
        part: {name: dict(zip(("label", "help", "enter"), resolve(part, name), strict=True)) for name in names(part)}
        for part in GUIDE
    }
