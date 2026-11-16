# GRC
This project is a research exercise.

A quantitative model for **G**overnance, **R**isk and **C**ompliance. 
GRC is normally treated as a cost center. The thesis here is that GRC (done well) **increases firm value**, and 
this can be made quantitative rather than asserted.

Even experts do not know how much risk to budget for. A chief actuary may not know how much reinsurance to purchase.
A computational tool can provide some guidance.
One of the main end goals is a model that works as a **decision-support tool for the CFO of a real business**.

## Academic Roots

Even a risk-neutral firm will pay to manage risks,
if losing money in bad states stops it from funding good investments. Risk management is then not insurance or reducing volatility — it is protection of the firm's capacity to keep investing and operating.

1. Risk Management: Coordinating Corporate Investment and Financing Policies (1993).
Kenneth A. Froot, David S. Scharfstein, Jeremy C. Stein, [The Journal of Finance, 48(5)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb05123.x).

2. Risk Management, Capital Budgeting, and Capital Structure Policy for Financial Institutions: An Integrated Approach (1998). Kenneth A. Froot, Jeremy C. Stein, [Journal of Financial Economics, 47(1)](https://www.sciencedirect.com/science/article/pii/S0304405X97000378?via%3Dihub).

Financial institutions should evaluate investments not just by their market risk, but also by how they interact with the institution's existing portfolio of **unhedgeable risks** and its scarce capital. The paper develops an integrated framework linking risk management, capital allocation, and investment decisions, including a comparison with RAROC. 


## The firm

The model is of a **crypto-native financial firm** that runs two vaults and
earns fees on both:

- **a dollar lending vault**: investors deposit stablecoins, which the firm
  lends to borrowers who post BTC worth more than the loan. If BTC falls far
  enough the collateral is sold to repay the loan; a sudden crash can leave
  bad debt;
- **a BTC yield vault**: investors deposit BTC and want more BTC back. The
  firm spreads it over strategies, and may swap part of it into stablecoins
  and lend those to the same borrowers, hedged so that it still tracks BTC.

Each period the firm decides how much equity to keep, how much BTC to hold
and hedge, what to charge borrowers and how much to lend against their
collateral, how to allocate the BTC vault, and how much to spend on GRC. GRC
spend builds protection that makes incidents rarer and losses smaller, and
wears off if it is not kept up. Following Froot and Stein, raising new equity
is expensive, so a loss costs the firm more than its face value.

## What is here

**A what-if tool for a CFO or risk manager.** Enter a firm and what it does;
see what could happen to it. It does not search for the best policy.

- **A dashboard** on your own machine: a form for the firm, its assumptions
  and its policy, with what each input means beside it; results as you type.
  A firm can be saved to a file and loaded back.
- **One period ahead**: what the firm expects to earn and spend, what its two
  groups of investors expect, and how likely the period is to end badly.
- **Stress scenarios**: what a fall in BTC, a crash or an incident would
  leave the firm, its vaults and their investors.
- **Several years ahead**: the firm carried forward on thousands of simulated
  paths — how often it has to raise equity, how often it fails, what it pays
  out.
- **Two policies side by side**: what changing from one to the other buys
  and costs.
- **What a figure leans on**: how much each result depends on the assumptions
  nobody can observe, such as how well GRC works.
- **An interface for other programs**: the same, as JSON in and JSON out.

**Research: the best policy.** Solvers that work out what the firm *should*
do in every situation, by policy iteration on a grid, for each vault alone
and coarsely for the two together. They are less mature than the tool, and a
learner to solve the full problem finely is planned, not built.

**The assumptions are illustrative, not calibrated.** The mechanics are
tested; the levels mean little yet. Calibration is the next stage.

## Running it

Everything goes through [`uv`](https://docs.astral.sh/uv/).

```bash
uv run python -m scripts.dashboard     # the dashboard, in a browser on this machine
uv run pytest                          # the tests: under a minute
uv run pytest -m slow                  # the solvers' tests: several minutes
```

The same tool from the command line, on a made-up example firm:

```bash
uv run python -m scripts.whatif --ltv 0.7 --stress --years 5                     # one quarter ahead, under stress, over five years
uv run python -m scripts.whatif --ltv 0.5 --b-ltv 0.7 --b-sleeve 0.2 --years 5   # two policies side by side
uv run python -m scripts.whatif --b-grc-spend 1.5e5 --years 5 --sensitivity      # what more GRC spend is worth, and what that leans on
uv run python -m scripts.api --example > request.json                            # an example request, as JSON
uv run python -m scripts.api request.json                                        # and its answer
```

## Where things are

- `quant/` — the model. The accounting for one period (`lending.py`,
  `btc_vault.py`, `integrated.py`, `firm.py`), and the tool built on it
  (`profile.py`, `policy.py`, `readout.py`, `stress.py`, `paths.py`,
  `compare.py`, `sensitivity.py`, `guide.py`, `api.py`).
- `quant/research/` — the solvers.
- `dashboard/` and `scripts/` — the dashboard's page, and the commands above.
- `tests/` — one file per module.
- `docs/` — below.

An earlier model (finite horizon, failure as an assumed rate) is kept on the
`legacy` branch.

## See also

* [`docs/overview.md`](docs/overview.md) — the project in plain language, for non-technical readers
* [`docs/design.md`](docs/design.md) — the model, and the reason for each choice
* [`docs/plan-calibration.md`](docs/plan-calibration.md) — the current plan
* [`docs/plan-tool.md`](docs/plan-tool.md) — how the what-if tool was planned and built
* [`docs/plan-integrated.md`](docs/plan-integrated.md) — the research on the best policy: what is built and what is open
* [`docs/framework.md`](docs/framework.md) — GRC pillars, risk taxonomy, operating environment, and the firm-value definition this project uses
