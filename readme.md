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
if losing money in bad states stops it from funding good investments. Risk management is then not insurance or reducing volatlity — it is protection of the firm's capacity to keep investing and operating.

1. Risk Management: Coordinating Corporate Investment and Financing Policies (1993).
Kenneth A. Froot, David S. Scharfstein, Jeremy C. Stein, [The Journal of Finance, 48(5)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb05123.x).

2. Risk Management, Capital Budgeting, and Capital Structure Policy for Financial Institutions: An Integrated Approach (1998). Kenneth A. Froot, Jeremy C. Stein, [Journal of Financial Economics, 47(1)](https://www.sciencedirect.com/science/article/pii/S0304405X97000378?via%3Dihub).

Financial institutions should evaluate investments not just by their market risk, but also by how they interact with the institution's existing portfolio of **unhedgeable risks** and its scarce capital. The paper develops an integrated framework linking risk management, capital allocation, and investment decisions, including a comparison with RAROC. 


## Model
The firm is modeled as a **financial intermediary on crypto rails**: it takes
deposits into vaults, and invests them. It runs the tokenized
infrastructure those assets live on. 

It carries the usual set of banking risks 
— credit is the largest single loss channel at 62% of expected loss.

A second family the rails add: 
- operational incidents at 16%
- cliff events such as a drained bridge at 13%
- and compliance penalties at 9% 

A firm of this shape has to budget a GRC program against both.

Todo: no units, ok. But these numbers seem out of place at the moment.
> **Note on units.** Money here has no specific denomination. Opening capital is 16 and
> every other quantity — a book of 77, a program costing 0.95, a franchise of
> 26.93 — is in the same arbitrary currency. Only ratios carry meaning, which is
> deliberate: the cost functions are built so the answer cannot depend on
> whether the firm is denominated in dollars or cents. Bare numbers are not
> percentages.

## Status

The model is a dynamic version of Froot & Stein (1998): a crypto-native firm
that decides risk management, capital budgeting and capital structure together,
every quarter, with no end date. GRC is a fourth lever alongside those three.
The design, and the reasoning behind each choice, is in
[`docs/design.md`](docs/design.md); a plain-language account is in
[`docs/overview.md`](docs/overview.md).

Built so far:

- **the stablecoin lending vault** (`quant/lending.py`, solved on a 3-D grid by
  `quant/research/lending_solver.py`): loans over-collateralized by BTC at a rate and
  loan-to-value the firm sets, liquidation at the threshold, bad debt from BTC
  crashes;
- **the BTC yield vault** (`quant/btc_vault.py`, solved on a 3-D grid by
  `quant/research/btc_vault_solver.py`): strategies, fees in BTC, flows;
- **the integrated firm** (`quant/integrated.py`): both vaults joined through
  one loan book, funded by the dollar vault and by a hedged lending sleeve of
  the BTC vault; solved on a coarse 4-D grid (`quant/research/integrated_solver.py`)
  and read out as a CFO would want it (`quant/research/integrated_results.py`,
  `scripts/research/integrated.py`);
- **a firm profile in real units** (`quant/profile.py`): a firm entered in
  dollars and yearly rates, mapped onto the model and back — the first step of
  the what-if tool;
- **the policy as an input** (`quant/policy.py`) and **the one-period
  read-out** (`quant/readout.py`): what the next period holds for that firm
  under the user's own decisions, with no search for the best ones;
- the firm's own books, shared by both (`quant/firm.py`): stablecoins and BTC,
  a hedge with perps, a runway of operating costs; BTC at the market's
  expectation plus the firm's bias, with occasional crashes;
- fat-tailed shocks with tail dependence and heavy-tailed jump losses,
  integrated by deterministic quadrature rather than Monte Carlo
  (`quant/quadrature.py`);
- an infinite-horizon solver on an N-dimensional grid (`quant/research/grid.py`), checked
  against the paper's own results in a two-period version (`quant/research/twoperiod.py`).

An earlier single dollar vault and a bank form were built and removed.

Next: a what-if tool for a CFO or risk manager, built on the accounting — the
user's own firm and policy in, what can happen out (`docs/plan-tool.md`). The
search for the best policy (grid solvers, and an actor-critic learner still
to build) continues as research (`docs/plan-integrated.md`).

```bash
uv run pytest               # seconds; add `-m slow` for the solver tests (about five minutes)
uv run python -m scripts.whatif --rate 0.08 --sleeve 0.2    # an example firm, one quarter ahead
uv run python -m scripts.research.integrated --periods 1
```

The previous model (finite horizon, hazard-based failure) is preserved on the
`legacy` branch.

## See also

* [`docs/overview.md`](docs/overview.md) — the project in plain language, for non-technical readers
* [`docs/design.md`](docs/design.md) — the model being built, and why
* [`docs/framework.md`](docs/framework.md) — GRC pillars, risk taxonomy, operating environment, and the firm-value definition this project uses
