# GRC

A quantitative model for **G**overnance, **R**isk and **C**ompliance. 
GRC is normally treated as a cost center. The thesis here is that GRC (done well) **increases firm value**, and 
this can be made quantitative rather than asserted.


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

The model is being rebuilt as a dynamic version of Froot & Stein (1998): a bank
that decides risk management, capital budgeting and capital structure together,
every period, with no end date. GRC is a fourth lever alongside those three. The
design, and the reasoning behind each choice, is in
[`docs/design.md`](docs/design.md).

Built so far (v0 foundations):

- the bank's one-period accounting, with deposits, a capital ratio and GRC as a
  depreciating stock (`quant/model.py`);
- fat-tailed shocks with tail dependence, integrated by deterministic
  quadrature rather than Monte Carlo (`quant/quadrature.py`);
- a two-period version that reproduces the paper's full-hedging result and its
  capital-budgeting rule exactly under normal shocks, and measures how far
  that rule drifts under fat tails (`quant/twoperiod.py`).

Next: the infinite-horizon solver.

```bash
uv run pytest
```

The previous model (finite horizon, hazard-based failure) is preserved on the
`legacy` branch.

## See also

* [`docs/overview.md`](docs/overview.md) — the project in plain language, for non-technical readers
* [`docs/design.md`](docs/design.md) — the model being built, and why
* [`docs/framework.md`](docs/framework.md) — GRC pillars, risk taxonomy, operating environment, and the firm-value definition this project uses
