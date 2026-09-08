# Quantitative GRC model — a tractable first formulation

This document turns `readme.md`'s aspirational goal into an explicit formulation,
and recommends a first tractable implementation. It builds directly on the choices made in
[`framework.md`](framework.md) — in particular, the firm-value definition in
its §3.

## 1. State variables

- Capital / equity, `E`
- Deposit volume, `D`
- Portfolio composition and risk metrics of deployed assets (e.g. expected
  loss, volatility, concentration)
- Liquidity buffer, `L`
- Operational-risk indicator (incident frequency/severity estimate, driven
  by the tech-infra overlay in `framework.md` §2)
- Compliance-risk score (proximity to breaching a binding regime from
  `framework.md` §1)

## 2. Controls

- GRC investment levels: risk staffing/tooling spend, compliance spend,
  security/tech-infra investment
- Capital allocation across lending/investment strategies
- Target liquidity buffer
- Deposit pricing / redemption-term incentives

## 3. Frictions and costs — where Froot-Stein plugs in

- **Convex cost of external financing** under a capital shortfall — the
  core Froot-Stein term: as `E` falls, the marginal cost of raising more
  capital rises faster than linearly.
- **Cost of regulatory penalties** — triggered by breaching a compliance
  constraint from `framework.md` §1.
- **Cost of tech/operational incidents** — losses and downside from the
  crypto-native risk column in `framework.md` §1 (hacks, oracle failures,
  depegs), scaled by the operational-risk state variable.
- **Cost of GRC investment itself** — assumed to have diminishing returns
  (each additional dollar of GRC spend reduces expected distress cost by
  less than the last).

## 4. Objective

Maximize the firm-value definition fixed in `framework.md` §3 — expected PV
of net cash flows minus expected distress/friction costs above — subject to:
- Solvency (`E > 0` under the modeled loss distribution)
- Liquidity (buffer `L` sufficient against a modeled redemption-run
  scenario)
- Regulatory constraints (the binding regimes named in `framework.md` §1)

This is the honest statement of the "full optimal control problem" the
readme already flags as impractical — stated explicitly here so later
simplifications are visible tradeoffs, not silent ones.

## 5. A tractable path to a first model

Solving §4 exactly is out of scope. Recommended sequence, each step
producing something usable before moving to the next:

1. **Static two-period Froot-Stein model.** Collapse the state to just `E`
   and one risk shock; derive the qualitative result that optimal GRC
   investment rises with the convexity of distress costs. Purpose: sanity
   check that the cost/objective shapes chosen in §3–4 actually produce the
   Froot-Stein intuition before building anything more elaborate.

2. **Monte Carlo simulation over risk events.** Simulate draws of 
* investment opportunities, losses, tech/operational incidents, and compliance breaches; 
   compare the resulting expected-value / loss-distribution across a few candidate GRC
   investment policies.

3. **MDP / value iteration (further out).** Only worth pursuing once the
   state space from §1 has been scoped down small enough to be numerically
   tractable — flagged here as a future option, not a near-term step.

## 6. First implementable milestone

A small Python prototype implementing step 5.1/5.2 above: the static
Froot-Stein tradeoff, plus a basic Monte Carlo loop over the three risk
categories in §3.

**Status: in progress.** Implemented in PyTorch (autograd + Adam), with
Apple Silicon (MPS) as the default device, in the `quant/` package:
- `quant/model.py` — the shared `FirmValueModel` (§3–4's cost/objective
  terms) and `optimize_policy()` autograd loop, used by both stages below.
- `quant/static.py` (§5.1) 
  * a two-state stochastic shock (needed for the
  financing-cost convexity to actually bite via Jensen's inequality
  * with a single deterministic risk shock, the optimal GRC investment comes out independent of financing-cost convexity 
  * mathematically, the Froot-Stein effect is a Jensen's-inequality result that requires genuine uncertainty, not just a convex cost applied to a fixed number. 
  * Claude switched the static model's shock to a two-state stochastic draw (reusing the batch-mean path already built for Monte Carlo), which restored the expected result: optimal `g` rises monotonically from 2.92 to 4.32 as convexity goes from 1.0 to 4.0.
  * Full derivation and debugging trace: [`static-model-debug-notes.md`](static-model-debug-notes.md).
  * Run: `uv run python scripts/run_static_model.py`.
- `quant/simulate.py` (§5.2) — placeholder risk-shock samplers per
  `framework.md` §1's taxonomy, feeding the same `FirmValueModel` /
  `optimize_policy()` with a large batch instead of two states. Run:
  `uv run python -m quant.simulate`.
- `tests/test_static_model.py` — checks the qualitative Froot-Stein result
  (optimal GRC investment rises with convexity) holds.

Not yet done: calibrating `quant/simulate.py`'s distributions to anything
real, and §5.3 (MDP).
