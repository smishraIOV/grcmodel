# Plan: the integrated two-vault model, solved by an actor-critic learner

Written 2026-11-13 so that another person or agent can pick the work up
without the conversation that produced it. Read this first, then
`CLAUDE.md` (working rules) and `docs/design.md` (the specification; §2 and §3
especially). Where this plan and the design disagree, the design wins; update
this plan.

**Status:** planned, not started. Steps need the user's approval as they come
(see "Working rules").

---

## 1. Where things stand

The project models a crypto-native firm as a dynamic version of Froot & Stein
(1998): risk management, capital budgeting and capital structure decided
together, every period, with no end date, and **GRC** (governance, risk,
compliance) as a fourth lever. One end goal is a **decision-support tool for
the CFO of a real business**.

Built and committed on branch `rework` (all tests pass with `uv run pytest`;
the slow solver tests take several minutes):

| file | what it is |
|---|---|
| `quant/params.py` | every parameter, stated per year; `per_period()` converts |
| `quant/core.py` | issuance cost, GRC mitigation and stock, jump probability, shareholders' cash flow |
| `quant/firm.py` | the firm's own books (stablecoins earning nothing, its BTC with a 1.5%/yr security cost, a short-perp hedge, a runway of a year's operating costs in stablecoins, the capital floor) and BTC's market (smooth fat-tailed move plus rare crashes, drift set so BTC's expected return is the market's `r_E` plus the firm's bias) |
| `quant/lending.py` | the stablecoin lending vault, one period: loan demand linear in the rate, liquidation at the threshold price with a sale discount GRC narrows, bad debt only from BTC crashes; one funding source |
| `quant/btc_vault.py` | the BTC yield vault, one period: three strategies (lending BTC, basis trading, cold custody), fees in BTC, crash and incident losses |
| `quant/lending_solver.py`, `quant/btc_vault_solver.py` | each vault alone on a 3-D grid (equity w, GRC stock S, vault size A), by policy iteration (`quant/grid.py`) |
| `quant/lending_results.py` | cost ratios (payroll and operating costs against revenue) |
| `quant/twoperiod.py` | the paper's own two-period bank, as a known-answer check |
| `tests/` | one test file per module; solver tests run annual and coarse |

Findings so far (exploratory, annual, not to be quoted in docs): neither vault
alone buys GRC; the BTC vault alone loses money at 2% operating costs and puts
everything into the highest-yield strategy; with a BTC bias the firm keeps
most of its equity in unhedged BTC. The user suspects something fundamental
is still missing and has chosen to build the integrated model before
revisiting it. Do not add investor risk-aversion, asymmetric flows or a
reputation state without the user asking.

## 2. What the integrated model is (agreed with the user)

Design reference: `docs/design.md` §2.1 ("One loan book, two sources of
funding") and §2.3 ("Stablecoin lending as a strategy, hedged").

- **State:** (w, S, A^L, A^B) — the firm's equity, its GRC stock, the dollar
  vault's size and the BTC vault's size, both in dollars.
- **One loan book.** Borrowers take b in stablecoins and post b / LTV in BTC.
  One borrower rate and one LTV, both set by the firm. Liquidation, crashes,
  sale discount and bad debt exactly as in `quant/lending.py`.
- **Two funding sources.** The dollar vault (A^L, outside stablecoin
  investors) and the BTC vault's **lending sleeve**: the BTC vault swaps a
  share a_4 of itself into stablecoins and lends them. Supply = A^L + a_4 A^B.
  Loan book L = min(demand(rate, LTV), supply).
- **Pro-rata sharing.** Each source lends in proportion to what it makes
  available; interest, penalties and bad debt are shared in the same
  proportion; unlent stablecoins of either source sit idle and earn nothing.
- **The sleeve is hedged.** The BTC vault holds a long BTC perp position of
  the sleeve's dollar size, so the sleeve still moves with BTC. In dollars the
  sleeve earns: its share of loan payoffs, plus the perp's BTC return, minus
  perp funding at the riskless rate r. (Funding premium above r: **pending**.)
- **BTC vault allocation** over four strategies: lending BTC, basis trading,
  cold custody, the hedged stablecoin-lending sleeve. Fees as now (1%/yr of
  the vault's BTC plus 15% of positive BTC yield, paid in BTC).
- **The firm** earns its interest share on the whole loan book and the BTC
  vault's fees; pays one set of operating costs (fixed in dollars) and GRC;
  repays a share of incident losses and bad debt to outside investors; keeps
  co-investment stakes in both vaults; holds its own equity as stablecoins and
  BTC with a hedge, under the shared rules in `quant/firm.py`.
- **Governance angle** (design §8, item 5): the firm's rate and the BTC
  vault's lending share move value between its two investor groups. Report
  it; do not build anything extra for it.

## 3. Decisions still pending (ask the user)

1. Relative size of the vaults: both capacities 1, or BTC vault larger.
2. Perp funding cost for the sleeve: r, or r plus a premium.
3. Operating costs for the combined firm: recalibrate against combined
   revenue so payroll is at least 30% and all operating costs 50–60% of
   revenue (the agreed bounds; `quant/lending_results.py`).

## 4. How it is solved: the actor-critic learner

Concepts 1, 2, 3 and 5 below were proposed and not objected to; concept 4 was
changed by the user to random scenarios. Re-confirm before relying on any of
them in a way not described here.

1. **Networks.** Two small fully-connected networks (a few layers of about 64
   units, smooth activations such as softplus): the **critic** V(state) and
   the **actor**, which outputs the raw (unconstrained) decisions. Decisions
   are squashed exactly as in the grid solvers (`sigmoid`, `softplus`,
   `softmax`), so every constraint holds by construction.
2. **Training mirrors the grid solver** (`quant/grid.py`), with networks in
   place of the grid. Repeat:
   - draw a batch of states;
   - **actor step:** maximize flow(w, K) + beta E[V_target(next state)] by
     gradient ascent through the known one-period model;
   - **critic step:** fit V(state) to max{0, flow + beta E[V_target(next)]}
     (walk-away floor);
   - update the **target network** V_target slowly toward V (standard).

   Outside the equity range, use the grid's rules: above the top, value grows
   one-for-one (pay out); below the bottom, issue equity back onto the range
   at cost C. Other state axes are clamped to their range.
3. **Training states:** a mix of uniform draws over the state box and states
   the current policy visits. (Own construction, close to standard practice.)
4. **Expectations by random scenarios** (user's decision). Rules:
   - draw on CPU in float64 with a seeded generator, cast last
     (`NumericsProfile.generator`, `CLAUDE.md`);
   - Student-t shocks as a scale mixture: one chi-squared W per scenario
     shared across components, normals correlated by the Cholesky factor;
   - BTC crashes: sampled (their probability does not depend on decisions);
   - **incidents must not be sampled as yes/no**, because their probability
     depends on GRC spend and sampling would kill the gradient. For each
     scenario, keep two branches — no incident with weight 1 - p(S'), an
     incident with a sampled severity with weight p(S') — as the grid code
     does with its quadrature weights;
   - fresh scenarios each training step; a fixed seeded set for evaluation
     and grading, so graded numbers are reproducible.
5. **Numbers:** float64 on CPU (the `reference` profile). If speed forces
   float32 or MPS, re-check final values in float64 before quoting anything.

## 5. Steps

Each step ends with tests passing and a commit. Stop for the user's review
where marked.

| step | what | acceptance |
|---|---|---|
| 1 | `quant/integrated.py`: one period of the integrated model, reusing `quant/firm.py`, `quant/lending.py` (`loan_payoff`, `loan_demand`) and `quant/btc_vault.py` pieces. Tests in `tests/test_integrated.py` against hand-worked cases. | Matches the hand-worked formulas to 1e-14. **Reduces exactly** to the lending vault alone when A^B = 0, and to the BTC vault alone (three strategies) when A^L = 0 and a_4 = 0. Pro-rata sharing checked. Sleeve hedge checked (sleeve's BTC-terms value unchanged by a BTC move, apart from interest and funding). **Stop for review.** |
| 2 | `quant/learner.py`: scenario sampler (section 4.4), networks, actor-critic loop (section 4.2), checkpointing. | Unit tests: sampler moments (variance, crash frequency), incident branch weights sum to one and carry a gradient in GRC spend, squashed decisions within bounds. |
| 3 | Grade the learner on the two solved 3-D problems: the lending vault alone and the BTC vault alone (use the grid solvers in `quant/lending_solver.py`, `quant/btc_vault_solver.py` as the reference). | Value within a stated tolerance of the grid over the grid's states (propose ~1% of the value's range), decisions close where the value is not flat. Report where they differ and why. |
| 4 | A coarse 4-D grid as a second reference: extend the grid solver to the integrated objective at about 17 × 3 × 5 × 5 points. | Learner and coarse grid agree within the coarse grid's own error. |
| 5 | Integrated runs (annual to explore, quarterly for realistic) and a read-out: GRC spend, the firm's capital and BTC, the allocation including the sleeve, the borrower rate and its split between investor groups, cost ratios. Update `docs/design.md` status and §7, `docs/overview.md` (plain language, no formulas), `CLAUDE.md` layout, `readme.md` status. | Numbers only in docs if from `reference` runs and the user agrees they are worth reporting (`CLAUDE.md`: no model results in docs yet). |

Rough cost: 2–3 working days plus compute.

## 6. Working rules that matter here (from `CLAUDE.md` and the user)

- Ask before long work; give estimates. Stages and steps need approval.
- Raise any new theoretical or numerical concept (with its source) before it
  goes into code or docs; record agreed ones in the design's decisions log.
- When the user asks a question, answer it; do not edit files.
- Plain language, US spelling, short replies.
- No comparisons against unrealistic counterfactuals (free equity, forced
  recapitalization, unlimited liability).
- Every function points to the design section it implements; keep both in
  sync. Every entry-point script is imported by a test.
- Commit each finished step with a message ending
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
