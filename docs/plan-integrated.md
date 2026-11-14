# Plan: the integrated two-vault model, solved by an actor-critic learner

**This is now the research path's plan** (2026-11-13). The project's current
plan is `docs/plan-tool.md`: a what-if tool built on the accounting, with no
search for the best policy. The learner and the grid solvers continue here as
research, at lower priority.

Written 2026-11-13 so that another person or agent can pick the work up
without the conversation that produced it. Read this first, then
`CLAUDE.md` (working rules) and `docs/design.md` (the specification; §2 and §3
especially). Where this plan and the design disagree, the design wins; update
this plan.

**Status (2026-11-13, later the same day):** steps 1 and 4 are built, with
a read-out and an entry point (part of step 5); step 4a has had a first
pass; steps 2 and 3 (the learner) are not started. The order changed: the
coarse 4-D grid came before the learner (section 5). Choices made while
building are listed in section 3; the user has confirmed them. Sections 6
and 7 record what the first solves showed and what is open in the solver.
Steps need the user's approval as they come (see "Working rules").

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
| `quant/research/lending_solver.py`, `quant/research/btc_vault_solver.py` | each vault alone on a 3-D grid (equity w, GRC stock S, vault size A), by policy iteration (`quant/research/grid.py`) |
| `quant/research/lending_results.py` | cost ratios (payroll and operating costs against revenue) |
| `quant/integrated.py` | the integrated firm, one period (section 2): one loan book, two funding sources, the hedged sleeve |
| `quant/research/integrated_solver.py` | the integrated firm on a coarse 4-D grid (w, S, A^L, A^B), by the same policy iteration |
| `quant/research/integrated_results.py` | the read-out at one state: decisions, expected income and costs, investors' yields, how the period may end, cost ratios against combined revenue |
| `scripts/research/integrated.py` | solves and prints the read-out: `uv run python -m scripts.research.integrated --periods 1` |
| `quant/research/twoperiod.py` | the paper's own two-period bank, as a known-answer check |
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
  perp funding at **r plus a premium** (user's decision, 2026-11-13; see
  section 3).
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

## 3. Decisions made, and what is still to confirm

Made by the user on 2026-11-13:

1. **The BTC vault is twice the dollar vault** (confirmed): BTC vault
   capacity 2, dollar vault capacity 1 (the scale stays "dollar vault
   capacity = 1").
2. **Perp funding is r plus a 3% a year premium, on every perp position**
   (confirmed): a parameter `perp_premium` = 0.03, one market funding rate.
   Longs pay it, shorts receive it. The BTC vault's long sleeve hedge pays
   r + premium; the firm's own short hedge **earns** r + premium, so hedged
   BTC becomes a carry trade earning r + premium less the 1.5% security cost.
   This changes the single-vault solvers' hedging economics too: apply it in
   `quant/firm.py` (`treasury_close`) and re-run their tests.
3. **Operating costs are recalibrated for the integrated firm** — a goal of
   this stage, not a one-off. One set of fixed dollar costs for the whole
   firm, set against **combined revenue** (the lending interest share plus
   the BTC vault's fees) at the terms the firm chooses, within the agreed
   bounds: payroll at least 30% and all operating costs 50–60% of revenue.
   The cost-ratio check (`quant/research/lending_results.py`) becomes an integrated
   check. Payroll stays sticky (fixed in dollars); layoffs after a large shock
   remain a later option.

Made by Claude while building step 1 and **confirmed by the user**
(2026-11-13); each is also in the design's decisions log:

4. **At most one incident a period, in either vault**: none, one in the
   lending system, or one in the BTC vault's other strategies, each with the
   single-vault chance. This keeps both reductions exact and the branches
   few. Independent incidents would add a small chance of two at once.
5. **A sleeve dollar is a dollar-vault dollar** in everything the lending
   system does: share lent, interest, penalties, bad debt, operating losses,
   the firm's interest share, and lending-system incidents. The BTC vault's
   own fees are then charged on top, so the firm is paid twice on the sleeve.
6. **Withdrawals after an incident come from the vault hit only.** The design
   allows both vaults to lose investors (§2.5); that needs a parameter.
7. **The capital requirement is on the two vaults' combined size;** the hedge
   may cover the firm's own BTC and its stake in the BTC vault.
8. **A floor on the BTC price (1% of its start)** where the sleeve's dollar
   results are put in BTC terms: a numerical guard.
9. **`INTEGRATED_DEFAULTS`** in `quant/params.py` carries the 2:1 sizes;
   `DEFAULTS` is unchanged, so the single-vault solvers and tests are too.
   The perp premium is in `DEFAULTS`, as decided, and the Prop. 2 test sets it
   to zero ("fair hedging").

## 4. How it is solved: the actor-critic learner

Concepts 1, 2, 3 and 5 below were proposed and not objected to; concept 4 was
changed by the user to random scenarios. Re-confirm before relying on any of
them in a way not described here.

1. **Networks.** Two small fully-connected networks (a few layers of about 64
   units, smooth activations such as softplus): the **critic** V(state) and
   the **actor**, which outputs the raw (unconstrained) decisions. Decisions
   are squashed exactly as in the grid solvers (`sigmoid`, `softplus`,
   `softmax`), so every constraint holds by construction.
2. **Training mirrors the grid solver** (`quant/research/grid.py`), with networks in
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

**Order changed (2026-11-13):** step 4 was built straight after step 1. The
grid solver already handles any number of dimensions, a coarse annual solve
takes about five minutes, and it shows what the integrated firm does before
two to three days go into a learner built to solve it finely.

| step | what | acceptance | status |
|---|---|---|---|
| 1 | `quant/integrated.py`: one period of the integrated model, reusing `quant/firm.py`, `quant/lending.py` (`loan_payoff`, `loan_demand`) and the BTC vault's formulas. Tests in `tests/test_integrated.py` against hand-worked cases. | Matches the hand-worked formulas to 1e-14. **Reduces exactly** to the lending vault alone when A^B = 0, and to the BTC vault alone (three strategies) when A^L = 0 and a_4 = 0. Pro-rata sharing checked. Sleeve hedge checked (sleeve's BTC-terms value unchanged by a BTC move, apart from interest and funding). **Stop for review.** | **done**; hand-worked cases to 1e-14, reductions scenario by scenario to 1e-13. The choices in section 3 are confirmed |
| 4 | A coarse 4-D grid as a second reference: `quant/research/integrated_solver.py`, about 17 × 3 × 5 × 5 points. | Converges; value non-negative and increasing in equity; decisions within bounds. | **built, but it settles only to about 0.2% of the firm's value** (section 7). Not yet checked: that more quadrature points or grid points do not move the answer |
| 4a | **Recalibrate operating costs** for the integrated firm (section 3, item 3): solve, read combined revenue at the chosen terms, set the fixed dollar costs so the ratios sit inside the bounds, re-solve, and check again (the costs move the terms, so iterate until stable). The integrated cost-ratio check is `ReadOut.within_bounds` in `quant/research/integrated_results.py`. | Payroll ≥ 30% and operating costs 50–60% of combined revenue at a full-size firm, after the final solve. | **first pass only** (section 6); the default is not changed until the user agrees the figure |
| 2 | `quant/learner.py`: scenario sampler (section 4.4), networks, actor-critic loop (section 4.2), checkpointing. | Unit tests: sampler moments (variance, crash frequency), incident branch weights sum to one and carry a gradient in GRC spend, squashed decisions within bounds. | not started. The incident branches are now three (none, lending system, BTC vault), not two |
| 3 | Grade the learner on the two solved 3-D problems: the lending vault alone and the BTC vault alone (use the grid solvers in `quant/research/lending_solver.py`, `quant/research/btc_vault_solver.py` as the reference), then on the coarse 4-D grid. | Value within a stated tolerance of the grid over the grid's states (propose ~1% of the value's range), decisions close where the value is not flat. Report where they differ and why. | not started |
| 5 | Integrated runs (annual to explore, quarterly for realistic) and a read-out: GRC spend, the firm's capital and BTC, the allocation including the sleeve, the borrower rate and its split between investor groups, cost ratios. Update `docs/design.md` status and §7, `docs/overview.md` (plain language, no formulas), `CLAUDE.md` layout, `readme.md` status. | Numbers only in docs if from `reference` runs and the user agrees they are worth reporting (`CLAUDE.md`: no model results in docs yet). | **partly**: the read-out and `scripts/research/integrated.py` are built; design, `CLAUDE.md` and readme are updated. `docs/overview.md` still describes the earlier single vault |

## 6. What the first integrated solves showed

Exploratory, annual, coarse, and from solves that had not fully settled
(section 7): shapes only, not to be quoted in docs.

- The integrated firm **buys GRC**, which neither vault alone did.
- It uses the **sleeve when the vaults are small** and not at full size.
- It holds **exactly its capital floor** at every state looked at, and often
  ends a year below it: raising equity is cheap next to carrying it. A
  constraint that always binds (`CLAUDE.md`, "check the regime") — a candidate
  for what is missing.
- The BTC vault still goes **all-in on the highest-yield strategy** at full
  size: investors do not price risk.
- Operating costs at the single-vault figure are about half the agreed share
  of combined revenue; about twice that figure sits inside the bounds.

## 7. The grid solver does not fully settle (open)

Found on 2026-11-13 when the perp funding premium made the hedge nearly free.

- **The stopping rule was wrong** and is fixed: it looked at the last
  policy-evaluation step, which goes to zero for any fixed policy. A solve
  could stop while decisions were still moving, and the BTC vault's answer
  then moved with the step budget (2e-3). It now stops on the change over a
  whole iteration.
- **The improvement step wobbled** and is fixed: it took each re-optimized
  decision even where it was slightly worse. A decision now changes only
  where a candidate beats it, so the value can only rise.
- **What remains is slow drift.** With both fixes the main loop settles
  cleanly, but repeated slow "polish" passes keep raising the value a little
  each round (about 2e-5 a round after 30 rounds on the test-sized integrated
  grid; 1.7e-3 in all). Probably decisions creeping toward a limit: the
  squashing functions approach a limit ever more slowly. Not confirmed which
  decision. The polish is capped at 3 rounds; values are good to about 2e-3
  and slightly low.
- **The BTC vault's unbiased solve stops short of its best.** A firm with a
  BTC bias of 4% finds a policy that, valued at the market's expectation,
  beats the unbiased firm's solved value by up to 5e-3 at some state, though
  both solves report tiny residuals and the unbiased answer does not move
  with the step budget. A policy the unbiased firm could have chosen is
  better than the one found: a local best, not the best. Not located;
  possibly a small unhedged BTC position near walk-away (design §3.4), which
  a start from almost no BTC cannot find. The test is marked as a known
  failure (`tests/research/test_btc_vault_solver.py`).
- **If taken up:** first find the drifting decision; if it is not one
  decision, how decisions at limits are represented needs rethinking — a new
  numerical method, to be discussed first. The learner would inherit the
  same problem, since it uses the same squashing.
- **Speed, measured roughly:** float32 on the CPU (`cpu-fast`) is nearly 5
  times faster than float64 on a finer grid (25 × 4 × 7 × 7) and agrees to
  about 3e-7 on one evaluation. The GPU (MPS) is not yet timed.

## 8. Working rules that matter here (from `CLAUDE.md` and the user)

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
  `Co-Authored-By:` naming the Claude model that did the work.
