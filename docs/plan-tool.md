# Plan: a what-if tool for a CFO or risk manager

Written 2026-11-13, agreed with the user the same day. This is the project's
current plan. Read it first, then `CLAUDE.md` (working rules) and
`docs/design.md` (the specification). Where this plan and the design
disagree, the design wins; update this plan.

**Status:** steps 0 to 7, 9 and 10 (the dashboard) are done. Step 8 (calibration) remains and needs the user: data sources, or a real firm's figures. Several choices in the accounting and in step 5 were made by Claude while the user was away and are listed in section 3 for review. Each step needs the user's
approval as it comes.

---

## 1. The direction

The project now has two paths that share one core.

- **The tool (this plan).** A what-if engine: the user enters their firm and
  their own policy, and it shows what can happen. It does not search for the
  best policy, so nothing in it depends on a solver converging. It is meant
  to sit behind a dashboard a CFO or risk manager can play with.
- **Research.** Everything that searches for the best policy: the grid
  solvers and the actor-critic learner. It asks the more academic question —
  what the firm *should* do, and what GRC is worth to a firm that does it.
  Its plan is `docs/plan-integrated.md`.
- **The shared core** is the one-period accounting and the scenarios:
  `quant/params.py`, `core.py`, `firm.py`, `lending.py`, `btc_vault.py`,
  `integrated.py`, `quadrature.py`, `numerics.py`. One copy, tested once, used
  by both paths.

**Why the split.** The accounting is checked against hand-worked cases to
1e-14 and reduces exactly to each vault alone. The search for the best
policy is not in the same state: its answers sit at the limits of almost
every decision, the grid solver settles only to about 0.2% of the firm's
value (design §11), and nothing is calibrated, so time spent on its
convergence is not yet well spent.

**What the tool gives up.** Without the search there is no single "value of
the firm" and no "best" anything. The tool reports ranges of outcomes and
comparisons between policies the user chooses.

**One directory, not a branch.** The research code is under
`quant/research/` (step 0). Separate branches would let the shared core drift
apart.

## 2. Steps

Each step ends with tests passing and a commit.

| step | what | done when |
|---|---|---|
| 0 | **Separate the two paths in the layout.** The solvers (`grid.py`, the three `*_solver.py`, `lending_results.py`, `integrated_results.py`, `twoperiod.py`) are under `quant/research/`, their tests under `tests/research/`, the entry point at `scripts/research/integrated.py`. The solver tests are marked `slow` and left out of the default run. | **Done (2026-11-14).** `uv run pytest` runs in about two seconds; `tests/test_layout.py` fails if the core imports research code. |
| 1 | **A firm profile in real units** (`quant/profile.py`). The firm's facts, its market assumptions and its risk assumptions, in dollars and rates per year; one function maps a profile onto `Params`, the firm's state and the scale (the dollar vault's capacity = 1), another maps back. The cost of raising equity, GRC's effect and GRC today are entered as plain statements (design §4.1). Assumptions can be filled from the model's defaults, sized to the firm. | **Done (2026-11-14).** A profile round-trips; the model is unchanged when the same firm is entered in cents; a parameter added to the model must be placed in the profile or named as left out (`tests/test_profile.py`). |
| 2 | **Policy as an input** (`quant/policy.py`). The user's own choices, in dollars and yearly rates: equity kept (so the payout or raise), GRC spend, own BTC and hedge, borrower rate, LTV, the BTC vault's allocation. Fixed targets for one period; rules that depend on the firm's state come with step 5, which needs them (section 3). | **Done (2026-11-14).** A policy that breaks a constraint is refused with every reason in plain words and dollar amounts; the constraints are checked to be the accounting's own; a sound policy round-trips. |
| 3 | **The one-period read-out, without a solver** (`quant/readout.py`, `scripts/whatif.py`). Takes a profile and a policy: capital and what the constraints ask for, GRC and the chance of an incident, the loan book, what the firm expects to earn and spend, cost ratios, what each group of investors expects against what they could earn elsewhere and the vaults' expected sizes, and how the period may end (expected equity, the chance of ending with less, the chance and size of a shortfall below the capital needed). In dollars; deterministic. | **Done (2026-11-14).** Matches the research read-out at a solved firm's own decisions, annual and quarterly (`tests/research/test_readout_matches.py`); matches the accounting by hand where nothing is random; unchanged in cents; finer scenarios do not move it. The one figure that did not settle at first, the BTC vault's yield with a sleeve, led to the change of BTC's price model (design §3.4). |
| 4 | **Named stress scenarios** (`quant/stress.py`; `scripts/whatif.py --stress`). Single outcomes worked through the accounting under the user's policy: BTC moving smoothly, a crash, an incident in the lending system or in the BTC vault, and combinations. For each: the firm's equity and its change, the shortfall below the capital needed and the cost of raising it, bad debt, what the firm repays, the vaults' sizes, and what outside investors receive. A standard set of ten, with round-number sizes; any other can be given. | **Done (2026-11-14).** Each standard stress matches the accounting written out by hand, annual and quarterly; a smooth fall and a crash to the same price differ as design §2.2 says; an incident can be imposed on a profile that expects none. Two incidents in one period are refused: the model allows one. |
| 5 | **Multi-year paths** (`quant/paths.py`; `scripts/whatif.py --years 5`). The firm simulated forward on random paths under a rule: pay out above a buffer over the capital needed, raise back to it when below, fail if the raise needed exceeds what shareholders will put in; everything else fixed. Loan demand follows BTC's price along a path. Gives, by year: the range of equity, the chance of having raised or failed, what is paid out and put in, the vaults' sizes, and what shareholders take over the horizon. | **Done (2026-11-14).** One simulated period agrees with the read-out within sampling error; the draws match their distributions; with nothing random every path follows the rule by hand; the same seed repeats. **The choices were Claude's, made while the user was away: to review** (design §11, and open question 0 in §8). |
| 6 | **Two policies side by side** (`quant/compare.py`; `scripts/whatif.py --b-ltv 0.7 ...`). The same firm and scenarios under two policies, with the change from the first to the second: the policies themselves, one period ahead, each stress, and over several years on the same random paths. This is what takes the place of "the best policy". | **Done (2026-11-14).** Each side is that policy's own read-out, stresses and paths; a policy against itself changes nothing; the difference over several years is measured path by path, with its own sampling error. The one summary figure, what shareholders take, does not count risk, and the report says so. |
| 7 | **Assumptions shown as assumptions** (`quant/sensitivity.py`; `scripts/whatif.py --sensitivity`). Thirteen dials for what cannot be observed — how well GRC works, incidents, operating losses, liquidation losses, withdrawals, BTC's volatility and crashes, loan demand, investors' response to yield, the cost of raising after a bad period — each turned to the low and high end of a wide range, one at a time. For each output: the dials that move it, most first, and how many do not move it at all. Works on the change between two policies too. | **Done (2026-11-14).** Each dial moves what it names and gives a valid profile at both ends; outputs do not move with assumptions they cannot depend on; with GRC assumed to do nothing, more GRC spend prevents nothing and only costs. The ranges are Claude's and are not estimates: to review. |
| 8 | **Calibration.** A source for each block of parameters: BTC's volatility and crash history; perp funding rates; liquidations and bad debt at lending protocols; incident losses; fee levels and flows. Recorded with the source and the date. | The user agrees the numbers are worth showing. |
| 9 | **An interface for the dashboard** (`quant/api.py`; `scripts/api.py`). A request as plain data — the firm's facts, any assumptions and decisions to change, what is wanted — and an answer as plain data: the inputs with every default filled in, and the read-out, stresses, paths (as the points a chart needs), comparison and sensitivity. `inputs()` lists every input with the example firm's values, enough to lay out a form. | **Done (2026-11-14).** The answer is what the modules return and survives strict JSON; a firm's facts alone are a valid request; sixteen kinds of mistaken request are answered with reasons, none with an error. Each input now carries how to show it and a one-line description (added with step 10). |
| 10 | **A dashboard** (`dashboard/index.html`, served by `scripts/dashboard.py` on this machine; added 2026-11-14 at the user's request). A form for the policy, the horizon, the firm's facts and the assumptions, each with its one-line description; six headline figures; the stresses as bars against the capital needed; the paths as ranges over the years (equity, the two vaults, BTC's price) and the chances of having raised or failed; the next period as a table; a second policy beside the first with the changes; and, on request, what a chosen figure leans on. Light and dark. | **Done (2026-11-14).** The server answers as the interface does and listens on this machine only; the page was looked at in both themes, with and without a second policy and with the sensitivity drawn; it loads nothing from outside. Not done: saving and loading a firm; anything for a phone-sized screen beyond stacking; labels written for a reader, not derived from the inputs' names. |

Steps 0–4 use nothing new. Step 5 is the main new work.

## 3. To discuss before building

- **Choices made on 2026-11-14 while the user was away.** Each is
  in the design's decisions log with its reason.
  1. *Slippage*: a smooth fall sells at the threshold less 25% of the fall
     past it; GRC narrows it.
  2. *Withdrawals from the other vault* after an incident: 10% of it if the
     vault hit were the whole business, scaled by the hit vault's share.
  3. *Raising equity*: 20% all in to raise 5% of the dollar vault's capacity,
     50% to raise 20%; three times dearer after BTC halves, three times after
     an incident. The solvers use the first part only.
  4. *Loan demand along a path*: moves as BTC's price to the power 0.5.
  5. *The rule over several years*: a buffer above the capital needed; raise
     back to exactly what is needed; fail if that takes more than a largest
     raise, by default the capital needed today.
  6. *Random draws*, 20,000 paths, seeded; "what shareholders take" counts
     equity at the end at book.
- **The choices above were reviewed and accepted by the user** (2026-11-14).
- **For calibration (step 8): BTC's expected return.** On the paths BTC's
  median price roughly halves over five years while its average rises. The
  arithmetic is right: at 60% volatility the typical path runs about 18
  points a year below the average, and the average is set at 6%, the return
  shareholders require of everything. What is not believable is 6% for an
  asset that volatile. The fix to consider is a market return for BTC of its
  own, higher than shareholders' — which reopens how the firm's own BTC is
  valued (design §3.4), so it needs discussing first. The user chose to leave
  it for calibration.
- **Also for calibration:** the example firm's costs exceed its income, so it
  wears its buffer down and then raises regularly; GRC looks very effective
  at the illustrative settings; expected bad debt looks high for a lender.
- **Gaps in the accounting that matter more for a tool than for research,**
  for the user to rank, not to be fixed unasked:
  - a BTC price that touches the liquidation threshold and recovers is not
    counted as a liquidation (design §8, item 3);
  - investors respond to their average yield, not to risk;
  - loans last one period and the book restarts each period (demand now
    follows BTC along a path, the cheap part of this);
  - perp funding is constant and stablecoins never lose their peg;
  - at most one incident a period, and GRC changes how often, not how bad;
  - operating costs for the integrated firm are still the single-vault
    figure. A first exploratory pass put them at about twice that to sit
    inside the agreed 50–60% of combined revenue; with real profiles (step 1)
    the user enters them, and the ratio becomes a check, not a calibration.

## 4. Working rules that matter here (from `CLAUDE.md` and the user)

- Ask before long work; give estimates. Steps need approval.
- Raise any new concept, with its source, before it goes into code or docs.
- When the user asks a question, answer it; do not edit files.
- Plain language, US spelling, short replies.
- No model results in docs yet. Nothing is calibrated.
- Do not spend long on solver convergence while nothing is calibrated.
- Every function points to the design section it implements. Every
  entry-point script is imported by a test.
- Commit each finished step with a message ending in a `Co-Authored-By:`
  line naming the Claude model that did the work.
