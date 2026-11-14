# Plan: a what-if tool for a CFO or risk manager

Written 2026-11-13, agreed with the user the same day. This is the project's
current plan. Read it first, then `CLAUDE.md` (working rules) and
`docs/design.md` (the specification). Where this plan and the design
disagree, the design wins; update this plan.

**Status:** agreed, not started. Each step needs the user's approval as it
comes.

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

**One directory, not a branch.** The research code moves under
`quant/research/` (step 0). Separate branches would let the shared core drift
apart.

## 2. Steps

Each step ends with tests passing and a commit.

| step | what | done when |
|---|---|---|
| 0 | **Separate the two paths in the layout.** Move the solvers (`grid.py`, the three `*_solver.py`, `lending_results.py`, `integrated_results.py`, `twoperiod.py`) under `quant/research/`, with their tests and `scripts/integrated.py`. Mark the slow solver tests so the default `uv run pytest` runs the core in seconds. | The core's tests run without importing anything from `quant/research/`. |
| 1 | **A firm profile in real units.** Vault sizes in dollars, fee schedule, payroll and other operating costs, treasury holdings, loan terms, the firm's assumptions about BTC. One function maps it onto `Params` and the model's scale (the dollar vault's capacity = 1), and back for reporting. | A profile round-trips; every output can be shown in dollars. |
| 2 | **Policy as an input.** The user's own choices: equity kept, payout and raising, own BTC and hedge, borrower rate, LTV, the BTC vault's allocation, GRC spend. Fixed targets first; simple rules later (pay out above a level, raise below one). | A policy that breaks a constraint (capital floor, runway, hedge above BTC held) is refused with a plain message. |
| 3 | **The one-period read-out, without a solver.** Today's read-out (`integrated_results.py`) reads a solved firm. Rebuild it to take a profile and a policy: what the firm expects to earn and spend, what each group of investors expects against what they could earn elsewhere, cost ratios, and how the period may end (chance and size of a shortfall below the capital floor). | Matches the existing read-out when given a solved firm's own decisions. |
| 4 | **Named stress scenarios.** Single outcomes worked through the accounting: BTC falls by a given share (smoothly, or as a crash), an incident in the lending system, one in the BTC vault, combinations. Shows the firm's equity, each vault's size and investors' losses after each. | Each scenario matches the accounting by hand. |
| 5 | **Multi-year paths.** The firm simulated forward under the chosen policy: the range of its equity, the chance and size of capital raises, runway, the vaults' sizes, investors' yields. **A new concept — to be discussed before it is built** (section 3). | To be set when the concept is agreed. |
| 6 | **Two policies side by side.** The same profile, scenarios and paths under two policies, with the differences. This is what takes the place of "the best policy". | — |
| 7 | **Assumptions shown as assumptions.** What cannot be observed — above all how well GRC spend works — is a dial with a range, and every output that depends on it says by how much. | — |
| 8 | **Calibration.** A source for each block of parameters: BTC's volatility and crash history; perp funding rates; liquidations and bad debt at lending protocols; incident losses; fee levels and flows. Recorded with the source and the date. | The user agrees the numbers are worth showing. |
| 9 | **An interface for the dashboard.** Profile and policy in, structured results out (JSON), stable and documented. The dashboard is built on it afterwards. | — |

Steps 0–4 use nothing new. Step 5 is the main new work.

## 3. To discuss before building

- **Simulating a fixed policy forward (step 5).** Monte Carlo simulation,
  standard. Questions: random draws or the quadrature scenarios chained from
  period to period; how many paths; how to report rare events (deep crashes)
  that few paths contain; what the firm does on a path when it falls below
  its floor (raise at the issuance cost, as in the model, or stop).
- **Policy rules (step 2).** Which simple rules are worth offering, and
  whether a rule may depend on the firm's state.
- **Gaps in the accounting that matter more for a tool than for research,**
  for the user to rank, not to be fixed unasked:
  - a BTC price that touches the liquidation threshold and recovers is not
    counted as a liquidation (design §8, item 3);
  - investors respond to their average yield, not to risk;
  - withdrawals after an incident come from the vault hit only;
  - loan demand does not move with the BTC price;
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
