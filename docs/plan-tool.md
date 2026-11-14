# Plan: a what-if tool for a CFO or risk manager

Written 2026-11-13, agreed with the user the same day. This is the project's
current plan. Read it first, then `CLAUDE.md` (working rules) and
`docs/design.md` (the specification). Where this plan and the design
disagree, the design wins; update this plan.

**Status:** steps 0 to 4 are done. Step 5 needs a discussion first (section 3). Each step needs the user's
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
