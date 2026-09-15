# Critical review — documentation and code

Review date: 2026-09-15. Reviewed at commit `53f29f9` on `main`, with
`svg-critic` at `30398aa` + 4.

> **Status: partly addressed.** Kept unedited below as the record of what was
> found and how, in the same way `docs/critical-review.md` was on the
> `svg-critic` branch. Every "current" statement describes `53f29f9`, not the
> code today. This block is the only thing that moves.
>
> **Landed.**
>
> | finding | commit |
> |---|---|
> | C1 — losses never reached the liquidity buffer | `6a2ffe4` |
> | C2 — the "expected loss only" comparison was not that | `fd0c3fa` |
> | C5 — one α, three different channel counts | `fc903ac` |
> | §5 — α read as a percentage ("an impossible 992%") | `fc903ac` |
> | C6 — the entry-point import test globbed only `run_*.py` | `f6c1b73` |
>
> **Open, needing no compute.** C3 (two definitions of annual failure), C4
> (`value_curvature` carries the confound `capitalization_band` fixes), C6a–c
> (the `DEFAULTS.alphas` leak, the grid's reward discount, the "barrier at 0"
> line), C6e (two parallel lineages, both advertised flat in the readme), D1–D8
> (the dangling and stale cross-references, and the two senses of "franchise"),
> and D9 — the history-free page, which is the largest single fix for the
> reader question in §5.
>
> **Open, needing compute.** The re-run of every liability-side table, now that
> C1 and C2 have moved them; the franchise sweep attached to the headline
> break-even (§7 item 3); and seeds on the remaining single-draw sweeps.
> Measured since: a full re-run is ~2.5h at current settings, ~7.2h across three
> seeds.
>
> **Deferred.** `svg-critic` (§6), which C1 and C2 have made slightly harder to
> port rather than easier.
>
> **Where this document was itself wrong or short.** C2 is described here as a
> *narrow* comparison; it was also a *biased* one — switching off the hazard
> rates handed the loss-only firm a safer world, so it took more balance-sheet
> risk, spent twice as much and survived less. C1's entry did not anticipate
> that the closed-form run boundary would need re-deriving, which turned out to
> be the most useful part of the fix.
>
> **Found while acting on this review, and not in it.** At 1024 paths and 2500
> steps the optimal GRC budget varies by **10% across scenario draws** — larger
> than several effects §6 reports from a single draw, and the sharpest available
> form of this document's own §4 warning. `GRC_STEADY_STATE[(12, 60)]` is
> ~4% stale (≈1.17 against 1.12), unresolved to better than ±0.04 and not yet
> pasted. And §6.5's prose puts credit's risk-neutral break-even at 1.462 where
> its own table says 1.725.

Scope: `readme.md`, `docs/framework.md`, `docs/quant-model.md`,
`docs/calibration.md`, the `quant/` package, `tests/`, `scripts/`, and the
`svg-critic` branch.

**What was executed.** The full suite — 130 passed in 4m30s, matching what
`readme.md` claims. Targeted probes of `StandardDynamics.step`, the hazard
loss-only comparison behind the survival-channel table, and the α → exposure
arithmetic.
The published studies were **not** re-run; a full `run_dynamic_model.py` is
order of an hour and nothing below depends on re-deriving a level. Findings
that required execution say so; the rest are readings of the code.

There is an earlier review, `docs/critical-review.md`, which survives only on
`svg-critic`. It describes commit `690c0ce` and is marked addressed. This file
is named differently so the two can coexist if that branch is ever merged.

---

## 1. Verdict

The project is unusually disciplined about the two things this kind of model
usually gets wrong — numerics provenance, and knowing which of its own results
are load-bearing. The solver ladder (closed form → grid → constant → neural →
perfect-information bound), the single `evaluate`, the sandwich asserted on
every run, and the profile-passed-never-looked-up rule are real defences, not
gestures. The traps in `CLAUDE.md` and the appendix are documented at the point
of use with measured numbers attached.

Two things do not hold up as well as the documents claim.

**The run channel — the newest and most load-bearing mechanism — has an
accounting hole.** Reserves are computed before the period's losses land, so a
loss never depletes the buffer the firm meets a run out of (§3, C1). The run's
*probability* is evaluated post-loss and its *meetability* is not, which is
precisely the correlation the stage exists to produce. The controls-versus-cash
finding is the one most exposed to it.

**The survival-channel headline measures a narrower comparison than its label.**
The "expected loss only" row removes exactly one hazard channel — licence — and
leaves the capital channel, the liquidity channel, and operational GRC's effect
on run probability intact (§3, C2). "GRC is bought about equally by survival and
by smaller losses" is a claim about that narrower comparison.

On the reader question: a new user can follow *what the model is for* in about
ten minutes and *what the model currently is* in about a day, and the gap
between those two numbers is the documentation's main cost (§5).

Nothing here is architectural. The dynamics factoring, the differentiability
engineering and the test suite would all survive every fix below.

---

## 2. What is working well

- **Numerics.** `quant/numerics.py` is the strongest module in the repo.
  Drawing randomness on CPU in float64 so a profile change alters arithmetic and
  never the scenario set turns a confound into a test; `accum_dtype` on every
  reduction; `production` written through `expm1` because the payoff is a 2%
  residue of two nearly-cancelling numbers; `fingerprint()` in every entry
  point's header. `get_profile` refusing to fall back silently is the right call.
- **Differentiability is engineered per mechanism and justified against the
  alternative.** The straight-through-versus-likelihood-ratio choice being made
  on *horizon* — bias that stays put versus variance that compounds — is a real
  argument, and §3's "Why the model is differentiable at all" is the best-written
  section in the documents.
- **The grid rung, and the warning attached to it.** Recognising that a
  restricted referee penalises a policy for using a control it cannot see, and
  saying so *before* the disagreement is read as a learner bug, is the kind of
  thing most projects discover after weeks.
- **`steady_state_firm` raising rather than guessing** (`quant/params.py:700`).
  Exactly right for a quantity whose stale value degrades into a plausible wrong
  answer.
- **`calibration.md` §3.** Sorting headlines by whether they survived four
  recalibrations is the most defensible thing the project can say about itself,
  and the admission that a confidently stated level has a 1-in-4 record is worth
  more than any single number in `quant-model.md`.

---

## 3. Findings in the code

Ordered by how much they move a published claim.

### C1 — Losses do not deplete reserves, so a bad quarter leaves the liquidity buffer intact

`quant/env/dynamics.py:590` passes `fundable` — equity *before* the period's
losses — to `reserves_held`. Everything downstream of it (`meet_withdrawal`,
`unmet_share`, the liquidity hazard, reserve income) is therefore computed on a
pre-loss buffer.

Executed: a path given a compliance loss of 40, driving equity to −26.1, reports
reserves of 17.919 — identical to three sibling paths that lost nothing.

```
loss       [40.04,  0.0099,  0.0007,  0.0861]
wealth     [-26.12, 13.91,   13.92,   13.83 ]
reserves   [ 17.92, 17.92,   17.92,   17.92 ]
```

The asymmetry is the problem. `run_probability` *is* evaluated on post-loss
equity, and `dynamics.py:394` calls that loop the whole mechanism: "losses thin
the capital, thin capital draws depositors out, meeting them forces a fire sale,
the fire sale thins the capital further". Half of it is implemented. The firm
becomes more likely to face a run after a bad quarter and no less able to pay
one, so `unmet_share` fires less often than the mechanism says, the liquidity
hazard is under-triggered, and the fire sale is under-priced.

Credit losses are arguably losses on the book rather than on cash and might
belong where they are; operational, compliance and cliff losses are cash
outflows and plainly are not. All four are currently netted out of equity only.

**What it touches.** The controls-versus-cash result, the run-rate sweep, the
reserve ratio, and `annual_liquidity_failure` — i.e. everything stage 6b and 6c
were built to produce. Direction is not obvious a priori: a smaller buffer makes
runs harder to meet, which should raise the value of operational controls, which
is the direction `calibration.md` §2 already names as the likeliest correction.

### C2 — The "expected loss only" budget still buys survival through two channels

`scripts/run_dynamic_model.py:321` builds the naive environment as

```python
hazard=replace(params.hazard, annual_operational_rate=0.0, annual_licence_rate=0.0)
```

`annual_operational_rate` is **already 0.0** on `main` — it was retired in favour
of the run mechanism. So it removes exactly one thing, the licence
hazard. What stays:

- the capital channel, which GRC reaches indirectly by leaving more equity behind
  (the documents say so, and count it as a survival channel);
- the liquidity channel;
- **operational GRC's effect on `run_probability`** (`dynamics.py:425`), which
  `hazard.py`'s own docstring calls the place "operational GRC still buys
  survival".

The row is labelled "budget set for expected loss only" and read in `readme.md`
as "a firm budgeting only for expected-loss reduction". It is not that firm. It
is a firm that cannot see the licence hazard.

The 1.62× ratio, the 38%-smaller programme, and "GRC is bought about equally by
survival and by smaller losses" are all statements about that narrower
comparison. Either widen it (zero the licence rate *and* make the loss-only firm's
`run_probability` GRC-independent *and* decide what to do about the capital
channel, which cannot be switched off without deleting the hazard) or relabel the
row. The honest label is probably "budget set as though compliance GRC did not
protect the licence".

Note this cuts *against* the current headline: a genuinely loss-only budget would
be smaller still, so the true ratio is above 1.62×, and the reversal recorded in
`calibration.md` §3 ("~55× → 1.6×") is somewhere between the two.

### C3 — Two definitions of "annual failure probability" in the published tables

`EvalResult.annual_death_probability` (`quant/env/env.py:568`) counts only mass
that operated and then failed; its docstring records why, and that counting a
tidy wind-down as a failure once reported "100% annual death". The script-local
`annual_death()` (`scripts/run_dynamic_model.py:47`) derives the same quantity
from `survival_rate`, which nets out voluntary exits and therefore counts them as
deaths.

`run_dynamic_model.py` prints the second in every table it produces;
`run_breakevens.py` prints the first. Benign at the defaults, where the exit rate
is 3.5e-5 — and not benign in the two places the exit rate is not zero: the
capitalization band at equity 4 (where `quant-model.md` §6 explicitly says one
seed had the firm winding down) and the recovery sweep. Those are the two tables
where a reader is invited to compare death probabilities across configurations.

### C4 — The convexity diagnostic carries the confound the band study fixed

`capitalization_band` (`quant/studies/breakeven.py:169-186`) scales
`initial_grc_stock` and the terminal franchise with equity, and records why: held
fixed, "a barely-capitalized firm reads as having everything to protect and the
band comes out monotone — reporting the constant rather than the capitalisation".

`value_curvature` (`:255`), thirty lines below, does neither. It replaces
`initial_equity` alone. At the sweep's low end (equity 2.0) the firm carries a
control stock sized for a firm eight times larger and a franchise of 20 against a
book of 2. Both make a thin firm look safer and more valuable than it is, and
both flatten the value function exactly where a convex region would appear.

`quant-model.md` §6.4 states "there is now no positive second difference at all"
as a finding about the model. It is a finding about the diagnostic least able to
see one.

### C5 — One α per family, but the families carry different numbers of channels

A unit of stock buys, at full strength and off the same α:

| family | channels driven by the stock |
|---|---|
| credit | loss mean (`dynamics.py:250`) |
| compliance | breach probability (`:284`) **and** licence intensity (`hazard.py:198`) |
| operational | loss severity (`:252`), cliff frequency (`:117`), **and** run frequency (`:425`) |

`quant-model.md` §6.5 tabulates the break-even α across families as one
comparable "effectiveness", and reads the compliance row as "this programme pays
provided you believe a unit of spend removes about 12% of exposure". A unit of
compliance spend in this model removes 12% of breach exposure *and* 12% of
licence-revocation intensity; a unit of operational spend removes 12% of three
different things at once.

This is the same shape of objection the project raised against the retired
operational hazard — "three claims in one parameter" — and reusing one α was a
deliberate choice made for a good reason (nothing can calibrate two). The
problem is not the choice, it is that the break-even table does not say the
families are buying different bundles, and then compares the numbers.

### C6 — Smaller code findings

- **`alpha_breakeven` leaks `DEFAULTS`.** `breakeven.py:239` builds the swept
  alphas from `replace(DEFAULTS.alphas, ...)` while the baseline two lines above
  uses `replace(params.alphas, ...)`. Inert today because every `ModelParams`
  carries the same default `GrcAlphas` — and it is the same seam the same file
  twice records closing in `solve()` and `exposures()`.
- **The grid discounts the reward one period differently from the rollout.**
  `gridvi.q_values:207` forms `reward + discount * (...)`; `path_values`
  (`state.py:210`) discounts the dividend by `discount**(step+1)`. Inert only
  because `ReducedSpec.payout = 0.0` makes the reward identically zero. It is a
  settable field.
- **`run_dynamic_model.py:243` prints "insolvency barrier at 0".** The config
  sets `equity_floor = -5.0 * initial_equity`, i.e. −80, and
  `EnvConfig.at_frequency`'s own comment says the barrier is deliberately far
  below zero and is no longer economic. A hardcoded provenance line contradicting
  the configuration it describes — the same failure the script has already fixed
  three times for its result lines.
- **The entry-point import test misses two entry points.**
  `tests/test_frequency.py:186` globs `scripts/run_*.py`, so
  `scripts/bench_profiles.py` — advertised in `readme.md` — is not covered, and
  neither is `scripts/compare_learners.py` on `svg-critic`. The test exists
  precisely because nothing else imports these files.
- **Two parallel lineages, similarly named.** `FrictionlessBenchmark`,
  `family_exposures` and `frictionless_benchmark` exist in both `quant/model.py`
  and `quant/solvers/analytic.py`; `quant/threshold.py` (static era, built on
  `quant.model` + the `ORACLE` parameters) and `quant/studies/breakeven.py`
  (current) are both break-even tools, and `readme.md` lists both in one flat run
  list without saying which is superseded. A new reader following
  `python -m quant.threshold` lands in the two-period model with no signal that
  they have.

---

## 4. Is the computational approach robust?

Within its own terms, yes, and more so than most work of this kind. The gradient
machinery is exact rather than sampled, every relaxation is named with its bias
direction, the absorbing-action trap is understood and initialised around, and
the gradient clip at norm 100 was chosen where two measured distributions
separate rather than as the first value that worked.

Three residual exposures, in order.

**The terminal constant is the most load-bearing number in the project, and the
headline is proportional to it.** `EnvConfig.at_frequency` installs
`PerpetuityValue(franchise=20.0)` against a firm value near 33 — about a third of
firm value, as `readme.md` concedes. The headline break-even is
`annual cost / (value × going_concern_share)`, and that denominator is roughly
34% the constant. So 354 bp is proportional to a Tier-3 parameter that
`calibration.md` describes as "a stand-in for everything past the horizon". The
documents name the problem and never connect it to the number it scales.

The comment justifying 20 (`env.py:165`) is written in pre-liability quarterly
units: "roughly the present value of the operating surplus of ~0.70 a quarter".
The firm it is now applied to decides monthly and funds a book near 77.

`franchise_breakeven` already sweeps this quantity at 0/3/8/15 and never at 20.
Running the hazard break-even across the same sweep would cost one extra solve
per row and would convert the headline from a level into the band the project's
own reporting rules ask for.

**Single-draw sweeps.** `run_seed_study.py` correctly reports five seeds with an
interval, and `calibration.md` §3 correctly demotes the capitalization argmax to
sampling noise on that evidence. Everything else still prints one draw: the
recovery sweep, the run-rate sweep, the payout comparison, the curvature check,
and the franchise break-even. The seed study established that a single draw at
this horizon can span a failed solve and a healthy one; that finding has not been
propagated to the tables it implicates.

**Optimizer budget.** `optimize_constant`'s step budget was checked once, for the
capitalization band, and found not to bind. `opening_scale` exists because it
*did* bind, silently, at the current balance-sheet scale — Adam moves a raw
parameter by at most `lr` per step. The check has not been repeated for the
sweeps that vary the balance sheet (the band moves opening equity 16×; the
recovery sweep does not).

What is *not* a worry: dtype and device. `tests/test_numerics.py` and the
CPU-float64 RNG rule mean a profile change is a test rather than a risk, and
sampling error dominates float32 rounding by the margin the readme quotes.

---

## 5. Will a new user follow the intent?

**The thesis lands immediately.** `readme.md`'s first three paragraphs — GRC as a
cost centre, Froot-Stein, protection of the capacity to operate rather than
insurance — are clear, correctly attributed, and correctly scoped. The units note
is exactly the right thing to put on page one. `framework.md` §3's explicit
rejection of "minimize ruin probability" and of a Sharpe target, with reasons, is
the moment a reader understands what kind of model this is.

**The current model does not land.** The obstruction is register, not clarity.
Almost every paragraph in `quant-model.md` §6 establishes a claim by narrating
the wrong claim it replaced, in the same tense, at the same length. That is the
right instinct — it is what makes `calibration.md` §3 possible — but the
cumulative effect is that the model is described in the negative space between
retracted findings. There is no page that states what the model is and what it
says without the history.

Concretely, a first reader hits these:

| where | problem |
|---|---|
| `readme.md:202` | points at `quant-model.md` **§9**; the document has eight sections and the SVG verdict is inside §6. `CLAUDE.md` has it right |
| `quant-model.md:1633` | refers to "§5.3, the MDP"; §5 has no subsections |
| `quant-model.md:362` | "Periods are **quarterly**" — the main line is monthly |
| `quant-model.md:334-357` | §4 "Objective" is still the two-period static problem, and says a run and a liquidity buffer "are stage 6b" — which §5's own table, twenty lines later, marks **built** |
| `quant-model.md:547` | the multi-period block opens "Eight quarters, 2048 paths, discount 0.9809/quarter, insolvency barrier at zero" and the next paragraph says "Monthly decisions over five years". Measured: 60 months, discount 0.99361, floor −80 |
| `quant-model.md:963-986` | a pre-recalibration table and a "+49% (23.16 → 34.55)" paragraph sit inline, two hundred lines after the same section's headline "+131%". The block-quote guard is three paragraphs upstream |
| `quant-model.md:753` vs `:777` | the same nominal configuration reads reserves 24.2% / book 68.2 / failure 4.29% in one table and 21.6% / 76.93 / 3.03% in the next. `readme.md` quotes 22% |
| everywhere | two distinct quantities are called "franchise": `PerpetuityValue.franchise` (the terminal constant, 20.0) and `HazardBreakeven.franchise` (going-concern firm value, 26.93). §6's "Franchise break-even" sweeps the first; the headline divides by the second |

**One substantive documentation error.** `readme.md:88` and `quant-model.md:1596`
read α as a percentage of exposure removed:

> a compliance programme pays if a unit of spend removes about 12% of exposure,
> where pure loss-reduction arithmetic demands an impossible 992%

Mitigation is `exp(-αg)`, so a unit of spend at α removes `1 - exp(-α)`:

```
alpha= 0.1215   removes  11.4%      (quoted as 12%)
alpha= 0.1684   removes  15.5%      (quoted as 17%)
alpha= 1.7250   removes  82.2%      (quoted as 173%)
alpha= 9.9240   removes 100.0%      (quoted as 992%)
```

The small-α rows coincide, which is why the sentence reads naturally. The large
ones do not, and 992% is not impossible — it is a programme that removes
essentially all of its exposure per unit of spend, which is implausible rather
than ill-defined. The rhetorical gap the headline rests on is an artifact of
reading a rate in 1/money as a percentage. The claim survives restatement
("...where expected-loss reduction alone would need a unit of spend to remove
essentially the entire exposure"), but not in its current form.

**What would fix the reader problem cheaply.** One page — call it `model.md` or a
new §0 — that states the firm, the state, the controls, the four hazard channels
and the objective as they are *today*, with no history and no retracted claims,
and links out to §6 for results and `calibration.md` for provenance. Every
existing document then keeps its notebook register without being the entry point.

---

## 6. The `svg-critic` branch

**It is not a parked experiment against this model.** The merge base is `30398aa`
and `main` carries 22 commits since, including the entire liability side, the run
channel, the recalibration it forced, the move from quarterly/two years to
monthly/five, and the gradient clip. The firm on that branch has no deposits, no
run, no fire sale, and funds a book near 25. `quant-model.md` correctly flags the
head-to-head table as pre-liability; what is not said is that re-testing is a
**port, not a rebase**.

The parking decision itself is sound and well argued. SVG lost monotonically in K
at four horizons; the K=1 collapse to `0.7 × 16` is a clean demonstration that a
critic deepens the absorbing-exit trap rather than damping it; and the
instability that would have been the strongest case for a critic was diagnosed as
a single step off a cliff, which truncation does not address. Keeping it off
`main` so a reader chasing a moved number has one fewer solver to rule out is the
right call and is recorded in the right place.

Three things worth knowing before anyone acts on the one live reason to revisit
(`CLAUDE.md`: tested only to 32 steps, default horizon now 60):

1. **`windowed_objective` and `train_svg` carry no gradient clip.** The clip
   exists because of the run channel's straight-through estimator — which is one
   of the two mechanisms the branch does not have. Porting SVG forward imports
   the amplifier without the guard.
2. **`FirmState.detach` on the branch has no `deposits` field.** Truncation
   breaks the window on `state.detach()`, so this is the exact seam that must
   carry the new state variable. It is a two-line fix and an easy one to miss.
3. **`returns_to_go` still matches `Trajectory.path_values` term for term** —
   including the abandon/survive decomposition and the `beta^(step+1)` on
   dividends and recoveries. That is the one thing that has not drifted, and it
   is the part that would have been hardest to re-derive.

Separately: the branch is the **only surviving copy** of `docs/critical-review.md`,
`docs/static-model-debug-notes.md` and `docs/status-report.md`, all deleted from
`main` in `53f29f9`. That is currently load-bearing by accident. If the branch is
ever rebased, squashed or deleted, three retired documents go with it. Worth an
explicit decision rather than an implicit one.

---

## 7. Recommendations, ranked

1. **Fix C1** — charge the period's losses against reserves before a run is met —
   and re-run the controls-versus-cash sweep. It is the newest mechanism, it is
   the finding most exposed to it, and the correction pushes in the direction
   `calibration.md` §2 already names as most likely.
2. **Fix or relabel C2.** Either widen the comparison or stop calling it
   "expected loss only". The headline that rests on it is one of the project's
   four load-bearing claims.
3. **Attach a franchise sweep to the 354 bp headline.** `franchise_breakeven`
   already runs the sweep; it just never runs it at 20 and never reports the
   break-even across it. This converts the project's flagship number from a level
   into a band, which is what its own reporting rules ask for.
4. **Write one history-free page describing the current model** and point the
   readme at it. Then fix the six stale cross-references and the two internal
   number conflicts in §5's table.
5. **Restate the α-as-percentage sentence** in `readme.md` and `quant-model.md`
   §6.5, and add one line to the break-even table saying how many channels each
   family's α drives (C5).
6. **Put seeds on the remaining single-draw sweeps** — recovery, run rate, payout,
   curvature — or mark each one "one draw" the way the capitalization band now is.
7. **Small, cheap:** C3 (one definition of annual failure), C4 (scale the stock
   and franchise in `value_curvature`), C6 (the `DEFAULTS` leak, the grid's reward
   discount, the "barrier at 0" line, the `run_*.py` glob).
8. **Decide `svg-critic` explicitly** — either port it forward with the two fixes
   above, or archive it with a note that it carries the only copies of three
   retired documents.

None of these are architecture. The parts that would be expensive to get wrong —
the environment seam, the numerics profile discipline, the differentiability
choices, the solver ladder — are right.
