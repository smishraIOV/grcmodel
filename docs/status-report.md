# GRC quantitative model — status report

*Written for a reader who has not seen this project before. Covers what it is,
what has been built, which approaches were tried and what happened to them, what
the model currently claims, and what is unresolved.*

---

## 1. What the project is

**GRC** is Governance, Risk and Compliance — the function inside a financial firm
that sets risk limits, runs controls, and keeps the firm on the right side of its
regulators. It is normally treated as a cost centre.

The thesis of this project is that GRC, done well, **increases firm value**, and
that this can be made quantitative rather than asserted.

The academic anchor is Froot and Stein: even a firm indifferent to risk in
principle will pay to manage it, if losing money in bad states stops it from
funding good investments. Risk management is then not insurance — it is
protection of the firm's capacity to keep operating.

The firm modelled is a **financial intermediary on crypto rails**: it takes
deposits, lends and invests them, and runs the technology those assets live on.
It does so on a real balance sheet: roughly **five units of assets for every
unit of shareholder capital**, the rest funded by deposits.

> **A note on units.** Money in this model has no denomination. Opening capital
> is 16 and every other quantity below — a book of 81, a programme costing 0.76,
> a franchise of 28 — is in the same arbitrary currency. Only ratios carry
> meaning, which is deliberate: the cost functions are built so that the answer
> cannot depend on whether the firm is denominated in dollars or cents, and
> every published output is a ratio, a rate or a threshold rather than a level.
> Bare numbers here are not percentages.

It carries the full set of banking risks. **Loan defaults, liquidity events and
depositor runs are central**, not background: credit is the largest single loss
channel at 62% of expected loss, and because assets are about five times
capital, a loss of one percent of assets is a loss of five percent of capital.
What the crypto rails add is a second family on top — operational incidents at
16%, cliff events such as a drained bridge at 13%, compliance penalties at 9%.
The claim is not that the novel risks displace the textbook ones; it is that a
firm of this shape faces both, and a GRC function has to be budgeted against
both.

The one textbook risk still named rather than mechanised is the **run**: the
deposits are there and can shrink, but they leave smoothly rather than suddenly.
§6 sets out what is missing.

Everything is a PyTorch model in `quant/`, about 4,700 lines, with 103 tests.

---

## 2. Where it started

The original model was **two periods**. The firm picked a GRC budget, a random
loss arrived, and it chose how much to invest, paying a convex penalty if it had
to raise money externally. Its headline:

> With external finance free, the firm spends 0.77 on GRC. Facing convex
> financing costs, the same firm spends 2.56. The gap is risk management that
> pays for itself only because capital is expensive to raise in bad states.

Real but limited, in three ways that turned out to matter. **The firm could not
die** — GRC changed how much money it lost, never whether it survived, which is
not the decision an executive faces. **Nothing accumulated** — GRC bought one
period of protection and vanished, where real control functions are built up and
decay slowly. And **the convex financing cost was a stand-in**, a smooth
mathematical penalty standing in for something the model never represented.

---

## 3. What the model is now

A firm that operates over many periods and can fail. Each period it chooses how
much to spend on GRC (split across credit, operational and compliance), how much
capital to deploy, how much profit to distribute, and whether to wind down.

And the world does the following to it:

| mechanism | what it represents |
|---|---|
| **GRC as capital stock** | Controls accumulate and depreciate (~25%/year). Spending today protects for years, which is what makes prevention worth anything. |
| **Deposit funding** | It funds its book with capital *plus a deposit base* about four times its equity. Deposits persist, pay interest whether or not the book earns, and shrink as its capital does. |
| **Funding constraint** | It cannot deploy more than it can fund, and the constraint bites with a lag: this quarter's losses shrink the deposit base, which binds *next* quarter's lending. |
| **Survival hazard** | It can fail three ways: running out of capital, an incident becoming public and depositors leaving, or losing its licence. GRC reduces the last two directly. |
| **Credit risk on the book** | Loan defaults scale with how much is lent, so expanding the balance sheet costs more risk. Reduced by credit GRC — better underwriting. |
| **Runs** | Depositors can leave suddenly. A run is likelier when operational controls are weak and when the quarter's losses have thinned the capital, so a bad quarter draws the run that makes it worse. |
| **Liquidity and fire sales** | Withdrawals come out of reserves at par and out of the unmatured book at a 35% discount. A firm that cannot pay even after liquidating everything has failed at something being solvent would not have prevented. |
| **Cliff events** | Rare, severe hits — a bridge drained — that leave it alive but badly impaired. GRC makes them rarer; it cannot make them smaller. |
| **Wind-down option** | It can stop deliberately and recover more than a disorderly failure would leave. |

The deposit base is what makes ordinary banking risk bite. Deposits are about
four times capital, so assets are about five times capital, and **a loss of one
percent of assets is a loss of five percent of capital.**
Until the liability side existed the model had no way to carry that — a loss on
an unlevered book costs capital its own size, not a multiple of it.

Default configuration: **quarterly decisions over two years**, equity of 16, a
book near 68, leverage about 4.9, and a **4.3%** annual probability of failure.
The firm holds **24% of its balance sheet in reserves** against a run, funded by
lending 15% less than it otherwise would.

---

## 4. Approaches considered, and what happened to them

This is the part worth reading. Most of what was learned came from things that
did not work.

### 4.1–4.3 Three things replaced, in brief

| tried | outcome | why |
|---|---|---|
| Convex financing cost as the friction | **Retired** | Once the firm could fail it bound on 1.9% of outcomes against 52% in the two-period model. It had been a *reduced form*: a two-period model must assume the curvature that makes risk matter, while a model where death destroys a going concern derives it. Over many periods it is also unbounded below — equity roughly squares once negative — so the problem was not well-posed without a failure barrier. |
| Firm fails when equity crosses zero | **Replaced by a smooth hazard** | A hard threshold cannot be optimised through: a failed firm is worth a fixed amount, so nothing about it responds to what the firm did. Measured, a solver from a poor guess converged to *worse* than a policy ignoring the state entirely. The economics agree — intermediaries fail because a loss becomes public and funding leaves, not on an accounting threshold. |
| Cliff events — first **declined**, then **reinstated** | **Built** | Declined because an event is either severe enough to kill (the hazard) or moderate (ordinary losses). Reversed on a better argument: that reasons about where a *loss* lives, not where a *decision* lives. The cliff is the case in between — alive, badly impaired, holding a real choice. **A model with only "moderate" and "fatal" has no state in which GRC's option value does anything.** Measurable: after a cliff, the share of paths that cannot fund the investment they want roughly doubles. |

### 4.4 Four ways to solve the model

The model has no closed-form answer, so it is solved numerically. Four methods
were built, and comparing them is how errors get caught.

| method | how it works | verdict |
|---|---|---|
| **Closed form** | An exact formula for a deliberately degenerate case | Kept as the exactness check |
| **Constant policy** | One fixed action for every period and state | Surprisingly hard to beat |
| **Backpropagation through the simulator** | The simulator is differentiable, so gradients are exact | **The main method** |
| **Grid value iteration** | Enumerate a coarse grid of states, work backwards | Independent referee |

**Truncated backpropagation with a critic** was also built and **rejected**. It
exists to fix instability over long horizons and there was no instability to fix:
it lost at every horizon tested, monotonically, and **never beat a constant
policy at any horizon while costing 2.3× the compute.** The learned estimate was
the problem — early in training it underestimates the value of continuing, which
is exactly the input the firm uses to decide whether to continue.

### 4.5 Credit risk that did not depend on the loan book

**The most serious error before the liability side.** Credit losses were drawn
independently of how much the firm had lent — expected loss was the same whether
it deployed 5 or 80. Expanding the balance sheet carried no additional risk,
which inverts the central decision a bank makes.

Fixed: credit loss is a rate on what is deployed. It forced the sequence within a
period to change — a bank sets its book from the capital it has and defaults
arrive on what it lent, so losses can no longer be computed before the lending
decision.

At the time the fix changed almost nothing, because the firm was constrained by
funding rather than by risk: it lent up to what it could fund either way. **That
changed with the next section.** Deposits took funding-constrained quarters from
71% to about 15%, so risk now prices the last stretch of the book rather than
funding rationing all of it — which is the regime in which making credit loss
depend on the book was supposed to matter.

### 4.6 The firm had no liabilities at all

**Found latest, and the largest structural gap.** The firm funded its book out
of its own capital plus a costless multiple of it, borrowed and repaid inside
the same period. It had no depositors — which deleted **leverage** (losses on an
unlevered book cost capital their own size, so loan defaults could not threaten
the firm), **the price of funding** ("how levered should we be" was not a
question the model could pose), and **the funding leaving** (a liability repaid
inside the period cannot run, so the hazard named *"depositors leave"* was a
label on a constant).

**Fixed:** deposits are a stock. They persist, pay interest whether or not the
book earns, and the base a firm's capital can carry falls as that capital does.

**The instructive part was what it broke.** Four parameters had been calibrated
against a book of 25 and none moved on their own when the book went to 80:

> The firm's *opportunity* stayed put while its funding capacity tripled, so it
> stopped wanting what it could now fund. Funding-constrained quarters fell from
> **71% to 0.0%** — the central Froot-Stein channel switching itself off — while
> every other diagnostic stayed in range and the whole test suite stayed green.

A model can lose its main mechanism and go on producing plausible numbers. That
is the argument for a named regime check per channel rather than one overall
verdict.

Three bugs came out of the change, each caught by a test failing for a reason
other than the one it was written for: reserve income computed from a capacity
that is infinite when the cap is off (infinite income, then NaN everywhere); an
optimiser whose lending control could not travel far enough in its step budget,
which converged to a book of 58 and read as the firm's *choice* — and handed
the state-feedback policy a spurious **7.4% advantage** over a baseline that had
simply run out of steps; and a production function computing a 2% residue as the
difference of two float32 numbers near 1.

**Still assumed rather than derived.** The run hazard does not consult the
deposit base — a firm funded four-to-one on demandable money faces the same
assumed run rate as one funded entirely by its owners. §6 has the rest.

### 4.7 The run that was asserted rather than modelled

The hazard channel named *"an incident becomes public and depositors leave"* was
an annual rate. It asserted three things in one parameter: that an incident
becomes public, that depositors leave, and that the firm therefore dies. There
were no depositors in the model to leave, so the middle claim had no
representation and the third followed from a number.

**Fixed:** depositors now leave for real. A run is a discrete event, likelier
when operational controls are weak and when the quarter's losses have thinned
the capital — drawn *after* losses, so a bad quarter draws the run that makes it
worse. Withdrawals come out of reserves at par and out of the unmatured book at
a 35% discount. A firm that cannot pay even after liquidating everything has
failed at something the solvency channel does not describe, and a fourth hazard
prices the share it could not meet. The old rate is retired, not kept alongside:
both together would charge the firm twice for one event.

**The result.** The same firm, before and after, each at its own steady state:

| | credit | operational | compliance | **total** | reserves |
|---|---|---|---|---|---|
| asserted rate | 0.020 | **0.147** | 0.030 | **0.197** | 9% |
| run modelled | 0.046 | **0.065** | 0.077 | **0.187** | 24% |

**The asserted rate was overstating operational GRC by about 2.3×** — the
channel that justified the largest line in the budget was the one asserting its
own conclusion. But the budget does not disappear; it *moves*: total spend falls
5% while credit and compliance each roughly double. And the firm starts holding
a real liquidity buffer, 9% of the balance sheet to 24%.

Sweeping how often runs happen separates the two defences. Introducing a run at
all makes the firm buy **both** — operational GRC from 0.002 to 0.065, reserves
from 10% to 24%. Past that it substitutes: at two runs a year reserves reach 38%
while operational GRC falls back. **Cash is the certain defence and controls are
the probabilistic one, so at the margin the cheaper certainty wins.** The useful
reading is the middle: a firm facing occasional runs should do both; one facing
frequent runs should hold capital and liquidity rather than buy its way out.

**One near-miss worth recording.** The first measurement of this said the budget
had collapsed from 0.191 to 0.043 — that retiring the hazard had destroyed the
case for GRC. It had not. The firm's opening stock of controls is a *fixed point
of the whole model*, and it had been solved while the retired channel was live,
so the firm was starting six times above the level it would now choose and spent
the horizon running it down. Re-solved, the fall is 5% rather than 78%. A stale
fixed point does not error; it produces a plausible wrong answer, and this one
would have been reported as a headline.

### 4.8 Does a reactive policy earn its place?

Is a policy that *reacts to circumstances* worth more than a fixed budget?

Measured properly — five random seeds, scored on data it was not trained on — the
answer at the default settings is **no**: −0.02% over two years and +0.12% over
eight, bands overlapping in both. A single run calling it better is reporting its
seed. That verdict survived the liability side unchanged, which is itself
informative: a second state variable was added for the learner to observe and it
still could not use it. At the long horizon the learner is six times noisier
across seeds than the fixed policy — its best seed wins by 2.6% and its worst
loses.

The reason is not that the method is weak. **The firm's state barely moves**, and
a well-chosen fixed action is near-optimal everywhere it goes. The fix is not a
better learner: raising the rate of cliff events — the one shock large enough to
move the state sharply — takes the advantage from +0.04% to **+9.32%**,
monotonically. **A reactive policy pays when the state makes large discrete jumps
worth responding to, not when it merely drifts.** (That sweep predates the
liability side; the verdict it supports was re-measured and holds, the figure
itself has not been.)

### 4.9 How often the firm decides

**Tried:** weekly decisions instead of quarterly, reasoning that a firm deciding
quarterly has nothing to react to because its next decision is three months away.

**Outcome:** no effect — +0.05% either way. The binding constraint is the one
above: a state that does not move gives a reactive policy nothing to work with,
however often it is consulted.

The exercise was still worth it. It forced every parameter to be expressed as a
rate per year rather than per period, and surfaced two real bugs that only a long
horizon could expose — including one where a probability calculation silently
underflowed to zero and produced a meaningless answer.

---

## 5. What the model currently says, and how much to believe it

No parameter here is calibrated to real data, so the model **refuses to quote a
budget** and inverts the question into claims a reader can argue with. That is
necessary but not sufficient: a threshold computed from invented parameters is
still computed from invented parameters.

The honest way to answer "why believe any of this" is evidence rather than
assurance. The model has been recalibrated **four times**, each for a defensible
reason, and each time every underlying number moved. What that did to the
conclusions:

| claim | survived four recalibrations? |
|---|---|
| Spend peaks at middling capitalisation | **yes** — shape held, location moved |
| GRC as a stock beats GRC as an expense | **yes** — sign held (+268%, +33%, +49%, +31%) |
| A cheaper failure buys a smaller programme | **yes** (−41%, −41%, −27%) |
| The firm is risk-averse near failure | **yes**, and strengthened |
| State feedback does not beat a fixed policy | **yes** |
| Hazard break-even, in basis points | no — 527 → 270 → 243 |
| Survival vs loss-reduction spend ratio | **reversed** — 55× → 9× → 1.6× |
| Cost of ignoring survival, as firm value | no — 0.3% → 4.9% → 2.1% → 0.5% |

**So: trust the shapes, not the levels.** Every claim about a *direction* has
survived changes that moved every number underneath it. No quoted level has.

**And one headline reversed outright.** "GRC is paid for almost entirely by
survival, hardly at all by loss reduction" was the sharpest claim this project
made. It is now roughly even. The two corrections that did it were both the
model becoming *more* honest — leverage, which makes ordinary losses far more
expensive; and retiring an asserted death rate that had been crediting controls
with preventing deaths never demonstrated. That reversal cuts both ways: it
shows the framework catching its own error, and it shows the record of a
confidently quoted level from this model.

[`calibration.md`](calibration.md) has the full table, sorts every parameter by
how calibratable it actually is, and names a data source for each. Most have
real external anchors that have simply never been used. The exception is the
effectiveness of GRC spending itself, which no public data can settle — which is
precisely why the outputs below are thresholds.

### The claims, at the current calibration

**The headline, with no uncalibrated effectiveness parameter in it:**

> A programme costing 0.66 a year, against a business worth 27.27 as a going
> concern, must cut the annual probability of failure by at least **243 basis
> points** to pay for itself.

Both inputs are things a board already has a view on.

**GRC is bought about equally by survival and by smaller losses.** A firm
budgeting only for expected loss reduction spends 40% less; the gap costs 0.5%
of firm value and 1.1 points of annual failure probability.

**Given a run, the firm buys liquidity as well as controls — and past a point,
instead of them.** Introducing runs takes reserves from 10% to 24% of the
balance sheet *and* operational GRC from 0.002 to 0.065. At two runs a year
reserves reach 38% while operational GRC falls back. Cash is the certain
defence, controls the probabilistic one.

**Spend peaks at middling capitalisation.** 0.166 a quarter at equity 16 and
0.126 at 8, against 0.018 at equity 4 and 0.003 at 64. A well-capitalised firm
faces too little risk to bother; a nearly-failed one is too far gone for a
programme to pull back.

**Accumulation matters more than the annual number.** Treating GRC as a stock
rather than an expense is worth **+31% of firm value** and takes two-year
survival from 74% to 92%. The firm buys *more* per period when it persists,
because a control that still works next period is worth much more than one that
does not.

**The less a failure would destroy, the less a programme to avoid it is worth.**
Spend falls 27% as recovery in failure rises from nothing to 80%.

**As an effectiveness threshold**, a compliance programme pays if a unit of
spend removes about 12% of exposure, where pure loss-reduction arithmetic
demands an impossible 992%. Credit's pair is 17% against 173% — the only one of
the three where the loss-reduction threshold is arguable rather than absurd.

---

## 6. What is unresolved

**A run damages the deposit base, but not the franchise.** This is the
assumption doing most of the work in §4.7's conclusion, and it should be
weakened before anyone acts on that conclusion. Deposits rebuild toward capacity
at a four-month half-life whether they left in a panic or drifted away. Real
deposit franchises do not come back that fast after a run — the reputational
damage outlasts the outflow by years. A slower rebuild would make runs costlier
in a way a liquidity buffer *cannot* offset, which is the most likely route to
operational controls being worth more than the model currently says. Until it is
built, read "cash beats controls" as conditional on runs being survivable events
rather than franchise-ending ones.

**The learner has become unstable.** At the eight-year horizon one seed of five
produced a firm worth 0.9 against a best of 32.7 — a failed solve, not variance.
The verdict on state feedback is unchanged and if anything firmer, but the
instability is new with the run channel and is not understood.

**One number still carries too much weight.** The model stops after two years
and values what is left with a constant standing in for "the business continues"
— **45% of total firm value**. Conditional on surviving, nothing the firm does
affects nearly half the objective, which inflates how important survival looks
relative to operations. The liability side neither fixed nor much worsened it
(44% before, 45% now): firm and franchise both grew. Nor can it be fixed by
solving for it — setting the constant self-consistently *diverges*, because a
larger franchise makes survival worth more, so the firm buys more GRC, so the
failure rate falls, so the franchise grows. Successive passes ran 15 → 40 → 58
with annual failure falling 5.5% → 3.1%: a model talking itself into being safe.

The agreed fix is to compute the continuation value from the model itself and
lengthen the horizon to ten years so it carries less weight; monthly rather than
weekly decisions keep that affordable. **That fix has a cost.** The natural way
to compute it uses the grid solver — currently the only check sharing nothing
with the main method. Feeding its output into the objective ends its
independence and the project loses its referee. Two replacements are open:
checking value is consistent across horizons, and a gradient-free search that
grades the policy on the full problem rather than a restricted one.

**Firm value is not comparable across horizons**, for the same reason: it
declines gently as the horizon lengthens, because a fixed terminal number is
discounted harder the longer it waits.

**The default horizon undervalues traditional risk management.** GRC is a stock
that takes time to build, so at two years a control bought to reduce *losses*
has not finished accumulating when the model stops, while one bought to reduce
the *failure rate* pays from the first period. On the pre-liability model credit
GRC spend rose fortyfold between a two-year and a ten-year horizon. That sweep
has **not** been re-run and should be: leverage already moved the
survival-versus-loss-reduction split toward loss reduction (§5) and a longer
horizon moves it further, so the two are confounded until it is.

**No parameter is calibrated.** Failure rates, loss severities and recoveries
have real external anchors — bank failure statistics, published exploit losses,
bankruptcy recoveries. The effectiveness of GRC spending does not, which is why
every output is a threshold rather than a recommendation.

---

## 7. Running it

```bash
uv run pytest                                  # 103 tests, about four minutes
uv run python scripts/run_dynamic_model.py     # the multi-period model
uv run python scripts/run_breakevens.py        # the decision-facing outputs
uv run python scripts/run_seed_study.py        # is a reactive policy worth it?
uv run python scripts/run_static_model.py      # the original two-period model
```

Documentation lives in `docs/`: `quant-model.md` is the specification and
results, `static-model-debug-notes.md` records the reasoning behind design
choices and the traps found along the way, `framework.md` covers the conceptual
structure.

**Code state:** everything through §4.5 is on `main`. The liability side of
§4.6, and the recalibration it forced, are on `liability-side`. The rejected
truncated-backpropagation experiment is on `svg-critic` and deliberately not on
`main`, so a reader chasing a number that moved does not have to rule out a
solver that lost at every horizon tested; its verdict stays in the
documentation.
