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
firm of this shape faces both, and a GRC **programme** has to be budgeted
against both.

*(Function or programme? The model has only one of these. It represents the
**programme** — the spend, and the stock of control capability that spend builds
up. Whether that capability sits in a standing GRC function, in the business
lines, or in outside counsel is a question the model cannot see and does not
need to.)*

All three textbook risks are now mechanised, including the run: deposits can
leave suddenly, the firm meets the withdrawal out of reserves or by selling its
book at a discount, and it fails if it cannot. What is *not* modelled is the
lasting damage a run does to a deposit franchise — see §6, which is the
assumption most worth arguing with.

Everything is a PyTorch model in `quant/`, about 4,700 lines, with 103 tests.

---

## 2. Where it started

The original model was **two periods**, which is Froot, Scharfstein and Stein's
own structure rather than an arbitrary choice: their argument needs exactly one
period in which risk is borne and one in which the investment opportunity
arrives, so that a bad draw in the first can be shown to cost real investment in
the second. Two periods is the smallest model in which risk management can add
value at all.

So: the firm picked a GRC budget; then a draw arrived across three risk families
— a continuous credit loss, an operational incident times a severity, and a rare
compliance breach — reducing its internal wealth; then it chose how much to
invest, paying a convex penalty on anything it had to raise externally. Its
headline:

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

### The balance sheet

It funds a book with its own capital **plus a deposit base about four times that
capital**. Deposits persist between periods, pay interest whether or not the book
earns, and the base its capital can carry shrinks as that capital does.

**It cannot deploy more than it funds.** The constraint needs no free parameter,
because the balance sheet states it: capacity is *equity, minus what it just
spent on GRC, plus the deposits it holds*. Whatever it does not lend sits as
reserves, earning less than the book would and still costing deposit interest.

The constraint bites with a **lag**, which is the interesting part: this period's
losses shrink the deposit base, and that binds *next* period's lending.

Leverage is what makes ordinary banking risk bite. Assets are about five times
capital, so **a loss of one percent of assets is a loss of five percent of
capital.** Until the liability side existed the model could not carry that — a
loss on an unlevered book costs capital its own size, not a multiple of it.

### What can go wrong

| mechanism | what it represents |
|---|---|
| **Credit losses** | Loan defaults, as a rate on what is lent — so expanding the balance sheet costs more risk. Reduced by credit GRC: better underwriting. |
| **Operational incidents** | Failures in process and systems. GRC contains them: it reduces the severity, not the frequency. |
| **Compliance breaches** | Penalties and enforcement. GRC prevents them: it reduces the frequency, not the severity. |
| **Cliff events** | Rare, severe hits — a bridge drained. GRC makes them rarer; it cannot make them smaller. |
| **Runs** | Depositors leave suddenly. A run is likelier when operational controls are weak *and* when the period's losses have thinned the capital — so a bad period draws the run that makes it worse. |
| **Fire sales** | Withdrawals come out of reserves at par and out of the unmatured book at a 35% discount. That discount is the only loss in the model caused by the *timing* of an obligation rather than by an asset going wrong. |

### How it actually dies

Three channels, and their intensities add:

| channel | what it is | GRC acting on it |
|---|---|---|
| **Capital** | Equity thins toward insolvency | indirectly — GRC leaves more equity behind |
| **Liquidity** | It was asked for deposits it could not pay, even after liquidating | indirectly — operational GRC makes the run rarer |
| **Licence** | A breach escalates to revocation | **directly** — compliance GRC |

Solvency and liquidity are deliberately *different* failures: a firm can be
solvent and unable to pay, and that is the case a reserve buffer exists for.

A fourth channel — "an incident becomes public and depositors leave" — was
retired. It asserted three things in one parameter, with no depositors in the
model to leave. The run channel above now carries that mechanism honestly, and
§5 reports what it cost to find out.

Against all of this the firm holds two defences, and it uses both: the GRC
stock, which accumulates and depreciates at **25% a year** (so spending today
still protects in two years' time, which is what makes prevention worth
anything), and **reserves**. It can also stop deliberately, recovering more than
a disorderly failure would leave.

Default configuration: **quarterly decisions over two years**, equity of 16, a
book near 68, leverage about 4.9, and a **4.3%** annual probability of failure.
The firm holds **24% of its balance sheet in reserves** against a run, funded by
lending 15% less than it otherwise would.

---

## 4. How the model is solved

The model has no closed-form answer, so it is solved numerically. Four methods
were built, and comparing them is how errors get caught — three agreeing solvers
on a broken simulator agree wrongly, so the agreement is only worth what the
independence behind it is worth.

| method | how it works | what it is for |
|---|---|---|
| **Closed form** | An exact formula for a deliberately degenerate case | The exactness check. Holds to machine precision, so it catches what a convergence tolerance would hide |
| **Constant policy** | One action, held fixed across every period and state — but the action itself is **optimised**, by gradient ascent, not assumed | The floor a learner must clear. Surprisingly hard to beat |
| **Backpropagation through the simulator** | The simulator is differentiable end to end, so gradients are exact rather than sampled | **The main method** |
| **Grid value iteration** | Enumerate a coarse grid of states, work backwards from the horizon | The independent referee — it shares only the dynamics, no gradients and no optimiser |

**Truncated backpropagation with a critic** is built and **parked**, on the
`svg-critic` branch. It cuts the gradient chain every few steps and replaces the
rest with a learned estimate, which is the standard remedy for instability over
long horizons. Tested at horizons up to 32 steps it lost at every one, because
there was no instability to fix: the learned estimate underestimates the value of
continuing early in training, which is exactly the input the firm uses to decide
whether to continue.

It is kept rather than discarded, for two reasons that have both strengthened
since. The horizon is going to roughly triple, and backpropagation cost is linear
in steps, so the problem the technique exists for is the one now approaching. And
the learner *has* become unstable — see below.

### Does a reactive policy earn its place?

Is a policy that reacts to circumstances worth more than a fixed budget?

Measured across five seeds and scored out of sample, the answer is **no**:
−0.50% over two years and +4.84% over eight, both inside the spread. Read the
range rather than the median — at the long horizon one seed produced a firm worth
**0.9 against a best of 32.7**. That is a failed solve, not variance, and it is
new with the run channel.

The reason is not that the method is weak: **the firm's state barely moves**, so
a well-chosen fixed action is near-optimal everywhere it goes. The fix is not a
better learner either. Raising the rate of cliff events — the one shock large
enough to move the state sharply — takes the advantage to **+9.32%**,
monotonically. **A reactive policy pays when the state makes large discrete jumps
worth responding to, not when it merely drifts.** (That sweep predates the
liability side; the verdict holds, the figure has not been re-measured.)

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
defence, controls the probabilistic one, so at the margin the cheaper certainty
wins. The useful reading is the middle: a firm facing occasional runs should do
both; one facing frequent runs should hold capital and liquidity rather than buy
its way out with controls.

**An assumed failure rate was overstating operational controls by about 2.3×.**
The hazard channel named *"an incident becomes public and depositors leave"* was
a single parameter asserting three things at once, with no depositors in the
model to leave. Replacing it with a mechanism the firm can respond to cut the
operational line of the budget from 0.147 to 0.065. The budget did not
disappear, though — it *moved*: total spend fell 5% while credit and compliance
each roughly doubled. **A channel that asserts its own conclusion will be paid
for out of the budget of the channels that do not.**

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
assumption doing most of the work in the controls-versus-cash finding (§5), and
it should be weakened before anyone acts on it. Deposits rebuild toward capacity
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

**No parameter is calibrated**, and only one of them is genuinely
uncalibratable. Failure rates, loss severities, recoveries, leverage, charge-off
rates and withdrawal fractions in a run all have real external anchors that have
never been used. The effectiveness of GRC spending has none, which is why every
output is a threshold rather than a recommendation.
[`calibration.md`](calibration.md) sorts every parameter by which case it is in,
names a source for each, and records which conclusions have survived four
recalibrations — that last table being the only direct evidence available about
what to believe here.

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

**Code state:** everything described here is on `main`. The parked
truncated-backpropagation experiment is on `svg-critic` and deliberately not on
`main`, so a reader chasing a number that moved does not have to rule out a
solver that has never yet earned its place. Its verdict, and the argument for
keeping it, are in §4.
