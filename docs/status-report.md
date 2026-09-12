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

The firm being modelled is a **financial intermediary on crypto rails**: it takes
deposits, lends and invests them, and runs the technology those assets live on.
It now does so on a real balance sheet — about 80 of assets funded by 16 of
capital and 64 of deposits.

It carries the full set of banking risks. **Loan defaults, liquidity events and
depositor runs are central**, not background. Credit losses are the largest
single channel at **62%** of expected loss, and because the book is funded
four-to-one against capital, they land on equity at four times their size on
assets. What the crypto rails add is a second family on top: bridge exploits,
depegs, and the loss of a licence to operate — together the remaining 38%. The
claim is not that the novel risks displace the textbook ones; it is that a firm
of this shape faces both, and that a GRC function has to be budgeted against
both.

| loss channel | share of expected loss per quarter |
|---|---|
| Credit — loan defaults, as a rate on what is lent | **62%** |
| Operational — incidents, custody failures | 16% |
| Cliff events — a bridge drained, a depeg | 13% |
| Compliance — penalties, enforcement | 9% |

The one textbook risk still named rather than mechanised is the **run**. The
deposits are there and can shrink, but they leave smoothly rather than suddenly,
and there is no liquidity buffer or fire-sale cost to make a run expensive. §6
sets out what is missing.

Everything is a PyTorch model in `quant/`, about 4,700 lines, with 103 tests.

---

## 2. Where it started

The original model was **two periods**. The firm picked a GRC budget, a random
loss arrived, and it then chose how much to invest, paying a convex penalty if it
had to raise money externally. Its headline result:

> With external finance free, the firm spends 0.77 on GRC. Facing convex
> financing costs, the same firm spends 2.56. The gap is risk management that
> pays for itself only because capital is expensive to raise in bad states.

That result was real but limited, in three ways that turned out to matter.

**The firm could not die.** GRC changed how much money it lost, never whether it
survived — which is not the decision an executive faces.

**Nothing accumulated.** GRC was an expense that bought one period of protection
and then vanished. Real control functions are built up and decay slowly.

**The convex financing cost was a stand-in.** A smooth mathematical penalty
standing in for something the model never represented.

---

## 3. What the model is now

A firm that operates over many periods and can fail. Each period it chooses:

- how much to spend on GRC, split across credit, operational and compliance;
- how much capital to deploy;
- how much of its profit to distribute;
- whether to wind down.

And the world does the following to it:

| mechanism | what it represents |
|---|---|
| **GRC as capital stock** | Controls accumulate and depreciate (~25%/year). Spending today protects for years, which is what makes prevention worth anything. |
| **Survival hazard** | The firm can fail three ways: running out of capital, an incident becoming public and depositors leaving, or losing its licence. GRC reduces the last two directly. |
| **Deposit funding** | It funds its book with capital *plus a deposit base* roughly four times its equity. Deposits persist between periods, pay interest whether or not the book earns, and shrink as its capital does. |
| **Funding constraint** | It cannot deploy more than it can fund. The constraint bites with a lag: this quarter's losses shrink the deposit base, which binds *next* quarter's lending. |
| **Cliff events** | Rare, severe hits — a bridge drained — that leave it alive but badly impaired. GRC makes them rarer; it cannot make them smaller. |
| **Wind-down option** | It can stop deliberately and recover more than a disorderly failure would leave. |
| **Credit risk on the book** | Loan defaults scale with how much is lent, so expanding the balance sheet costs more risk. Reduced by credit GRC — better underwriting. |

The deposit base is what makes ordinary banking risk bite. **A one percent loss
on a book funded four-to-one against capital is a five percent loss of capital.**
That is the textbook route by which loan defaults take an intermediary down, and
until the liability side existed the model had no way to carry it — a loss on an
unlevered book costs capital the same amount, not a multiple of it.

Default configuration: **quarterly decisions over two years**, a firm with equity
of 16 running a book near 81, levered 4.9 times, earning **23% a year** on that
equity and facing a **4.8%** annual probability of failure. Funding binds in
about 15% of quarters, against 71% before deposits existed — a bank with a
deposit franchise is less often capital-rationed than one financing itself out
of retained earnings, which is close to the whole reason to be one.

---

## 4. Approaches considered, and what happened to them

This is the part worth reading. Most of what was learned came from things that
did not work.

### 4.1 The convex financing cost, replaced by survival

**Tried:** keeping the smooth convex penalty as the friction.
**Outcome:** retired.

Once the firm could fail, the penalty stopped doing anything — it bound on 1.9%
of outcomes, against 52% in the two-period model. It had been a *reduced form* of
something: a two-period model has to assume the curvature that makes risk matter,
while a model where death destroys a going concern derives it. Once survival was
explicit, the stand-in was redundant.

It was also dangerous. Over many periods it is unbounded below — equity roughly
squares each period once it goes negative — so the multi-period problem was not
well-posed without a failure barrier.

### 4.2 Hard failure threshold, replaced by a smooth hazard

**Tried:** the firm fails when equity crosses zero.
**Outcome:** replaced by a continuous failure *rate* that rises as capital thins.

A hard threshold cannot be optimised through. A failed firm is worth a fixed
amount, so nothing about it responds to what the firm did — the optimiser
receives no signal about how failure might have been avoided. Measured, a solver
starting from a poor guess converged to *worse* than a policy ignoring the state
entirely.

A hazard fixes the economics too. Crypto intermediaries do not fail on an
accounting threshold; they fail because a loss becomes public and funding leaves.

### 4.3 Cliff events — declined, then reinstated

**First decision:** not to model them. An event is either severe enough to kill
(already covered by the hazard) or moderate (already covered by ordinary losses),
so a third category earns nothing.

**Reversed**, on a better argument: that reasons about where a *loss* lives, not
where a *decision* lives. A moderate hit and outright failure both leave
management with almost no choices. The cliff is the case in between — the firm
survives, badly impaired, holding a real decision about whether to rebuild, run
down, or stop. **A model with only "moderate" and "fatal" has no state in which
GRC's option value does anything**, because in both outcomes the choice has
already been made for you.

That turned out to be measurable: in the quarters after a cliff, the share of
paths that cannot fund the investment they want roughly doubles.

### 4.4 Four ways to solve the model

The model has no closed-form answer, so it is solved numerically. Four methods
were built, and comparing them is how errors get caught.

| method | how it works | verdict |
|---|---|---|
| **Closed form** | An exact formula for a deliberately degenerate case | Kept as the exactness check |
| **Constant policy** | One fixed action for every period and state | Surprisingly hard to beat |
| **Backpropagation through the simulator** | The simulator is differentiable, so gradients are exact | **The main method** |
| **Grid value iteration** | Enumerate a coarse grid of states, work backwards | Independent referee |

**Truncated backpropagation with a critic** was also built and **rejected**. The
idea is standard in reinforcement learning: instead of differentiating the whole
episode, cut the chain every few steps and replace the rest with a learned
estimate. It exists to fix instability over long horizons.

There was no instability to fix. It lost at every horizon tested, monotonically —
and at the longest horizon both variants collapsed entirely while plain
backpropagation was fine. The learned estimate made things worse: early in
training it underestimates the value of continuing, which is exactly the input
the firm uses to decide whether to continue, and once it has decided to stop
there is nothing left to learn from. **It never beat a constant policy at any
horizon, while costing 2.3× the compute.**

### 4.5 Credit risk that did not depend on the loan book

**Found late, and the most serious error in the model.** Credit losses were
drawn independently of how much the firm had lent — expected loss was the same
whether it deployed 5 or 80. Expanding the balance sheet carried no additional
risk, which inverts the central decision a bank makes.

Fixed: credit loss is now a rate on what is deployed. The level is unchanged at
the firm's current book, so only its *dependence* on the book is new.

Two things about the fix are worth recording. It forced the sequence within a
period to change — a bank sets its book from the capital it has and defaults
arrive on what it lent, so losses can no longer be computed before the lending
decision. And the effect on the firm's behaviour is small, because **the firm is
constrained by funding rather than by risk**: it lends up to what it can fund
either way, so the new marginal cost of risk never binds on the size of the
book. The structural error is fixed; whether it changes anything depends on a
calibration where funding is not the binding constraint.

**That calibration arrived with the next change.** Giving the firm deposits took
the share of funding-constrained quarters from 71% to about 15%, so risk now
prices the last stretch of the book rather than funding rationing all of it —
which is the regime in which making credit loss depend on the book was supposed
to matter.

### 4.6 The firm had no liabilities at all

**Found latest, and the largest structural gap in the model.** The firm funded
its book out of its own capital plus a costless multiple of it, borrowed and
repaid inside the same period. It had no depositors. That deleted three of the
risks a bank exists to manage:

- **Leverage.** Losses on an unlevered book cost capital the same amount rather
  than a multiple of it, so loan defaults were present but could not threaten
  the firm.
- **The price of funding.** Borrowing was free, so there was no reason not to
  take all of it, and "how levered should we be" was not a question the model
  could pose.
- **The funding leaving.** A liability repaid inside the period cannot run. The
  hazard channel named *"an incident becomes public and depositors leave"* was a
  label on a constant, because the model contained no depositors to leave.

**Fixed:** deposits are now a stock. They persist between periods, pay interest
on the base outstanding whether or not the book earns, and the base a firm's
capital can carry falls as that capital does — on both terms at once, since the
leverage limit multiplies a smaller number and market access is closing at the
same time. Funding the firm has not lent earns a lower rate, so surplus deposits
are parked rather than burned.

**What it cost, and the part worth reading.** Four parameters had been
calibrated against a book of 25 and none of them moved on their own when the
book went to 80. The instructive one:

> The firm's *opportunity* stayed where it was while its funding capacity more
> than tripled, so it simply stopped wanting what it could now fund. The share
> of quarters in which funding bound fell from **71% to 0.0%** — the central
> Froot-Stein channel switching itself off — while every other diagnostic stayed
> in range and the whole test suite stayed green.

Nothing failed. A model can lose its main mechanism and go on producing
plausible numbers, which is the argument for having a named regime check per
channel rather than one overall verdict.

Three bugs came out of the change, and each was caught by a test failing for a
reason other than the one it was written for:

- Reserve income was computed from the funding *capacity*, which is infinite
  when the cap is switched off. Infinite reserves, infinite income, and a NaN
  through every path — caught by a test about a different friction in a
  different file.
- The optimiser started the firm's lending control near zero and could only move
  it a fixed amount per step, so reaching a book near 80 consumed most of the
  budget before the search began. A sweep converged to a book of 58 and read as
  the firm's *choice*. Worse, it handed the state-feedback policy a spurious
  **7.4% advantage** over a baseline that had simply run out of steps — which the
  seed study in §4.7 would have reported as a finding.
- The production function computed a 2% residue as the difference of two float32
  numbers near 1. Harmless at the old scale and not at the new one.

**Still assumed rather than derived.** The run hazard is an annual rate that
GRC bends and that does not consult the deposit base — a firm funded four-to-one
on demandable money faces the same assumed run rate as one funded entirely by
its owners. Deposits adjust smoothly toward capacity, which is a run in slow
motion. Withdrawals as a shock, and the fire-sale cost of meeting them out of a
book that has not matured, are the next stage.

### 4.7 Does a reactive policy earn its place?

A recurring question: is a policy that *reacts to circumstances* worth more than
a fixed budget?

Measured properly — five random seeds, scored on data it was not trained on — the
answer at the default settings is **no**: −0.02% over two years and +0.12% over
eight, with the uncertainty bands overlapping in both cases. A single run calling
it better is reporting its seed.

That verdict survived the liability side unchanged, which is itself informative:
a second state variable was added for the learner to observe and it still could
not use it. At the long horizon the learner is also six times noisier across
seeds than the fixed policy — its best seed wins by 2.6% and its worst loses.
The spread is the finding, not the median.

The reason is not that the method is weak. **The firm's state barely moves.**
Across everything it experiences, its equity stays within about 9% of its median,
and a well-chosen fixed action is near-optimal everywhere it goes.

The fix is not a better learner. Raising the rate of cliff events — the one shock
large enough to move the state sharply — takes the advantage of a reactive policy
from +0.04% to **+9.32%**, monotonically. **A reactive policy pays when the state
makes large discrete jumps worth responding to, not when it merely drifts.**
Which is exactly the case the cliff channel was added to represent.

*(That sweep predates the liability side and has not been re-run. The verdict it
supports — no edge at the default, a large edge when the state jumps — was
re-measured and holds; the +9.32% figure itself has not been.)*

### 4.8 How often the firm decides

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

## 5. What the model currently says

Because none of the parameters are calibrated to real data, the model
deliberately **refuses to quote a budget**. It inverts the question into claims a
reader can agree or disagree with.

**The headline, with no uncalibrated parameter in it:**

> A programme costing 0.76 a year, against a business worth 28.25 as a going
> concern, must cut the annual probability of failure by at least **270 basis
> points** to pay for itself.

Both inputs are things a board already has a view on.

**Leverage makes ordinary loss reduction matter far more.** This is the largest
change the liability side produced, and it goes against what the pre-liability
model said. A firm budgeting only for expected loss reduction, ignoring survival
entirely, used to spend fifty times less than one budgeting for both. It now
spends **nine times less** — 0.023 a quarter against 0.197. Ignoring the survival
channel costs 2.1% of firm value and adds 2.7 points to the annual failure
probability, which is still a large error; but the split between "GRC is bought
by survival" and "GRC is bought by smaller losses" moved substantially toward
the second the moment the firm became levered. Losses on a book four times
capital are simply worth much more to avoid.

**GRC spend peaks at middling capitalisation.** 0.196 a quarter at equity 8 and
0.191 at 16, against 0.087 at equity 4 and 0.002 at 64. A well-capitalised firm
faces too little risk to bother; a nearly-failed one has too little business
left to protect. GRC is a genuine decision only in the band between — and that
band is now wider at the bottom than it was, because a firm with thin capital
still has a deposit-funded business worth protecting.

**Accumulation matters more than the annual number.** Treating GRC as a stock
that persists rather than an expense that vanishes is worth **+49% of firm
value** — and the firm buys *less* per quarter to get it (0.36 down to 0.20),
because a control that persists delivers the same protection for a smaller flow.
Survival over two years goes from 67% to 91%.

**The less a failure would destroy, the less a programme to avoid it is worth.**
Spend falls 41% as recovery in failure rises from nothing to 80%: 0.24 a quarter
when failure destroys everything, 0.15 when it destroys a fifth.

**As an effectiveness threshold**, a compliance programme pays if you believe a
unit of spend removes about 9% of exposure, where pure loss-reduction arithmetic
would demand an impossible 992%. The equivalent figures are 10% against 146% for
credit and 4% against 569% for operational. Credit's loss-reduction threshold is
the one that came down to something arguable — because the book it applies to is
now four times larger.

**The firm now pays a dividend.** Small, 4.4% of value, but it was exactly zero
before. With a deposit base to fund lending, capital is no longer scarce enough
that hoarding every unit beats distributing it — which is the first time the
payout control has done anything.

## 6. What is unresolved

**One number still carries too much weight.** The model stops after two years
and values whatever is left with a constant standing in for "the business
continues". That constant is **45% of total firm value** — discounted and
survival-weighted, 15.6 of a firm worth 34.5. Conditional on surviving, nothing
the firm does affects nearly half the objective, which inflates how important
survival looks relative to operations.

The liability side did not fix this and did not much worsen it: the share was
about 44% before and is 45% now. The firm and its franchise both grew, so the
ratio stayed put. It is the one headline figure in this report that the
recalibration left alone, and not for a reassuring reason.

It also cannot be fixed by solving for it. Trying to set the constant
self-consistently diverges: a larger franchise makes survival worth more, so the
firm buys more GRC, so the failure rate falls, so the franchise grows again.
Successive passes ran 15 → 40 → 58 with annual failure falling 5.5% → 3.1% — a
model talking itself into being safe. The number in use is a one-shot estimate
from a measured operating surplus, which is honest about being an assumption.

The agreed fix is to compute the continuation value from the model itself rather
than assume it, and to lengthen the horizon to ten years so it carries less
weight. Monthly rather than weekly decisions keep that affordable.

**That fix has a cost worth weighing.** The natural way to compute it uses the
grid solver — currently the only check that shares nothing with the main method.
Using its output as an input to the objective makes the two no longer
independent, and the project loses its referee. Two candidate replacements are
open, and they test different things: checking that value is consistent across
horizons, and building a gradient-free search that can grade the policy on the
full problem rather than a restricted one.

**Firm value is not yet comparable across horizons**, for the same reason. It
declines gently as the horizon lengthens, because the firm retains all its
earnings and a fixed terminal number is discounted harder the longer it waits.

**The default horizon undervalues traditional risk management.** GRC is a stock
that takes time to build, so at the two-year default a control bought to reduce
*losses* has not finished accumulating when the model stops, while one bought to
reduce the *failure rate* pays from the first period. Measured on the
pre-liability model, credit GRC spend rose fortyfold between a two-year and a
ten-year horizon — from half a percent of the budget to a sixth of it. That
sweep has **not** been re-run since deposits arrived, and it should be: leverage
already moved the survival-versus-loss-reduction split a long way toward loss
reduction (§5), and a longer horizon moves it further in the same direction. The
two effects are being confounded until it is.

**The liability side is half built.** Deposits now exist as a stock, are priced,
and shrink with the firm's capital, which is what makes leverage — and therefore
ordinary loan defaults — matter (§4.6). What is still missing is everything that
makes a *run* a run:

- no **withdrawal shock**, so deposits leave smoothly rather than suddenly;
- no **liquidity buffer the firm must hold**, only the reserves that happen to
  be left over after it decides how much to lend;
- no **fire-sale cost**, because the book matures inside the period, so there is
  never an unmatured asset that has to be dumped to meet a withdrawal.

The failure channel named "an incident becomes public and depositors leave" is
therefore still an assumed annual rate that GRC bends, and it does not consult
the deposit base. **A firm funded four-to-one on demandable money faces exactly
the same assumed run rate as one funded entirely by its owners**, which is
plainly wrong — and it is now wrong in a visible way rather than an invisible
one, because there is a deposit base sitting in the state for it to ignore.

**No parameter is calibrated.** Failure rates, loss severities and recoveries have
real external anchors — bank failure statistics, published exploit losses,
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
