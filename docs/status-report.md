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

It carries the full set of banking risks. **Loan defaults, liquidity events and
depositor runs are central**, not background — credit losses are the single
largest loss channel in the model, at 42% of expected loss. What the crypto
rails add is a second family on top: bridge exploits, depegs, and the loss of a
licence to operate. The claim is not that the novel risks displace the textbook
ones; it is that a firm of this shape faces both, and that a GRC function has to
be budgeted against both.

Everything is a PyTorch model in `quant/`, about 4,300 lines, with 90 tests.

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
| **Funding constraint** | It cannot deploy more than it can fund, and its access to outside money shrinks exactly when it has taken losses. |
| **Cliff events** | Rare, severe hits — a bridge drained — that leave it alive but badly impaired. GRC makes them rarer; it cannot make them smaller. |
| **Wind-down option** | It can stop deliberately and recover more than a disorderly failure would leave. |

Default configuration: **quarterly decisions over two years**, a firm earning
about 19% a year on equity of 16, facing roughly 7% annual probability of failure.

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

### 4.5 Does a reactive policy earn its place?

A recurring question: is a policy that *reacts to circumstances* worth more than
a fixed budget?

Measured properly — five random seeds, scored on data it was not trained on — the
answer at the default settings was **no**: +0.03%, with the uncertainty bands
overlapping. A single run calling it better is reporting its seed.

The reason is not that the method is weak. **The firm's state barely moves.**
Across everything it experiences, its equity stays within about 9% of its median,
and a well-chosen fixed action is near-optimal everywhere it goes.

The fix is not a better learner. Raising the rate of cliff events — the one shock
large enough to move the state sharply — takes the advantage of a reactive policy
from +0.04% to **+9.32%**, monotonically. **A reactive policy pays when the state
makes large discrete jumps worth responding to, not when it merely drifts.**
Which is exactly the case the cliff channel was added to represent.

### 4.6 How often the firm decides

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

> A programme costing 0.99 a year, against a business worth 19.07 as a going
> concern, must cut the annual probability of failure by at least **517 basis
> points** to pay for itself.

Both inputs are things a board already has a view on.

**GRC spend should peak at middling capitalisation.** A well-capitalised firm
faces too little risk to be worth buying down; a nearly-failed one has too little
business left to protect, and below a threshold should be winding down instead.
GRC is a genuine decision only in the band between.

**The programme is paid for almost entirely by survival, not by smaller losses.**
A firm budgeting only for expected loss reduction would spend seventy times less.
As an effectiveness threshold, a compliance programme pays if you believe a unit
of spend removes 15% of exposure, where pure loss-reduction arithmetic would
demand an impossible 992%.

**Accumulation matters more than the annual number.** Treating GRC as a stock
that persists rather than an expense that vanishes is worth **+34% of firm
value**.

**The less a failure would destroy, the less a programme to avoid it is worth.**
Spend falls 41% as recovery in failure rises from nothing to 80%.

---

## 6. What is unresolved

**One number carries too much weight.** The model stops after two years and values
whatever is left with a constant standing in for "the business continues". That
constant is **45% of total firm value**, and the headline conclusion moves with
it: set it to zero and the GRC programme becomes uneconomic. Conditional on
surviving, nothing the firm does affects nearly half the objective — which
inflates how important survival looks relative to operations.

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

**The liability side of the balance sheet is not modelled.** This is the largest
known gap. The firm's assets and the losses against them are represented in
detail, but its *funding* is not: deposits are not tracked, there is no liquidity
buffer, and no depositor outflow. The failure channel described as "an incident
becomes public and depositors leave" is currently a rate reduced by operational
GRC spending — it does not depend on the deposit base, on how liquid the firm is,
or on anything a run would actually involve. A bank run is the canonical
liquidity event for a firm of this kind, and at present the model names it
without mechanising it.

**No parameter is calibrated.** Failure rates, loss severities and recoveries have
real external anchors — bank failure statistics, published exploit losses,
bankruptcy recoveries. The effectiveness of GRC spending does not, which is why
every output is a threshold rather than a recommendation.

---

## 7. Running it

```bash
uv run pytest                                  # 90 tests
uv run python scripts/run_dynamic_model.py     # the multi-period model
uv run python scripts/run_breakevens.py        # the decision-facing outputs
uv run python scripts/run_seed_study.py        # is a reactive policy worth it?
uv run python scripts/run_static_model.py      # the original two-period model
```

Documentation lives in `docs/`: `quant-model.md` is the specification and
results, `static-model-debug-notes.md` records the reasoning behind design
choices and the traps found along the way, `framework.md` covers the conceptual
structure.

**Code state:** all work sits on branches; `main` is untouched. The main line is
`weekly-decisions`; the rejected truncated-backpropagation experiment is isolated
in three files and can be separated cleanly.
