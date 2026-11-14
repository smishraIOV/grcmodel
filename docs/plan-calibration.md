# Plan: calibration

Written 2026-11-14, proposed by Claude and to be agreed step by step with the
user. This is the project's current plan. Read it first, then `CLAUDE.md`
(working rules), `docs/design.md` (the specification) and
`docs/plan-tool.md` (the what-if tool, built; its step 8 is this plan).

**Status:** proposed, not started. Step 1 needs a decision from the user
before anything else.

---

## 1. What calibration means here

Every assumption in the tool is illustrative. The mechanics are tested; the
levels mean little. Calibration replaces the assumptions that matter with
values that have a source, a date and a stated range, and says plainly which
ones remain guesses.

It is **not** fitting the model to one firm, and it is not a claim of
precision. Several assumptions cannot be observed at all — above all how well
GRC works. For those the aim is a defensible range, not a number.

**The tool says where to spend the effort.** Its sensitivity
(`quant/sensitivity.py`) shows which assumptions move the results and which
do not. Assumptions that move little stay as rough guesses, labeled so.

## 2. Steps

Each step ends with tests passing and a commit, and needs the user's
approval as it comes.

| step | what | needs | done when |
|---|---|---|---|
| 1 | **BTC's expected return.** Today the market is assumed to expect of BTC the 6% a year shareholders require of everything (design §3.4). At 60% volatility that makes the typical BTC path fall by about half in five years while the average rises. The arithmetic is right; a 6% expected return for an asset that volatile is not believable. Decide what replaces it (section 3). Every multi-year figure depends on this, so it comes first. | a discussion and the user's decision, then about half a day | BTC's typical path under the defaults is one the user would defend; the reason is in the design |
| 2 | **Rank the assumptions.** Run the sensitivity on three made-up firms that differ in shape (the example; one that is mostly lending; one that is mostly the BTC vault), under a cautious and a bold policy, one period ahead and over five years. Sort the assumptions into: *needs a source*, *rough guess is enough*, *does not matter*. | nothing from the user; about two hours | a short table in this plan, agreed with the user |
| 3 | **Find sources for the ones that need them.** For each: a value, a range, the source, the date, how the value was derived from it, and how much to trust it. Candidate kinds of source, to be confirmed as they are looked up, are in section 4. | web research; about a day | each such assumption has an entry in `docs/calibration.md` |
| 4 | **Put the values in.** Change the defaults in `quant/params.py`; set the sensitivity's ranges (`standard_dials`) from the evidence in place of "half or twice"; update the ranges suggested in `quant/guide.py`; and replace the tool's single "illustrative" flag with a status for each assumption: sourced, rough, or a guess. | step 3 | the example firm's read-out, stresses and paths, shown to the user, with what changed and why |
| 5 | **Run one real firm through it.** Enter a firm's own figures in the dashboard, save the file, and go through the results together: what looks right, what looks wrong, and which assumption is behind each surprise. | a firm's figures from the user, real or realistic | a list of what to change, or agreement that it reads sensibly |
| 6 | **A measure that charges for failure.** "What shareholders take" treats failing as free beyond the equity lost, which flatters risky policies. Decide what failing costs — the value of the business lost, a fixed cost, or something else — and report a figure that includes it. A new measure: to be discussed first. | a discussion | the comparison of two policies no longer favors the bolder one by construction |
| 7 | **Re-run the research code** on the calibrated defaults and record what changed, including the two known failures in its tests. No work on its convergence. | steps 1 and 4; about half an hour of computing | `docs/plan-integrated.md` updated |

Steps 2 and 3 can overlap with step 1's discussion. Steps 5 and 6 can come
in either order.

## 3. Step 1: the question to decide

**What is wrong.** The model has one required return, 6% a year, for the
firm's business and for BTC alike. A riskier asset should be expected to earn
more.

**Options.**

1. **Give BTC its own market return**, higher than shareholders' 6%. Simple
   to state and to enter. The difficulty: the firm's own BTC would then earn
   more than shareholders require, so holding BTC inside the firm looks
   attractive again — the problem the design solved on 2026-11-13 by pricing
   BTC at the shareholders' return. It would need the firm's BTC to be valued
   at BTC's own required return, which means discounting part of the firm's
   cash flows at a second rate.
2. **Keep one required return and lower BTC's risk to match**: a volatility
   and crash rate consistent with a 6% asset. This keeps the model as it is
   and gives up describing BTC as it has actually behaved.
3. **Raise the single required return** to what BTC's risk would command.
   This keeps one rate and makes the firm's ordinary business look poor,
   since its income would be discounted as heavily as BTC.
4. **Leave the valuation alone and change only what the paths show**: draw
   BTC's price on the paths with a higher drift, as the firm's view does
   today, and say so. Cheap, and it mixes "what the market expects" with
   "what is simulated".

Claude's leaning is option 1, done carefully, because it is the only one
that describes both BTC and the firm's business honestly. It is also the most
work and reopens a design decision, so it is the user's call.

**What the tool's own input already does.** "The firm's own view on BTC"
raises the BTC return the firm plans with. It is the firm's opinion, not the
market's, so it is not the fix; but it shows the effect of a higher BTC
return today, before any change.

## 4. Step 3: kinds of source, by block

To be confirmed as each is looked up. Nothing here is a finding yet.

| block | assumptions | kind of source |
|---|---|---|
| BTC's price | volatility; how often it crashes and by how much | BTC's daily price history: the volatility of the logarithm of the price, and the frequency and size of sudden falls. How a "crash" is told apart from an ordinary fall needs a rule, stated |
| Perp funding | the premium above the riskless rate | funding-rate history on the main perpetual-futures venues |
| Liquidations | the sale discount; slippage in a smooth fall; bad debt | liquidation and bad-debt records and risk reports of the large lending protocols |
| Incidents | how often; how large | public records of exploits and custody failures, against the value exposed at the time |
| Loans | demand at a rate; the rate at which nobody borrows; how demand moves with loan-to-value and with BTC's price | borrow rates and utilization history at lending protocols |
| Investors | what they can earn elsewhere; how strongly they follow yield; how many leave after an incident | yields on offer elsewhere; deposit flows around past incidents. Likely to stay rough |
| The firm | fees, costs, stakes, the cost of raising equity | the user's own figures and judgment; published fee schedules |
| GRC | how well it works on each risk; how fast it wears off | not observable. A range from what is published on control effectiveness, and from the user's judgment; shown as a dial |

## 5. Rules that matter here (from `CLAUDE.md` and the user)

- Raise any new concept, with its source, before it goes into code or docs.
  Steps 1 and 6 are new concepts.
- Numbers go into `docs/` only with a source and a date, or from a real run
  under the `reference` profile.
- A value from research is recorded with how sure it is. A guess is called a
  guess.
- When a default changes, the tests that rest on it change with it, and the
  example firm's results are shown before and after.
- No work on the research code's convergence while calibrating.
- Do not push; the user pushes.
