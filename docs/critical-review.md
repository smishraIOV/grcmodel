# Critical review — objective, current state, and next steps

Review date: 2026-09-08. Reviewed at commit `690c0ce`.
Review by Opus 5 (high effort) of initial work done by Sonnet 5 (medium)

> **Status: addressed.** F1–F7 and the §7 recommendations were implemented after
> this review; see [`quant-model.md`](quant-model.md) §6 for the current results
> and [`static-model-debug-notes.md`](static-model-debug-notes.md) for the
> derivations. The convexity result discussed below did **not** survive the
> correction and has been withdrawn rather than repaired — §7 item 1 asked for an
> honest re-run, and this was its outcome. The document is kept unedited as the
> record of what was wrong and how it was found; every "current" statement in it
> describes commit `690c0ce`, not the code today.

Scope: `readme.md`, `docs/framework.md`, `docs/quant-model.md`,
`docs/static-model-debug-notes.md`, the `quant/` package, `tests/`, and
`scripts/`. All code was executed; the numerical claims below are
reproductions, not readings.

---

## 1. Verdict

The conceptual scaffolding is well above average for a project this young.
The document chain (`readme` → `framework` → `quant-model` → code) is
genuinely traceable, and `static-model-debug-notes.md` is a model of how to
record a derivation rather than just a fix.

The implementation does not yet support the claim made for it. The headline
result — "optimal GRC investment rises with financing-cost convexity" —
**reproduces, but is an artifact of a dimensionally inconsistent cost
function combined with one specific parameterization.** Re-denominate the
same firm in different currency units, or capitalize it well enough that
GRC actually changes whether it survives, and the result weakens, vanishes,
or reverses. Details and reproductions in §4.

Separately, and more fundamentally: the model implemented is not the
Froot-Stein mechanism. It is a convex-penalty Jensen effect, which is one
ingredient of Froot-Stein but not the part that makes risk management
*value-adding* (§4, F3). That matters because value-addition is the entire
premise in `readme.md`.

None of this is fatal. The bugs are in the cost function and the
parameterization, not the architecture, and the architecture is sound.

---

## 2. What is working well

- **Document-to-code traceability.** Nearly every function carries a
  pointer to the doc section it implements. This is rare and worth
  protecting as the code grows.
- **`static-model-debug-notes.md`.** Bug 2's derivation — showing
  algebraically that $d(\text{value})/dg = 0$ forces $A = 0$, and that $A$
  contains no convexity term — is correct, and the diagnosis (Froot-Stein is a
  Jensen's-inequality effect that needs genuine uncertainty) is exactly
  right. Keeping this separate from the model spec was the right call.
- **The softplus reparameterization** (`quant/model.py:82-87`) is the
  correct fix for the clamp gradient dead zone, and the reasoning recorded
  for it is accurate.
- **Shared model across stages.** `quant/static.py` and `quant/simulate.py`
  differ only in batch size against the same `FirmValueModel` /
  `optimize_policy`. That is the right factoring, and it made the two-state
  fix free — as the debug notes correctly observe.
- **The convexity = 1.0 row is a correct self-check.** It reproduces the
  closed-form risk-neutral benchmark
  $g^\star = \tfrac{1}{\alpha}\ln(\alpha \bar{L}) = 2.9182$ to four
  decimals. Whether or not this was deliberate, it is a real validation
  that the optimizer and the objective agree with analysis.

---

## 3. Reproduction

```
$ uv run pytest                          # 1 passed
$ uv run python scripts/run_static_model.py
device: mps
 convexity |  optimal g | firm value
      1.00 |     2.9182 |    -6.5031
      1.50 |     3.3485 |    -9.9229
      2.00 |     3.7087 |   -17.1836
      2.50 |     3.9574 |   -32.6883
      3.00 |     4.1256 |   -66.1132
      3.50 |     4.2420 |  -138.6671
      4.00 |     4.3232 |  -296.9041

$ uv run python -m quant.simulate
optimal g: 5.1227
expected firm value: -25.3409
```

Everything runs clean on first try. Two things are visible in the output
before any analysis:

1. **Firm value is negative in every row, and gets worse by a factor of
   45x across the sweep.** The "optimal" policy leaves the firm deeply
   underwater. A reader is entitled to ask what is being optimized.
2. A `Failed to initialize NumPy` warning on every invocation (§6).

---

## 4. Critical findings

### F1 — The financing cost function is dimensionally inconsistent, so the convexity sweep is not a controlled experiment

`quant/frictions.py:17`:

```python
return scale * shortfall**convexity
```

`shortfall` is money, so `shortfall**convexity` carries units of
$\text{money}^{\gamma}$. `quant/model.py:56` then computes
`net_position - cost`, subtracting $\text{money}^4$ from money. The units do
not agree, which means **the model's answer depends on the choice of currency
unit.**

Reproduction — take the *same firm* and redenominate it (scale equity,
losses and $g$ by $k$; $\alpha$ has units $1/\text{money}$ so it scales by
$1/k$). A well-posed model must return $g^\star(k) = k \cdot g^\star(1)$, i.e.
$g^\star/k$ constant:

| $k$ | $\gamma = 1.0 \to g^\star/k$ | $\gamma = 2.0 \to g^\star/k$ | $\gamma = 4.0 \to g^\star/k$ |
|---|---|---|---|
| 0.01 | 2.9182 | 2.9969 | 2.9189 |
| 0.1  | 2.9182 | 3.3497 | 3.3222 |
| 1.0  | 2.9182 | 3.7087 | 4.3248 |
| 10.0 | 2.9182 | 3.7783 | 3.9832 |

The linear case (convexity = 1.0) is exactly invariant, as it must be —
which rules out the optimizer as the cause. Every convex case is not. The
same firm, described in cents rather than dollars, gets a materially
different real GRC decision (2.997 vs 3.778, a 26% difference at
convexity 2).

The mechanism is simple: for $\text{shortfall} > 1$ raising the exponent
inflates the penalty enormously; for $\text{shortfall} < 1$ it *shrinks* it. So "increasing
convexity" in this model simultaneously changes the curvature **and** the
overall level of the distress penalty — by three orders of magnitude across
the sweep. The reported result cannot distinguish "GRC rises because
distress got more convex" (the Froot-Stein claim) from "GRC rises because
distress got 1000x more expensive" (a trivial level effect).

**Fix.** Introduce a reference distress level $K$ carrying money units and use

$$
P(e) = \sigma\, K \left( \frac{e}{K} \right)^{\gamma}
$$

Then $\gamma$ is a pure shape parameter and the sweep is a controlled
experiment. Verified: with this form, $g^\star/k$ is constant to four decimals
across $k \in \{0.1, 1, 10\}$.

Note this is not merely cosmetic. Under the corrected form with $K = 10$,
the sweep is **no longer monotone** — $g^\star$ rises to 3.40 at convexity 2.5
then falls back to 3.32 at convexity 4.0. The qualitative conclusion is
sensitive to $K$, which is currently an implicit, unstated $K = 1$.

### F2 — The headline result holds only in a regime where the firm is certainly insolvent; it reverses in the regime that matters

`INITIAL_EQUITY = 3.0` (`quant/static.py:30`) against a mean loss of 8.0.
At the optimum the firm is underwater in **both** states, at every
convexity:

| convexity | net position (good state, bad state) |
|---|---|
| 1.0 | (−1.17, −5.33) |
| 2.0 | (−1.69, −4.98) |
| 4.0 | (−2.14, −4.88) |

So the solvent branch of the cost function is never exercised, and GRC
investment never changes *whether* the firm survives — only how deep the
hole is. That is not the decision an executive faces.

Sweeping equity, holding everything else fixed:

| initial equity | $g^\star$ across $\gamma = 1.0 \to 4.0$ | monotone? |
|---|---|---|
| 3.0 (always insolvent) | 2.918 → 4.323 | yes ✅ |
| 8.0 (solvent in good state, marginal in bad) | 3.549 → 3.006 | **no — decreasing** |
| 12.0 (always solvent) | 2.918 → 2.918 (flat) | vacuous |
| 16.0 (always solvent) | 2.918 → 2.918 (flat) | vacuous |

The $E = 8.0$ reversal was confirmed by brute-force grid search with the
optimizer removed entirely ($\gamma = 1.0 \to g^\star = 3.550$;
$\gamma = 4.0 \to g^\star = 3.005$), and is stable to 20,000 optimizer steps.
It is real.

The $E = 12/16$ flatness is correct behaviour (no shortfall ever ⇒ financing
convexity is irrelevant ⇒ $g^\star$ collapses to the risk-neutral benchmark).
The $E = 8.0$ reversal is the F1 unit bug biting in the other direction: the
bad-state shortfall there is ≈ 0.03, and raising a number below 1 to a
higher power makes it *smaller*.

Taken together, F1 and F2 mean the current result is a coincidence of
`INITIAL_EQUITY = 3.0` and $K = 1$, not a robust property of the
formulation.

### F3 — This is not the Froot-Stein mechanism

`quant/model.py:52-56`:

```python
mitigated_loss = risk_draw.total_loss() * (1.0 - mitigation)
net_position   = self.initial_equity - grc_investment_cost(g) - mitigated_loss
shortfall      = torch.clamp(-net_position, min=0.0)
value          = net_position - cost
```

There is no revenue term, no return on deployed capital, and no investment
opportunity. Firm value is strictly decreasing in loss, and GRC is a pure
loss-reduction expense.

Froot, Scharfstein & Stein (1993) requires three ingredients: (a) costly
external finance, (b) **investment opportunities whose value depends on
internal wealth**, and (c) a risk-management instrument. This model has (a)
and (c). Without (b), the model cannot express the actual result — that
risk management is valuable because it *preserves the ability to fund
positive-NPV investment in bad states* — and so it cannot support
`readme.md`'s central claim that GRC increases firm value. It can only say
"losses are convexly expensive, so reduce them."

That distinction is exactly the one `framework.md` §3 was careful to make
when it rejected "minimize ruin probability" for not capturing "the
Froot-Stein point that GRC is a value-adding investment." The code as
written has landed closer to the rejected alternative than to the chosen
one.

**Fix.** Add an investment/return term: capital surviving into the second
period earns a return $r$ on deployed capital, with deployment capped by
available internal funds. Then GRC has a value-creation channel (protecting
investment capacity), not just a loss-avoidance channel, and firm value
stops being negative everywhere.

### F4 — The risk taxonomy is decorative

`RiskDraw` carries `credit_loss`, `op_incident_loss`,
`compliance_breach_loss`, but `total_loss()` (`quant/model.py:32-33`) sums
them immediately and a single `grc_mitigation` factor is applied uniformly.
The three-way split is mathematically inert.

Verified: replacing the three-way split with a single aggregate loss of the
same total produces $g^\star$ identical to six decimal places (3.708688 at
$\gamma = 2.0$; 4.323193 at $\gamma = 4.0$).

The `RiskDraw` docstring is honest that this is a simplification, which is
good practice. But `framework.md`'s carefully built risk taxonomy currently
contributes nothing to the model, and the code gives a misleading impression
of structure it does not have. The taxonomy only starts to earn its place
when each family gets its own GRC sensitivity — and, more importantly, when
GRC acts on the right *moment* per family: a compliance programme reduces
breach **probability**, a security programme reduces incident **severity**,
credit underwriting reduces **mean**. Right now everything is a proportional
haircut on realized loss.

### F5 — The test locks in the artifact

`tests/test_static_model.py:11-13` asserts monotonicity of $g^\star$ in
convexity. Per F1 and F2, that assertion passes because of the unit bug and
the equity parameterization. Fixing either one breaks the test — so the test
currently defends the bug rather than the intended economics.

A test that would survive the fixes: assert **unit invariance** (F1's
table), assert the $\gamma = 1.0$ row equals the closed-form risk-neutral
benchmark $\tfrac{1}{\alpha}\ln(\alpha \bar{L})$, and assert $g^\star$
responds to loss *spread* at fixed mean (the actual Jensen claim) rather than
to convexity.

That last one is the important one. The Froot-Stein/Jensen mechanism the
debug notes correctly identified is fundamentally about **spread**. A
mean-preserving spread test is a direct test of the mechanism and is
immune to all of the cost-function-shape problems above.

### F6 — `simulate.py` does not do what `quant-model.md` §5.2 specifies

§5.2 promises: "compare the resulting expected-value / loss-distribution
across a few candidate GRC investment policies." `run_policy_search`
(`quant/simulate.py:60-72`) optimizes a single scalar against one fixed
sample and returns one number. There is no policy comparison, no loss
distribution reported, and no sampling-error estimate.

The sampling error is not negligible — severities are exponential with
mean 20 gated by rare Bernoullis, so the estimator is heavy-tailed:

| n_paths | $g^\star$ across 5 seeds | spread |
|---|---|---|
| 1,024 | 4.846 – 5.125 | 0.279 |
| 4,096 (default) | 5.052 – 5.148 | 0.096 |
| 16,384 | 4.948 – 5.179 | 0.231 |
| 65,536 | 5.057 – 5.100 | 0.044 |

Note 16,384 is *worse* than 4,096 — the tell of a heavy-tailed estimator
where a single extreme draw dominates. Reporting `optimal g: 5.1227` to
four decimals implies precision the estimate does not have. The default
`seed=0` makes this invisible.

### F7 — Doc/code drift on scope

`quant-model.md` §1 lists six state variables and §2 lists four control
classes. The code implements one state ($E$) and one scalar control ($g$).
Deposits $D$, liquidity buffer $L$, the operational-risk indicator and the
compliance-risk score are absent, as are three of the four controls.

This is legitimate staging, and §5 does present a sequence. But §6's
"Status: in progress" understates the gap, and nothing in the docs marks
which of §1/§2's items are aspirational versus implemented. A reader
arriving at `quant-model.md` will substantially overestimate what exists.

---

## 5. On the objective itself

Three observations that are about the project's aim rather than its code.

**The Froot-Stein anchor is defensible but is currently doing rhetorical
rather than analytical work.** It is a real and appropriate piece of theory
for this argument — "even risk-neutral firms benefit from risk management"
is exactly the right lever for justifying GRC spend to a CFO. But per F3,
the implemented model does not contain the mechanism. Either implement the
investment channel, or restate the claim honestly as "convex distress costs
imply loss reduction has increasing returns" — which is true, useful, and
much weaker than what the docs currently assert.

**The binding constraint on usefulness is calibration, not mathematics.**
The single most consequential parameter is $\alpha$ — how much loss reduction a
dollar of GRC spend buys. There is currently no plan for estimating it, and
no amount of modelling sophistication downstream compensates for not knowing
it. Steps 5.2 and 5.3 (Monte Carlo, then MDP) both make the model *more
elaborate* without making it *more calibrated*, which is the wrong axis to
move on next.

**Reframe the output from point estimate to threshold.** An executive
cannot act on $g^\star = 3.7087$ — the units are unstated, the parameters are
invented, and the precision is fictional. An executive *can* act on: "this
GRC programme pays for itself provided you believe a dollar of spend
reduces expected loss by at least $X\%$" or "GRC spend above $Y$ is
value-destroying under any plausible mitigation curve." Inverting the
problem into a break-even/threshold question makes the output robust to the
calibration gap rather than hostage to it. This is a small change to what
gets computed and a very large change to what the model is good for, and I
would rank it above every remaining item in §5's roadmap.

**Repo identity.** The project is a GRC framework living in a repo named
`claude-code-101`, alongside `notes101.md` (unrelated Claude Code course
notes, still tracked, with its ignore line commented out in `.gitignore`).
Worth resolving before anyone else is invited in. *(Resolved: the project
directory is now `GRCmodel`, and `notes101.md` has been removed.)*

---

## 6. Smaller issues

- **`numpy` is not installed**, so every run emits `Failed to initialize
  NumPy`. Add it to `pyproject.toml` dependencies.
- **`torch.tensor(self.financing_convexity)` is constructed inside the
  optimization loop** (`quant/model.py:55`), on CPU, once per step —
  3,500 redundant allocations per sweep, plus a silent CPU/MPS mix. Hoist
  it into `FirmValueModel` or pass the float through.
- **`scripts/run_static_model.py:15-20` duplicates `quant/static.py:49-54`
  verbatim.** Have the script call a single shared `main()`.
- **No `CLAUDE.md`.** `notes101.md:3-4` contains an instruction addressed to
  Claude, with a note-to-self that it belongs in `CLAUDE.md`. Agent
  instructions embedded in data files are unreliable and will be read by
  anything that globs the repo; move it.
- **No CI, no linter, no formatter.** One test, run manually.
- **No `LICENSE`.**
- **`financing_scale` is defined but never varied** — dead parameter for now.
- **`quant/__init__.py` is empty**, and `[tool.uv] package = false` means
  imports work only via pytest's rootdir insertion and the `sys.path` hack
  in `scripts/`. Making it an installable package would remove both.

---

## 7. Recommended changes, in priority order

1. **Fix the financing cost's dimensions** (F1). Add a money-valued
   reference level `K` to `FirmValueModel`; use
   `scale · K · (shortfall/K)^convexity`. Then re-run the sweep and report
   honestly what survives — including if the monotone result does not.
2. **Replace the monotonicity test with mechanism tests** (F5): unit
   invariance, the risk-neutral benchmark identity at convexity = 1, and a
   mean-preserving-spread test. The MPS test is the real Froot-Stein claim
   and is robust to cost-function shape.
3. **Re-parameterize so the firm is not certainly insolvent** (F2). Choose
   equity such that GRC investment moves the firm across the solvency
   boundary in some states — that is the only regime where the model has
   anything to say. Report results across an equity sweep, not at one point.
4. **Add the investment/return channel** (F3). Without it the project cannot
   substantiate its own thesis.
5. **Report uncertainty in `simulate.py`** (F6). Multiple seeds, a
   confidence interval on `g*`, and the realized loss distribution. Stop
   printing four decimals on a ±0.1 estimate.
6. **Add the threshold/break-even output mode** (§5). Highest ratio of
   decision-usefulness to implementation effort in this list.
7. Housekeeping from §6 — `numpy`, `CLAUDE.md`, CI, dedupe the script.

Explicitly *not* recommended yet: `quant-model.md` §5.3 (MDP / value
iteration). The state space is not the bottleneck; the cost function's
correctness and the parameters' provenance are.

---

## 8. Suggested next steps

**Immediate (unblocks everything else):** items 1–3 above. These are small
edits — a reference level in one function, three tests, one constant — and
until they are done, no number this repo produces should be quoted to
anyone.

**Next:** item 4, the investment channel. This is the difference between a
model *about* Froot-Stein and a model *of* it, and it is what lets firm
value be positive, which in turn makes the whole thing legible to a
non-technical reader.

**Then:** items 5–6, turning the Monte Carlo stage into something that
reports a decision rather than a number.

**Open question worth deciding before more building:** is smooth convex
distress cost the right shape for this business at all? For a regulated
crypto intermediary, the dominant risks in `framework.md`'s crypto-native
column — a bridge exploit, a depeg, losing a licence — are closer to
discrete cliff events than to a smooth convex penalty, and compliance
failure in particular is nearer a loss of licence-to-operate than a
proportional cost. A survival/real-option framing may fit the actual
business better than the continuous-cost framing inherited from the
corporate-finance literature. That is a genuine modelling fork, and it is
cheaper to take it now than after §5.3.
