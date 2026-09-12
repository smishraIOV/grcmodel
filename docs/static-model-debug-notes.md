# Derivations behind the model

Reasoning that shaped `quant/`'s formulation but does not belong in the model
spec. Kept separate from [`quant-model.md`](quant-model.md) so that document can
stay a statement of what the model *is*.

## 1. Constrained controls need softplus, not clamp

GRC budgets and investment must stay non-negative. Enforcing that with
`torch.clamp(g, min=0.0)` deadlocks the optimizer when a control starts at zero —
the natural starting point for a spend level.

`torch.clamp`'s gradient passes through only where the input is *strictly* inside
the bounds. At `input == min` exactly the gradient is $0$, not $1$:

```python
g = torch.zeros(1, requires_grad=True)
torch.clamp(g, min=0.0).backward()
print(g.grad)   # tensor([0.]) -- dead zone
```

So Adam never receives a signal to move away from the boundary. Both controls are
therefore reparameterized as $g = \operatorname{softplus}(r)$ and the
unconstrained $r$ is optimized. Softplus is smooth and strictly positive
everywhere.

One consequence shows up in the tests: a budget whose true optimum is exactly zero
is only approached asymptotically, since $\operatorname{softplus}(r) \to 0$
requires $r \to -\infty$. Assertions against a zero benchmark get a looser
tolerance ($\sim 10^{-3}$) than assertions against an interior optimum
($\sim 10^{-6}$, and in practice $\sim 10^{-15}$).

## 2. Froot-Stein needs genuine uncertainty

With a single deterministic loss $L$, GRC investment $g$, mitigation
$1 - e^{-\alpha g}$ and a convex cost on the shortfall, the value function is

$$
\begin{aligned}
\operatorname{net}(g)   &= E - g - L e^{-\alpha g} \\
\operatorname{value}(g) &= \operatorname{net}(g) - \sigma \cdot \operatorname{shortfall}(g)^{\gamma}
\end{aligned}
$$

Differentiating where $\operatorname{shortfall} > 0$, and writing
$A = -1 + \alpha L e^{-\alpha g}$ for the marginal trade-off between the direct
cost of $g$ and its mitigation benefit:

$$
\frac{d\,\operatorname{value}}{dg}
= A \left( 1 + \gamma\, \sigma\, \operatorname{shortfall}(g)^{\gamma - 1} \right)
$$

The second factor is $1$ plus non-negative terms, so it is strictly positive and
can never be why the derivative vanishes. The first-order condition therefore
requires $A = 0$ — and $A$ contains neither $\gamma$ nor $\sigma$. The optimum
lands at

$$
g^{\star} = \frac{1}{\alpha} \ln(\alpha L)
$$

regardless of how the cost function is shaped.

That is an algebraic identity, not a numerical artifact. The reason is that
Froot-Stein is a Jensen's-inequality effect: for convex $C$ and random $X$,

$$
\mathbb{E}[C(X)] \ \geq\ C(\mathbb{E}[X])
$$

with the gap growing in both $C$'s convexity and $X$'s spread. Risk management
narrows the spread and so shrinks the gap. A single fixed number has no spread, so
there is nothing for convexity to act on.

Hence every draw in `quant/` carries real probability weight over multiple states.
This is also why the headline test is a **mean-preserving spread** test: spread is
what the mechanism is about, so that is what should be asserted.

## 3. …and it needs an investment opportunity

Uncertainty alone is still not enough. With value

$$
\operatorname{value} = E - g - L_{\text{mitigated}} - P(\text{shortfall})
$$

firm value is strictly decreasing in loss and GRC is a pure expense. The model can
then say "losses are convexly expensive, so reduce them", but it cannot say that
risk management *creates* value — which is the claim in `readme.md`.

Froot, Scharfstein & Stein (1993) needs three ingredients: costly external
finance, **an investment opportunity whose funding depends on internal wealth**,
and a risk-management instrument. Adding the second is what makes the value
function concave in wealth: after a bad draw the firm must either underinvest or
pay the convex premium, so wealth in bad states is worth more at the margin than
wealth in good states. That endogenous risk aversion is the whole result.

Concretely: with the friction switched off, the value function is linear in wealth
and the optimal budgets collapse onto the separable closed form

$$
g_f^{\star} = \max\left( 0,\ \frac{\ln(\alpha_f X_f)}{\alpha_f} \right)
$$

for family $f$ with exposure $X_f$. Switching the friction on raises every budget.
That difference is the Froot-Stein premium, and it is what `run_friction_premium`
reports.

## 4. The financing cost needs a reference level

Writing the premium as $\sigma \cdot \operatorname{shortfall}^{\gamma}$ subtracts
$\text{money}^{\gamma}$ from money, so the model's answer changes with the
currency unit. It is written

$$
P(e) = \sigma\, K \left( \frac{e}{K} \right)^{\gamma}
$$

instead, with $K$ carrying money units. An earlier convexity-sweep result depended
on the missing $K$ and does not survive the correction — see
[`critical-review.md`](critical-review.md) F1–F2.

## 5. A probability channel cannot be differentiated through a sample

Compliance GRC reduces the *probability* of a breach, not its cost. But
`torch.bernoulli` is not differentiable in $p$, so the budget cannot act on a
drawn indicator.

Replacing the indicator by its probability would restore differentiability and
destroy the point: it removes exactly the discrete tail whose spread the convex
premium prices.

The fix is likelihood-ratio reweighting. Draw indicators once at a base $p_0$,
then weight each path by

$$
\frac{p(g)}{p_0} \ \text{ on breach paths,} \qquad
\frac{1 - p(g)}{1 - p_0} \ \text{ otherwise.}
$$

This is unbiased, differentiable in $g$, and leaves the breaches discrete. Two
properties make it well behaved here: the ratio is $\leq 1$ on the rare branch, so
it *reduces* rather than inflates estimator variance; and the weights integrate to
one, which is why the compliance family's expected loss still reduces to
$X_k\, e^{-\alpha_k g_k}$ and shares the closed form in §3.

Both the unbiasedness and the gradient are asserted in the test suite against
analytic values.

## 6. The regime has to be checked, not assumed

The model only has content where the finance constraint binds in *some* states and
not others. Two degenerate regimes flank it:

- **Never binds** (equity high relative to losses): no shortfall ever, the friction
  is irrelevant, and every budget collapses to the risk-neutral benchmark.
- **Always binds** (equity low): GRC changes only how deep the hole is, never
  whether the firm survives — which is not the decision an executive faces.

Both look like working models and quietly produce flat or reversed comparative
statics. `optimize_policy` therefore returns a `constrained_fraction` diagnostic,
and a test asserts the default parameters sit strictly between the two extremes.

This is also why compliance severity is set to 30 rather than something more
dramatic: at 45 and above, every state is constrained and the model slides into the
second degenerate regime.

## 7. There is no cliff-event loss channel, and that is a modelling decision

The staged path once carried an open item for "discrete cliff events": a bridge
exploit, a depeg, a licence revocation. Half of it was built and the other half
is deliberately not going to be.

The half that exists is the fatal one. Licence revocation and a run following a
public incident are **hazard** channels (`quant/hazard.py`): GRC reduces the
intensity, death is absorbing, and the whole thing is differentiable because
intensities are already expectations rather than sampled events. Nothing about
that was hard.

The half that is not built would be a discrete loss event that does **not**
kill the firm — large enough to matter, survivable, with GRC reducing its
frequency rather than its severity.

### Why it would have needed new machinery

`torch.bernoulli(p)` has no gradient in `p`, so any channel where GRC moves a
*frequency* hits §5's problem. The escape §5 used — likelihood-ratio
reweighting — is a poor fit for a rare severe event: per-step weights multiply
along a trajectory, so their variance compounds geometrically, and at a base
rate of 0.02 with 512 paths roughly ten paths would carry the entire gradient
signal per quarter.

Smoothing it away is not available either. The hazard is convex in equity, so
$\mathbb{E}[h(E - L)] \neq h(E - \mathbb{E}[L])$: replacing a 2% chance of
losing 40 with a certain loss of 0.8 deletes exactly the tail the model exists
to price.

### The three conditions, and why they do not hold together

A cliff loss channel earns its machinery only if an event is *all three* of:

1. frequency-reducible by GRC,
2. **survivable**, and
3. severe enough that smoothing it misprices the tail.

An event severe enough to usually kill belongs in the hazard, which is built
and differentiable. An event moderate enough to survive comfortably is a
heavier tail on the existing operational or compliance loss channels, which
need no new mechanism. The channel only exists in the band between — a firm
that takes a catastrophic hit and limps on.

**The decision is that this band is not part of the model.** For a firm of this
shape, an event is either severe enough to be fatal or it is moderate; there is
no limping. That is a statement about the business rather than about the
mathematics, and it closes the item rather than deferring it again.

### If it is ever reopened

The analysis, so it does not have to be redone. Four estimators, ranked by how
badly they fail rather than how well they work:

- **Straight-through relaxed Bernoulli** — hard indicator forward,
  $\text{sigmoid}((\text{logit}\,p - \text{logit}\,u)/\tau)$ backward. Pathwise,
  so low variance; the forward pass keeps the true discrete tail; no compounding
  in the horizon. Biased, but the bias is a temperature knob rather than
  something that grows with $T$. The best default.
- **Per-step score function with a baseline** — $\nabla\mathbb{E}[R] =
  \mathbb{E}[R \sum_t \nabla \log p_t]$ sums scores instead of multiplying
  weights, so variance grows linearly in $T$ rather than exponentially, and it
  is unbiased. Needs a baseline to be usable, and a baseline is a critic. Note
  that §6 of `quant-model.md` finds a critic is *not* needed for training
  stability; this is the place it would earn its keep.
- **Analytic mixture** — exact, zero variance, fully differentiable, but a
  survivable event branches $2^T$. Tractable only when the event is absorbing,
  which is precisely why the hazard channel can do it and a loss channel
  cannot. Still useful at $T = 1$ or $2$ as an exact check on whichever
  estimator is chosen.
- **Likelihood-ratio reweighting** — what the compliance channel already does.
  Adequate there because the weights stay near one; poor for a rare severe
  event, for the reasons above.
