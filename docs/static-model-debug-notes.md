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

## 7. The cliff channel, and how a frequency is differentiated

The staged path carried an open item for "discrete cliff events". It was once
declined on the grounds that an event is either severe enough to be fatal — in
which case it is already a hazard channel — or moderate, in which case it is a
heavier tail on an existing loss channel. That reasoning was wrong, and the
correction is worth stating because it is a claim about where a *decision*
lives rather than about where a loss lives.

A moderate hit and outright failure both leave management with very few degrees
of freedom: absorb it, or it is over. The cliff is the case in between, and it
is the one worth avoiding precisely *because* it does not resolve anything. The
firm emerges alive, badly impaired, holding a real decision — rebuild, run
down, or wind up — with a balance sheet that no longer funds its opportunity
and a hazard that has risen sharply. A model that offers only "moderate" and
"fatal" has no state in which GRC's option value is doing anything, because
both of its outcomes are ones where the choice has already been made for you.

Measured, that limbo is real and it works through funding rather than through
the loss itself: over the quarters following a strike, the share of paths that
want to deploy more than they can fund roughly doubles against unstruck paths
(0.33 versus 0.14 immediately, 0.78 versus 0.36 later). The cliff does not
merely cost money, it impairs the balance sheet that gates investment.

### GRC acts on the frequency, not the severity

A bridge is either drained or it is not. Controls make the exploit less likely;
they do not make it smaller. That is the opposite of the operational loss
channel, where controls contain an incident that happens anyway, and the two
are modelled separately because they are different claims. Modelling GRC as
reducing both would let one $\alpha$ buy the same protection twice, with the
second purchase free.

Severity is drawn as a fraction of opening equity — exponential with mean 0.35,
capped at one — so it scales with the firm and stays unit-invariant. It leaves
43% of strikes above 0.3 of equity, 10% above 0.8, and only 6% total: heavy
enough to matter, and not so heavy that the interesting middle disappears.

### Differentiating an occurrence probability

`torch.bernoulli(p)` has no gradient in $p$, which is §5's problem in a new
place. The escape §5 used — likelihood-ratio reweighting — is a poor fit here:
per-step weights multiply along a trajectory so their variance compounds, and
at a base rate of 2.5% only a handful of paths per quarter would carry the
entire gradient.

Smoothing it away is not available either. The hazard is convex in equity, so
$\mathbb{E}[h(E - L)] \neq h(E - \mathbb{E}[L])$: replacing a 2.5% chance of
losing a third of the firm with a certain loss of 1% deletes exactly the state
the channel exists to represent.

What is used instead is a **straight-through relaxation**. The forward pass is
the true hard indicator, so the jump and its tail are exact; the backward pass
differentiates $\text{sigmoid}\big((\text{logit}\,p - \text{logit}\,u)/\tau\big)$.
The gradient is biased, but the bias is a temperature knob rather than
something that grows with the horizon — a far better failure mode than a
variance that compounds.

### The bias is measured, not assumed

Against the exact analytic mixture $p(G) \times \text{severity}$, which is
affordable at one period because the event is binary. Ratio of the relaxed
gradient to the exact one, large sample:

| $\tau$ | 1.00 | 0.50 | 0.25 | 0.12 | 0.06 | 0.03 |
|---|---|---|---|---|---|---|
| gradient ratio | 3.40 | 1.48 | 1.11 | 1.03 | 1.02 | 1.01 |

The variance runs the other way. Relative standard deviation of the gradient
across scenario draws at 512 paths is 0.24 at $\tau = 0.5$ and 1.41 at 0.03.
The default of 0.1 sits where the bias has flattened and the variance has not
yet taken over. **The sign was never wrong at any temperature tested**, which
is the property that actually matters for a descent direction.

The analytic mixture stays useful as a permanent check rather than as an
implementation: over $T$ periods a survivable event branches $2^T$, which is
precisely why the *hazard* channel can be exact — one branch terminates — and a
loss channel cannot.

### What it did not do

The wind-down option stays unexercised. A struck firm's funding is impaired but
the production franchise is unconditional, so continuing still dominates
winding down even from the limbo state. That is the open item recorded in
`quant-model.md` §6 rather than anything about this channel.

Worth noting for anyone extending it: strikes are rare by construction — around
nine paths in four thousand per quarter — so a learned policy receives almost
no gradient signal about how to behave *after* one. The limbo state is the
hardest decision in the model and the least trained. Stratifying the proposal
during training, and correcting by weight, is the obvious remedy and is not
built.
