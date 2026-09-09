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
