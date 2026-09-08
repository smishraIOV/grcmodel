# Debugging notes: why the static model needs a two-state shock

This documents two bugs found while building `quant/static.py`'s convexity
sweep (the §5.1 milestone in `quant-model.md`), and the reasoning that led
to each fix. Kept separate from `quant-model.md` because it's debugging
history/derivation, not the model spec itself.

## Bug 1: gradient dead zone from a hard clamp at the boundary

The GRC investment level `g` must stay non-negative. The first version
enforced this with `torch.clamp(g, min=0.0)` inside the optimization loop,
with `g` initialized at exactly `0.0`.

`torch.clamp`'s gradient is defined as passing through only where the input
is *strictly* inside the bounds — at `input == min` exactly, the gradient is
`0`, not `1`. Since `g` started exactly at the boundary, every gradient
computed through the clamp was `0`, so Adam never had a signal to move `g`
away from `0`. Confirmed directly:

```python
g = torch.zeros(1, requires_grad=True)
c = torch.clamp(g, min=0.0)
c.backward()
print(g.grad)   # tensor([0.]) -- dead zone
```

**Fix:** reparameterize through `softplus` instead of clamping —
`g = softplus(raw)`, optimize the unconstrained `raw`. `softplus` is smooth
and strictly positive everywhere, including at `raw = 0`, so there's no
boundary for the gradient to die at.

## Bug 2: optimal g came out independent of convexity

After fixing the dead zone, the sweep ran, but `optimal g` was identical
(to 4 decimal places) across every convexity value from 1.0 to 4.0. A swept
parameter that produces no change in the output at all is a sign the math
is canceling it out structurally, not that the effect is merely small.

### Deriving why

With a single deterministic risk shock (total loss `L`, GRC investment `g`,
mitigation `1 - e^(-αg)`), the model's value function is:

- `net_position(g) = E - g - L·e^(-αg)`
- `shortfall(g) = max(0, -net_position(g))`
- `value(g) = net_position(g) - scale·shortfall(g)^convexity`

Differentiating (in the region where `shortfall > 0`), and writing
`A = -1 + α·L·e^(-αg)` for the marginal trade-off between the direct cost of
`g` and its loss-mitigation benefit:

```
d(value)/dg = A · (1 + convexity · scale · shortfall(g)^(convexity−1))
```

The second factor is `1` plus non-negative terms, so it is always strictly
positive — it can never be the reason the derivative hits zero. That means
the *only* way to satisfy the first-order condition `d(value)/dg = 0` is
`A = 0`. But `A`'s definition doesn't contain `convexity` or `scale` at
all. So no matter how the financing-cost function is shaped, the optimal
`g` always lands at the same point: `g* = (1/α)·ln(α·L)`. This is an
algebraic identity of the formulation, not a numerical artifact — it
confirmed the flat sweep was a real structural problem, not an optimizer
issue.

### Why: Froot-Stein is a Jensen's-inequality effect

Froot & Stein's result depends on Jensen's inequality: for a convex cost
function `C` and a random loss `X`,

```
E[C(X)] >= C(E[X])
```

with the *gap* between the two sides growing both with `C`'s convexity and
with `X`'s spread (variance). Hedging (or here, GRC investment) that
narrows the spread of `X` shrinks that gap — and a more convex `C` makes
that gap-shrinking more valuable, which is exactly the "optimal hedging
rises with convexity" result. This mechanism has nothing to act on when `X`
is a single fixed number: there is no spread to narrow, so `C`'s convexity
becomes irrelevant to the optimal choice. That is precisely bug 2 above:
the deterministic single shock removed the one ingredient (uncertainty)
the whole result depends on.

### Fix: a two-state stochastic shock

Instead of one fixed loss value, the shock now takes one of two possible
values with equal weight — a "good state" and a "bad state":

```python
credit_loss=torch.tensor([2.0, 6.0])
op_incident_loss=torch.tensor([1.0, 5.0])
compliance_breach_loss=torch.tensor([0.0, 2.0])
```

State 1 totals a loss of 3, state 2 totals 13 — same mean (8) as the old
deterministic shock, but now with spread. `FirmValueModel.compute_value`
computes `value(g)` separately in each state (mitigation, shortfall,
convex financing cost all applied per-state) and averages the two — i.e.
computes `E[value(g)]`. Because the financing cost is convex, the bad
state's cost is disproportionately large, and GRC investment (by shrinking
losses in both states) is rewarded for suppressing that disproportionate
cost more as convexity rises. This is the smallest amount of randomness
that lets the Froot-Stein mechanism exist at all — one probability-weighted
alternative outcome instead of a single certain number.

After the fix, the sweep produces the expected qualitative result: optimal
`g` rises monotonically (2.92 -> 4.32) as convexity goes from 1.0 to 4.0.

### Why this cost nothing architecturally

`RiskDraw`'s fields were already designed to hold batched tensors (for
`quant/simulate.py`'s Monte Carlo draws), and `FirmValueModel.compute_value`
already averaged over the batch dimension when present. A two-state shock
is just a batch of size 2 — the smallest possible Monte Carlo sample — so
the static model could adopt it without any change to `quant/model.py`.
