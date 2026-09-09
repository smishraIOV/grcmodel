# GRC
Initial explorations for building a GRC framework. GRC is Governance (corporate or organizational), Risk management, and Compliance. It's wider than ERM.

Working assumption is that GRC, if done well, can increase organization's value

In a narrow, academic sense of corporate finance, this is related to the work of Froot and Stein, that even risk-neutral firms can gain value from adopting active risk management. In the theoretical models, a sort of endogenous risk-aversion arises from the concavity of cost functions (financial frictions).

A full organizational level optimal control problem of maximizing firm value through GRC seems impractical. However, this can still serve as an aspirational goal for this project.

We hope to build a pragmatic, quantitative model of optimal GRC,one that can serve as an aid to executive decision making


## operating environment

The organization is a financial intermediary, like a bank on crypto rails. It has three main arms
* deposits
* lending / investments / loans
* core technology infrastructure: this is the "crypto-rails" part. All assets (deposits, loans) exist in the form of digital cryptocurrencies or tokenized real world assets (such as bonds)

Deposits go into "vaults". Investment managers invest these assets into various strategies.

## Running the model

```bash
uv run python scripts/run_static_model.py   # Froot-Stein premium + mean-preserving spread
uv run python -m quant.simulate             # Monte Carlo, reported across seeds
uv run python -m quant.threshold            # break-even analysis
uv run pytest                               # test suite
```

The headline number: with external finance costless the firm spends 0.77 on GRC;
facing convex financing costs the same firm spends 2.56. That gap is the
Froot-Stein premium — risk management that pays for itself only because capital
is expensive to raise in bad states. It is the quantitative form of the claim
above that GRC can increase firm value.

Parameters are illustrative and uncalibrated, which is why `quant/threshold.py`
reports the effectiveness a programme must reach to be worth running rather than
a spend recommendation.

## This runs on CPU in float64, deliberately — not on the GPU

Despite being a PyTorch project on Apple Silicon, the default device is **CPU**
and the default dtype is **float64**. This is not an oversight, and switching it
to MPS to "speed it up" will make things worse on both counts:

* **MPS cannot allocate float64 at all.** PyTorch raises
  `Cannot convert a MPS Tensor to float64 dtype`. The precision is not optional
  here: the zero-friction control test asserts that optimal GRC budgets land on a
  closed-form analytic benchmark, and it matches to ~1e-15. Under float32 that
  assertion degrades to roughly 1e-7 and stops being a meaningful check on
  whether the optimizer and the objective agree.
* **MPS is also slower for this problem.** Measured: 200 backward passes over
  20,000 paths take 0.015 s on CPU and 0.031 s on MPS. The tensors are tiny — a
  three-element vector of controls plus a few thousand paths — so kernel-launch
  overhead dominates the arithmetic and the GPU never gets to work.

`quant/device.py` keeps `get_device(prefer_accelerator=True)` as an opt-in, and
`get_dtype()` will hand back float32 if you take it, because MPS accepts nothing
else. Worth revisiting only if batch sizes grow by orders of magnitude.

## See also

* [`docs/framework.md`](docs/framework.md) — GRC pillars, risk taxonomy, operating-environment structure, and the firm-value definition this project uses
* [`docs/quant-model.md`](docs/quant-model.md) — the quantitative model formulation (state, controls, frictions, objective) and the current results
* [`docs/static-model-debug-notes.md`](docs/static-model-debug-notes.md) — derivations behind the formulation, and the failure modes each one guards against
* [`docs/critical-review.md`](docs/critical-review.md) — critical review of the objective and the earlier implementation, with prioritized next steps
