# GRC framework — conceptual detail

This document expands the vision in [`readme.md`](../readme.md) into enough
structure to reason about, and to feed the quantitative model in
[`quant-model.md`](quant-model.md).

## 1. GRC pillars for this business

**Governance** — decision rights and escalation across the three arms:
- *Deposits*: who sets vault terms (redemption windows, caps) and who can
  freeze withdrawals in a stress event.
- *Lending / investment*: who sets strategy-level risk limits, and who can
  override an investment manager's allocation.
- *Core tech infra*: who owns upgrade authority over the contracts and keys the
  vaults run on.

The common thread: each arm needs a named owner of risk appetite and a named
authority who can halt activity, independent of day-to-day management.

**Risk** — split into two families, because they behave very differently:

| Traditional (bank-like) | Crypto-native |
|---|---|
| Credit risk (borrower default) | Smart-contract bugs |
| Market risk (price moves on collateral/strategies) | Oracle failure / manipulation |
| Liquidity / run risk  | Custody / key-management risk |
| Interest-rate / funding risk | Stablecoin depeg |
| | Bridge / interoperability risk |


**Compliance** — the regimes that actually bind:
- Banking-style prudential regulation (capital/liquidity requirements), to
  the extent the entity is regulated as a bank or bank-like intermediary.
- AML/KYC obligations on deposits and counterparties.
- Crypto-specific regimes (e.g. MiCA-style licensing in the EU, US
  money-transmitter law state-by-state) governing custody and token
  issuance/transfer.

The lists above are what actually constrains decisions, and they feed the
quantitative model's constraint set.

The three families are not interchangeable buckets of expected loss, and GRC
spend does not act on them the same way: better underwriting shifts the **mean** of credit
losses, security and operational controls contain the **severity** of an incident
without preventing it, and a compliance programme reduces the **probability** of
a breach without softening the penalty once one lands. Compliance is modelled as
rare but severe, because what is at stake is licence to operate rather than a
proportional fine — that asymmetry is what makes the family worth separating at
all. See [`quant-model.md`](quant-model.md) §3.

**Open:** risks that resist explicit modelling need a generic home in the
quantitative model. Nothing currently plays that role.

## 2. Structural map of the operating environment

A balance-sheet-style view:

- **Liabilities**: deposits, represented as vault shares (claims on vault
  NAV, redeemable per the vault's terms).
- **Assets**: capital deployed by investment managers into strategies
  (lending, market-making, yield strategies, tokenized RWA holdings).
- **Overlay, not a balance-sheet line**: core tech infra. It doesn't hold
  value itself — it's the substrate every vault, strategy, and asset runs
  on, and it's the source of the crypto-native risk column above. Model it
  as a multiplier on operational-risk probability/severity, not as an
  asset or liability.

Chain: *depositor → vault (liability) → investment manager → strategy
(asset)*, with tech infra cutting across every link in that chain.

## 3. Defining "firm value"

This is the hinge between this document and the quantitative model.

**Chosen definition: expected present value of future net cash flows, minus
expected costs of financial distress and operational/compliance frictions.**

Rejected alternatives:
- *Minimize ruin probability* — misses the Froot-Stein point that GRC is a
  value-adding investment, not only a tail-risk backstop.
- *Risk-adjusted return target (Sharpe-like)* — belongs to the
  investment-manager layer, where the decision is strategy selection.

This definition is what `quant-model.md`'s objective function maximizes.
