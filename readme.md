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

## See also

* [`docs/framework.md`](docs/framework.md) — GRC pillars, risk taxonomy, operating-environment structure, and the firm-value definition this project uses
* [`docs/quant-model.md`](docs/quant-model.md) — the quantitative model formulation (state, controls, frictions, objective) and a tractable path to a first implementation
