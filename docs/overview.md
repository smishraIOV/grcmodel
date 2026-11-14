# What this project is about, in plain language

This is the non-technical companion to the project. The technical design is in
[`design.md`](design.md). This page is updated as the model grows.

## The question

Governance, risk and compliance — GRC — is the part of a company that sets
risk limits, runs controls and keeps the firm on the right side of its
regulators. Most companies treat it as a cost: necessary, but something to
keep as small as possible. Our premise is that a good GRC
program can **make a firm more valuable**. And we can
try to put numbers on that - instead of just asserting it.

## The idea we build on

The idea comes from two well-known papers in corporate finance, by Kenneth
Froot, David Scharfstein and Jeremy Stein (1993, and Froot and Stein 1998).

Their insight is that a firm with plenty of spare cash does not need to
worry much about bad luck: if it loses money, it absorbs the loss and carries
on. But most firms do not have unlimited spare cash, and raising new money
from investors is expensive, especially right after a loss, when investors are
nervous.

So a loss does more damage than its face value. It can force a firm to pass up
good opportunities because it can no longer afford them, or to raise money on
bad terms. That is why even a firm whose owners don't mind risk in principle
will pay to manage it. **Risk management is not about avoiding volatility for
its own sake. It is about protecting the firm's ability to keep investing and
operating.**

The 1998 paper applies this to banks and shows that a bank's
decisions about how much risk to hedge, which investments to take on, and how
much capital to hold are really **one decision**, not three. All three depend on
how much the bank fears a loss at that moment, and that fear depends on how
much capital it has.

Our contribution is to add GRC to that picture as a **fourth lever**, alongside
hedging, investment and capital.

## The firm we model

We model a **crypto-native financial firm**. Bitcoin is its home asset: it holds
BTC for the long term, out of conviction, and much of its business arrives in
BTC — deposits, and the collateral its borrowers post. It also holds dollar
stablecoins, to offset BTC's swings, to have cash on hand, and to pay its
costs, which are in dollars. We measure everything in dollars.

The firm runs two **vaults**. Investors put money into a vault and own a share
of it: they gain when it does well and lose when it does badly.

- **A dollar lending vault.** Investors deposit dollar stablecoins. The firm
  lends them to borrowers who hand over BTC worth more than the loan as
  security. If BTC falls far enough, the firm sells that BTC to repay the
  loan. The interest borrowers pay is the investors' return, and the firm
  keeps a share of it.
- **A BTC yield vault.** Investors deposit BTC and want more BTC back. The
  firm hires managers to earn a return on it — lending the BTC out,
  market-neutral trading, or simply keeping it safe — and charges fees: a
  small share of the vault every year, and a share of its gains.

The two vaults are joined. The BTC vault can turn part of its BTC into dollars
and lend them to the same borrowers, alongside the dollar vault, while
protecting itself so that it still gains and loses with BTC. Both then share
the interest, and the losses, in proportion to what each put in. So one
decision — what to charge borrowers — affects both groups of investors.

Every quarter, the firm decides:

- **what to charge borrowers**, and how much to lend against their BTC;
- **how to split the BTC vault** between its strategies, lending dollars
  included;
- **how much capital to keep**, and how much to pay out to its owners — or,
  after a bad quarter, whether to raise new money;
- **how much to spend on GRC**;
- **how much BTC to hold**, and how much of its BTC risk to hedge;
- and, if things are bad enough, **whether to shut down**.

It must hold a minimum amount of its own capital against the size of the vaults
it runs.

We first built a version in which the firm works like a bank, borrowing from
depositors at a fixed rate. With its conviction about BTC, that firm ended up
betting depositors' money on BTC — the familiar danger of a bank whose owners
keep the upside while depositors carry the downside. In a vault, depositors
share both, so that danger largely goes away.

## What can go wrong

The firm faces four kinds of risk:

- **Market risk**, mainly the price of BTC. This can be hedged in financial
  markets, but the firm keeps some of it deliberately, because it believes in
  BTC.
- **Everyday losses**: loans that go bad, routine operational mistakes. These
  come all the time and vary in size, but rarely in a dramatic way.
- **Rare, severe losses**: a drained bridge, a hacked smart contract, a failed
  custodian. These are unlikely in any given year, but when they happen they
  can wipe out a large part of a vault.
- **Losing the franchise**: after poor results or an incident, depositors take
  their money elsewhere, and the fees go with them. The firm also makes good
  part of its depositors' losses after an incident, to protect its reputation.

Depositors bear the vault's losses directly, but the firm pays for them
indirectly, through lost fees and compensation. That is where GRC comes in:
better controls make losses smaller, less variable, or less frequent, and so
protect the business the firm has built.

We use **fat-tailed** risk models. Many past financial crises were
worse than expected because the models behind them assumed that extreme events
were far rarer than they really are, and that different risks would not all go
wrong at once. Our model allows for both.

## What we want to measure

1. **How much GRC is worth** to a firm like this.
2. **How that value changes with the firm's situation** — its capital, the
   risks it carries, and how severe its worst losses can be. Scarce capital
   makes losses more damaging than their face value, so GRC should be worth
   more to a thinly capitalized firm facing severe risks. We measure GRC's
   value to the firm as it is; we do not compare it with imaginary firms that
   could raise money for free, because no such firm exists.
3. **How the firm's appetite for risk changes with its capital.** A
   well-capitalized firm can afford to take risk; a thinly capitalized one
   becomes cautious; and one on the brink of failure may start to gamble,
   because its owners have little left to lose and the downside falls on
   depositors.
4. **The return a new line of business must earn** before it is worth taking
   on — which depends not just on its own risk, but on how that risk adds to
   what the firm already carries.

## How we work it out

The firm's decisions play out quarter after quarter with no fixed end date. 
A computer program works out the best decision
in every situation the firm could find itself in — every combination of
capital and GRC investment — taking into account everything that could happen
next quarter and how that would affect the quarters after. While we are
testing ideas we can also run it once a year, which is much quicker.

We do not simply trust the computer's answers. We check them against cases
where the right answer is known in advance: simplified versions of the problem
that the original papers solved with pen and paper, or situations simple
enough to work out by hand. The program has to reproduce those before we
believe it on the harder questions.

## What comes next

- **A what-if tool first.** The part of the model that keeps the books — what
  the firm earns, owes and loses in any given situation — is the most solid.
  The next step builds on it directly: enter a firm's own figures and its own
  choices, and see what could happen to it, in a bad quarter or over several
  years.
- **Finding the best decisions, as research.** Working out what the firm
  *should* do is harder, and the first, rough version for the joined-up firm
  is not yet precise enough. That work continues alongside.
- **Getting the economics right before reading results.** We do not report
  the model's numbers yet. Earlier versions spent nothing on GRC and piled
  into BTC; each time the cause was something missing from the model rather
  than a finding. Results come once the model is realistic enough to trust.
- **Paying performance fees only on new highs** — a "high-water mark", so the
  firm earns nothing extra while it is recovering past losses.
- **Separate GRC programs** for different kinds of risk.
- **Calibration**: setting the model's numbers from real-world data, so its
  results describe a real kind of firm rather than an illustrative one.

## A few terms

- **Capital** (or equity): the owners' money in the firm, which absorbs losses
  before depositors are affected.
- **Hedging**: paying someone else to take on a risk you do not want to carry.
- **Recapitalizing**: raising new money from investors after losses.
- **Walking away**: the owners giving up on the firm, at which point it fails.
- **Fat tails**: extreme events being more likely than a simple "bell curve"
  model would suggest.
- **Limited liability**: the owners of a firm can lose what they invested,
  but no more; anything beyond that falls on creditors and depositors.
- **Vault**: a pool of depositors' money invested on their behalf; depositors
  own shares in it and share its gains and losses.
- **Assets under management**: the size of the vaults the firm runs.
- **Management fee / performance fee**: what the firm charges — a share of the
  vault's size each year, and a share of its profits.
