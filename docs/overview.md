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

We model a **financial firm on crypto rails**: it takes deposits into vaults,
lends and invests them, and runs the blockchain infrastructure those assets
live on. It is a bank in most respects, with some new risks attached.

Every quarter, the firm decides:

- **how much to lend**;
- **how much capital to keep**, and how much to pay out to its owners — or,
  after a bad quarter, whether to raise new money;
- **how much to spend on GRC**;
- **whether to take on a new line of business**, and how much of it;
- **how much market risk to hedge** — for example, the risk of crypto prices
  falling;
- and, if things are bad enough, **whether to shut down**.

It must pay its depositors before its owners see anything, and, like any bank,
it must hold a minimum amount of capital against what it lends.

## What can go wrong

The firm faces three kinds of risk:

- **Market risk**, such as crypto prices moving. This can be hedged in
  financial markets at a fair price, so a sensible firm hedges it away.
- **Everyday losses**: loans that go bad, routine operational mistakes. These
  come all the time and vary in size, but rarely in a dramatic way.
- **Rare, severe losses**: a drained bridge, a hacked smart contract, a failed
  custodian. These are unlikely in any given year, but when they happen they
  can wipe out a large part of the firm's capital.

The last two cannot simply be sold off to someone else. The firm has to carry
them, and that is where GRC comes in: better controls make losses smaller,
less variable, or less frequent.

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

- **Rare, severe losses** (in progress): does GRC become much more valuable
  when it can prevent the losses that threaten the firm's survival?
- **Several business lines at once**, so we can see how the firm should spread
  its scarce capital between them.
- **Deposits that can leave**, as they do on crypto rails, including runs.
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
