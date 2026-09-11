#!/usr/bin/env python3
"""Measure where the accelerator starts paying, per numerics profile.

readme.md used to assert that MPS is slower for this problem. That was true,
and is still true at the sizes the static model runs at -- but it is a
statement about batch size, not about the hardware, and the dynamic model in
docs/quant-model.md section 5 runs several orders of magnitude above it. This
script is what that claim is now sourced from; re-run it on new hardware or a
new torch rather than trusting the numbers in the readme.

Two workloads, chosen because they bracket what the project actually does:

  path-control   Adam over a per-path control vector. The static and Monte
                 Carlo stages' shape: elementwise transcendentals over a batch,
                 no matmul.
  policy-net     Adam over a small MLP mapping state to action. The shape the
                 learner will have.

Usage: uv run python scripts/bench_profiles.py [--steps 200]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from quant.numerics import PROFILES, NumericsProfile

# Swept low enough to bracket the crossover, not just to show the large end
# winning. quant/static.py runs at n=4; quant/simulate.py at 8192.
PATH_SIZES = [2**2, 2**8, 2**10, 2**12, 2**13, 2**15, 2**17, 2**20]
NET_SIZES = [2**8, 2**10, 2**12, 2**15, 2**18]
STATE_DIM, ACTION_DIM, HIDDEN = 6, 4, 64


def _sync(profile: NumericsProfile) -> None:
    if profile.device.type == "mps":
        torch.mps.synchronize()


def _time(step, profile: NumericsProfile, n_steps: int) -> float:
    """Steady-state cost, warmed up first.

    The warmup is not a courtesy to the GPU -- leaving it out is how the
    earlier "MPS is 2x slower at 8192 paths" measurement happened. MPS pays a
    one-off compilation cost on first launch (tens of ms), which swamps a
    200-step timing loop and says nothing about an optimizer that runs for
    thousands. Anything short-lived enough for that cost to matter is also too
    short to be worth moving off the CPU.
    """
    for _ in range(10):
        step()
    _sync(profile)
    start = time.perf_counter()
    for _ in range(n_steps):
        step()
    _sync(profile)
    return time.perf_counter() - start


def bench_path_control(profile: NumericsProfile, n: int, n_steps: int) -> float:
    """Elementwise transcendentals over a path batch -- quant/model.py's shape."""
    loss_draw = profile.to(torch.rand(n, dtype=torch.float64) * 10.0)
    raw = profile.zeros(n, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.05)

    def step():
        optimizer.zero_grad()
        control = torch.nn.functional.softplus(raw)
        payoff = control - torch.exp(-control) * loss_draw - (control / 5.0) ** 2
        (-profile.sum(payoff)).backward()
        optimizer.step()

    return _time(step, profile, n_steps)


def bench_policy_net(profile: NumericsProfile, n: int, n_steps: int) -> float:
    """A small state-to-action network -- the learner's shape."""
    states = profile.to(torch.rand(n, STATE_DIM, dtype=torch.float64))
    net = torch.nn.Sequential(
        torch.nn.Linear(STATE_DIM, HIDDEN),
        torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN, HIDDEN),
        torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN, ACTION_DIM),
    ).to(device=profile.device, dtype=profile.dtype)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    def step():
        optimizer.zero_grad()
        (-profile.sum(net(states))).backward()
        optimizer.step()

    return _time(step, profile, n_steps)


def run(name: str, bench, sizes: list[int], profiles: list[NumericsProfile], n_steps: int) -> None:
    print(f"\n{name}: {n_steps} optimizer steps, seconds (lower is better)")
    print(f"{'n':>10} | " + " | ".join(f"{p.name:>10}" for p in profiles) + " |  best")
    for n in sizes:
        timings = [bench(profile, n, n_steps) for profile in profiles]
        best = profiles[timings.index(min(timings))].name
        cells = " | ".join(f"{t:>10.3f}" for t in timings)
        print(f"{n:>10} | {cells} |  {best}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=200)
    n_steps = parser.parse_args().steps

    profiles = [p for p in PROFILES.values() if p.available()]
    unavailable = [p.name for p in PROFILES.values() if not p.available()]
    print(f"torch {torch.__version__}, profiles: {', '.join(p.name for p in profiles)}")
    if unavailable:
        print(f"skipped (device unavailable): {', '.join(unavailable)}")

    run("path-control", bench_path_control, PATH_SIZES, profiles, n_steps)
    run("policy-net", bench_policy_net, NET_SIZES, profiles, n_steps)

    print(
        "\nThe crossover is the row where 'best' switches away from a cpu profile."
        "\nBelow it the accelerator's per-step launch overhead -- a fixed cost, flat"
        "\nin n -- dominates and the GPU never gets to work; above it arithmetic"
        "\ndominates and the cpu columns grow while the fast column stays flat."
        "\nPick profiles per workload accordingly, but keep published numbers on"
        "\n'reference' regardless (quant/numerics.py)."
    )


if __name__ == "__main__":
    main()
