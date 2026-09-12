"""Multi-seed studies: turning single-run point estimates into intervals.

Every number this project has published so far comes from one draw of the
scenarios and one network initialization. That is exactly the practice
`quant/simulate.py` was corrected for in the static model -- quoting four
decimals on an estimate whose sampling error is wider than the fourth -- and
the dynamic model reintroduced it, because a rollout looks deterministic once
the common random numbers are fixed. It is deterministic. It is also one
sample.

Two things vary here, and they are different sources of error:

    scenario seed   which paths the world takes. Averaging over it is ordinary
                    Monte Carlo error.
    solver seed     where the optimizer starts. Averaging over it is the
                    question of whether the answer is a property of the model
                    or of one lucky initialization.

Both move together, so a spread here bounds both at once.

**Scored out of sample.** A policy trained on a set of paths and graded on the
same paths is marking its own homework, and the gradient solvers here fit a
free parameter per path in one case. Training and evaluation use disjoint
scenario draws. Measured, the in-sample bias is small -- 0.26% at eight
quarters, 1.17% for the neural policy at thirty-two -- but it is a bias, and
knowing its size is the only way to know it is small.
"""

import statistics
from dataclasses import dataclass, field

from quant.env.env import EvalResult, FirmEnv, evaluate
from quant.env.shocks import CommonRandomNumbers
from quant.numerics import NumericsProfile
from quant.solvers.neural import train_pathwise
from quant.solvers.pathwise import optimize_constant

# Offset between a run's training and evaluation scenarios. Large enough that
# no seed's training set is another's test set.
HELD_OUT_OFFSET = 10_000


@dataclass
class SeedStudy:
    """One solver, across seeds."""

    name: str
    held_out: list[float] = field(default_factory=list)
    in_sample: list[float] = field(default_factory=list)
    diagnostics: list[EvalResult] = field(default_factory=list)

    def median(self) -> float:
        return statistics.median(self.held_out)

    def spread(self) -> tuple[float, float]:
        return min(self.held_out), max(self.held_out)

    def half_width(self) -> float:
        """95% half-width on the mean. Reported alongside the range because the
        two answer different questions: the range is where a single run might
        land, the half-width is how well the median is pinned down."""
        if len(self.held_out) < 2:
            return 0.0
        return 1.96 * statistics.stdev(self.held_out) / len(self.held_out) ** 0.5

    def overfit(self) -> float:
        """Mean in-sample advantage, as a share of the held-out value."""
        pairs = zip(self.in_sample, self.held_out)
        return statistics.fmean((a - b) / abs(b) for a, b in pairs)

    def line(self) -> str:
        low, high = self.spread()
        return (
            f"{self.name:>10} | {self.median():>9.3f} | +/- {self.half_width():>6.3f} | "
            f"[{low:>8.3f}, {high:>8.3f}] | {self.overfit():>7.2%}"
        )


def header() -> str:
    return (
        f"{'solver':>10} | {'median':>9} | {'95% CI':>10} | "
        f"{'range over seeds':>20} | {'overfit':>7}"
    )


def run_seeds(
    env: FirmEnv,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    train_paths: int = 512,
    test_paths: int = 4096,
    constant_steps: int = 1500,
    neural_steps: int = 1000,
) -> dict[str, SeedStudy]:
    """Train and score both solvers across seeds, evaluating out of sample."""
    profile: NumericsProfile = env.config.profile
    horizon = env.config.horizon
    studies = {name: SeedStudy(name) for name in ("constant", "neural")}

    for seed in seeds:
        train = CommonRandomNumbers(seed, horizon, train_paths, profile)
        test = CommonRandomNumbers(seed + HELD_OUT_OFFSET, horizon, test_paths, profile)

        policies = {
            "constant": optimize_constant(env, train, n_steps=constant_steps)[0],
            "neural": train_pathwise(env, train, n_steps=neural_steps, seed=seed).policy,
        }
        for name, policy in policies.items():
            scored = evaluate(policy, env, test)
            studies[name].held_out.append(scored.value)
            studies[name].in_sample.append(evaluate(policy, env, train).value)
            studies[name].diagnostics.append(scored)
    return studies


def edge(studies: dict[str, SeedStudy], over: str = "constant", under: str = "neural") -> float:
    """Median advantage of one solver over another, as a share."""
    return studies[under].median() / studies[over].median() - 1.0
