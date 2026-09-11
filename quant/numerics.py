"""Device, dtype and RNG policy — the one place numerics decisions are made.

Replaces the earlier quant/device.py, which resolved a device and derived a
dtype from it. That was too coarse in both directions: it forced float64 on
every workload because *one* test needs it, and it let modules construct
tensors without a dtype at all, which silently produced float32 (see the
"silent float32" note below).

A NumericsProfile is passed down — into EnvConfig, into solvers, into
samplers — and never looked up globally. One `profile.tensor(...)` per
constant makes an un-dtyped tensor structurally impossible.

Three rules this module exists to enforce:

1. **Randomness is always drawn on CPU in float64, then cast.** A
   torch.Generator on MPS uses a different algorithm from CPU's: same seed,
   different numbers. Managing that is hopeless; eliminating it is easy. Draw
   on CPU and the *scenario set is bit-identical across devices*, so a profile
   change alters arithmetic rounding only, never which paths were averaged
   over. That turns "does float32 on MPS agree with float64 on CPU" from a
   confound into a test (tests/test_numerics.py::test_profiles_agree).

   Draw with `draw_uniform`, do the inverse-CDF transform at full precision,
   and call `to()` last. Casting the uniforms first throws away tail
   resolution exactly where the heavy-tailed severities live.

2. **Reductions accumulate in `accum_dtype`.** Under a float32 compute dtype,
   summing 10^6 path values loses more than the arithmetic does.

3. **Published numbers come from REFERENCE, and say so.** Optimizer
   trajectories differ across devices even when the scenarios do not (fused
   vs non-fused Adam, MPS reduction order), so every entry point prints
   `fingerprint()` in its header.

Which profile for which workload, per docs/quant-model.md section 5:

    analytic oracles, published tables    REFERENCE
    perfect-information bound             REFERENCE (CPU_FAST at scale)
    grid / fitted value iteration         FAST
    RL rollouts and critic training       FAST
    threshold / sensitivity studies       CPU_FAST

The crossover is measured, not assumed -- run scripts/bench_profiles.py to
re-measure on other hardware. On an M4 Pro under torch 2.14 the shape is: MPS
carries a fixed dispatch cost of roughly 160 microseconds per optimizer step
that does not shrink with the batch, while CPU cost grows with it. So the
accelerator loses on small batches by a constant and wins on large ones by a
widening margin, crossing over around 4k elements for elementwise path work
and around 1k once a small network is in the loop. quant/static.py runs at
n=4 and quant/simulate.py at n=8192; the dynamic model's rollouts and grid
sweeps sit orders of magnitude above both.

Measure warm, not cold. MPS pays a one-off compilation cost on first launch,
tens of milliseconds, and timing a few hundred steps without discarding it is
how this repo previously concluded that MPS was twice as slow at 8192 paths.
Anything short-lived enough for that cost to matter is also too short to be
worth moving off the CPU.
"""

import functools
import subprocess
from dataclasses import dataclass

import torch

CPU = torch.device("cpu")
MPS = torch.device("mps")


@dataclass(frozen=True)
class NumericsProfile:
    """Where arithmetic happens, in what precision, and how randomness is drawn.

    `dtype` is the compute dtype; `accum_dtype` is used for reductions and must
    be at least as wide. `rng_device` is always CPU and `rng_dtype` always
    float64 — see rule 1 in the module docstring. `deterministic` records
    whether this profile promises run-to-run reproducibility; MPS does not.
    """

    name: str
    device: torch.device
    dtype: torch.dtype
    accum_dtype: torch.dtype
    rng_device: torch.device = CPU
    rng_dtype: torch.dtype = torch.float64
    deterministic: bool = True

    # -- construction ----------------------------------------------------

    def tensor(self, data, *, dtype: torch.dtype | None = None, requires_grad: bool = False):
        return torch.tensor(
            data, device=self.device, dtype=dtype or self.dtype, requires_grad=requires_grad
        )

    def zeros(self, *shape: int, requires_grad: bool = False):
        return torch.zeros(
            *shape, device=self.device, dtype=self.dtype, requires_grad=requires_grad
        )

    def ones(self, *shape: int):
        return torch.ones(*shape, device=self.device, dtype=self.dtype)

    def full(self, shape: tuple[int, ...], value: float):
        return torch.full(shape, value, device=self.device, dtype=self.dtype)

    def to(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move and cast an existing tensor onto this profile."""
        return tensor.to(device=self.device, dtype=self.dtype)

    # -- randomness ------------------------------------------------------

    def generator(self, seed: int) -> torch.Generator:
        """A seeded generator on the RNG device (always CPU)."""
        return torch.Generator(device=self.rng_device).manual_seed(seed)

    def draw_uniform(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        """Raw uniforms at full precision on the RNG device.

        Deliberately *not* moved onto the compute device: transform first
        (inverse CDF, thresholding), then call `to()` on the result. Same seed
        gives the same draw under every profile.
        """
        return torch.rand(
            shape, generator=generator, dtype=self.rng_dtype, device=self.rng_device
        )

    # -- reductions ------------------------------------------------------

    def sum(self, tensor: torch.Tensor, dim: int | None = None) -> torch.Tensor:
        """Sum in `accum_dtype`, return in the input dtype."""
        wide = tensor.to(self.accum_dtype)
        total = wide.sum() if dim is None else wide.sum(dim=dim)
        return total.to(tensor.dtype)

    # -- provenance ------------------------------------------------------

    def fingerprint(self) -> str:
        """Identifies a run well enough to reproduce it, for docs and headers."""
        return f"{self.name}|{self.device.type}|{_dtype_name(self.dtype)}|torch {torch.__version__}|git {_git_sha()}"

    def available(self) -> bool:
        return self.device.type != "mps" or torch.backends.mps.is_available()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@functools.cache
def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


REFERENCE = NumericsProfile(
    name="reference", device=CPU, dtype=torch.float64, accum_dtype=torch.float64
)
CPU_FAST = NumericsProfile(
    name="cpu-fast", device=CPU, dtype=torch.float32, accum_dtype=torch.float64
)
FAST = NumericsProfile(
    name="fast", device=MPS, dtype=torch.float32, accum_dtype=torch.float32,
    deterministic=False,
)

PROFILES = {profile.name: profile for profile in (REFERENCE, CPU_FAST, FAST)}
DEFAULT_PROFILE = REFERENCE


def get_profile(name: str = DEFAULT_PROFILE.name) -> NumericsProfile:
    """Look up a profile by name, refusing one whose device is unavailable.

    Deliberately no silent fallback: a run that quietly drops from MPS to CPU
    reports timings and tolerances for a configuration nobody chose.
    """
    try:
        profile = PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}") from None
    if not profile.available():
        raise RuntimeError(
            f"profile {name!r} needs {profile.device.type}, which is not available here"
        )
    return profile


if __name__ == "__main__":
    for profile in PROFILES.values():
        mark = " " if profile.available() else "  (unavailable)"
        print(f"{profile.name:>10}: {profile.fingerprint()}{mark}")
