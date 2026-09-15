"""Monte Carlo stage (docs/quant-model.md section 5.2).

Samples batched risk draws for the families in docs/framework.md section 1 and
feeds them to the same FirmValueModel / optimize_policy harness quant/static.py
uses with its four explicit states. The distributions are illustrative, not
calibrated -- which is exactly why quant/threshold.py reports break-even
parameter values rather than treating any single answer as a point estimate.

Results are reported across several seeds with a confidence interval. The
severity distributions are heavy-tailed, so a single-seed answer quoted to four
decimals overstates what the sample supports.

Sampling follows the rule in quant/numerics.py: uniforms are drawn on CPU in
float64 and the inverse-CDF transform is applied there, with the cast to the
profile's compute dtype happening last. Two consequences worth knowing -- the
scenario set is identical under every profile, so a profile change alters only
arithmetic rounding; and the tail resolution of the heavy-tailed severities
survives, which casting the uniforms first would destroy exactly where it
matters most.
"""

import statistics
from dataclasses import dataclass, field

import torch

from quant.cli import parse_profile, print_header
from quant.model import (
    FirmValueModel,
    GrcAlphas,
    GrcBudgets,
    PolicyResult,
    RiskDraw,
    optimize_policy,
)
from quant.numerics import DEFAULT_PROFILE, NumericsProfile
from quant.params import DEFAULT_SEEDS, DEFAULTS, FirmParams, SamplerParams

# torch.quantile refuses tensors above 2**24 elements and its result differs
# across backends; loss_quantiles sorts instead once a batch gets that large.
QUANTILE_LIMIT = 2**24


def sample_risk_draw(
    n_paths: int,
    sampler: SamplerParams,
    generator: torch.Generator,
    profile: NumericsProfile = DEFAULT_PROFILE,
) -> RiskDraw:
    def exponential(mean: float) -> torch.Tensor:
        uniform = profile.draw_uniform((n_paths,), generator)
        return profile.to(-mean * torch.log1p(-uniform))

    def bernoulli(probability: float) -> torch.Tensor:
        uniform = profile.draw_uniform((n_paths,), generator)
        return profile.to((uniform < probability).to(profile.rng_dtype))

    return RiskDraw(
        credit_loss=exponential(sampler.credit_loss_mean),
        op_occurs=bernoulli(sampler.op_probability),
        op_severity=exponential(sampler.op_severity_mean),
        compliance_occurs=bernoulli(sampler.compliance_probability),
        compliance_severity=profile.full((n_paths,), sampler.compliance_severity),
        base_weight=profile.ones(n_paths),
        compliance_base_probability=sampler.compliance_probability,
    )


def build_model(firm: FirmParams = DEFAULTS.firm, **overrides) -> FirmValueModel:
    params = dict(
        initial_equity=firm.initial_equity,
        financing_convexity=firm.financing_convexity,
        distress_reference=firm.distress_reference,
        production_scale=firm.production_scale,
        production_curvature=firm.production_curvature,
        alphas=GrcAlphas(),
    )
    params.update(overrides)
    return FirmValueModel(**params)


def run_policy_search(
    model: FirmValueModel,
    n_paths: int = 8192,
    sampler: SamplerParams | None = None,
    seed: int = 0,
    n_steps: int = 3000,
    profile: NumericsProfile = DEFAULT_PROFILE,
) -> PolicyResult:
    """One sample-average-approximation solve: draw a batch, optimize against it."""
    generator = profile.generator(seed)
    draw = sample_risk_draw(n_paths, sampler or DEFAULTS.sampler, generator, profile)
    return optimize_policy(model, draw, n_steps=n_steps, profile=profile)


@dataclass
class SeedStudy:
    """Policy results across independent samples, so the sampling error in the
    answer is visible rather than hidden behind a fixed default seed."""

    results: list[PolicyResult] = field(default_factory=list)

    def interval(self, attribute: str) -> tuple[float, float]:
        """Mean and 95% confidence half-width for one reported quantity."""
        values = [getattr(result, attribute) for result in self.results]
        mean = statistics.fmean(values)
        if len(values) < 2:
            return mean, 0.0
        return mean, 1.96 * statistics.stdev(values) / len(values) ** 0.5


def run_seed_study(
    model: FirmValueModel,
    n_paths: int = 8192,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    sampler: SamplerParams | None = None,
    n_steps: int = 3000,
    profile: NumericsProfile = DEFAULT_PROFILE,
) -> SeedStudy:
    return SeedStudy(
        [
            run_policy_search(
                model,
                n_paths=n_paths,
                sampler=sampler,
                seed=seed,
                n_steps=n_steps,
                profile=profile,
            )
            for seed in seeds
        ]
    )


def loss_quantiles(
    draw: RiskDraw,
    result: PolicyResult,
    alphas: GrcAlphas,
    quantiles: tuple[float, ...],
    profile: NumericsProfile = DEFAULT_PROFILE,
) -> list[float]:
    """Distribution of mitigated loss at the chosen policy -- the tail that the
    convex financing premium actually prices."""
    budgets = GrcBudgets(
        credit=profile.tensor(result.credit),
        operational=profile.tensor(result.operational),
        compliance=profile.tensor(result.compliance),
    )
    losses = draw.mitigated_loss(budgets, alphas)
    if losses.numel() <= QUANTILE_LIMIT:
        return [torch.quantile(losses, q).item() for q in quantiles]
    ordered = losses.sort().values
    last = ordered.numel() - 1
    return [ordered[min(last, int(q * last))].item() for q in quantiles]


def main(profile: NumericsProfile | None = None) -> None:
    profile = profile or parse_profile(__doc__.splitlines()[0])
    print_header(profile)

    n_paths = 8192
    model = build_model()
    study = run_seed_study(model, n_paths=n_paths, profile=profile)

    print(f"Monte Carlo policy search: {n_paths} paths x {len(DEFAULT_SEEDS)} seeds")
    print(f"{'quantity':>22} | {'mean':>9} | {'95% CI':>13}")
    for label, attribute in (
        ("credit GRC", "credit"),
        ("operational GRC", "operational"),
        ("compliance GRC", "compliance"),
        ("total GRC", "total"),
        ("firm value", "value"),
        ("constrained fraction", "constrained_fraction"),
    ):
        mean, half_width = study.interval(attribute)
        print(f"{label:>22} | {mean:>9.3f} | +/- {half_width:>9.3f}")

    generator = profile.generator(DEFAULT_SEEDS[0])
    draw = sample_risk_draw(n_paths, DEFAULTS.sampler, generator, profile)
    quantiles = (0.5, 0.9, 0.99)
    values = loss_quantiles(draw, study.results[0], model.alphas, quantiles, profile)
    print("\nMitigated loss distribution at the chosen policy (seed 0)")
    print("  " + "  ".join(f"p{int(q * 100)}={v:.2f}" for q, v in zip(quantiles, values)))
    print("\nPrecision note: budgets are reported to 3 decimals because the")
    print("confidence interval above is wider than the 4th.")


if __name__ == "__main__":
    main()
