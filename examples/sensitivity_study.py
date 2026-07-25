"""Compare prior-draw convergence and Delta-nu binning choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asterodetect import (
    AsteroScaleSamples,
    AstrophysicalInjectionFactory,
    ObservationModel,
    build_detection_study,
    run_sensitivity_study,
)
from asterodetect.asteroscale import ASTERO_SCALE_PARAMETERS


def solar_like_samples(draws: int = 1024, seed: int = 123):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=draws)
    values = {name: np.ones(draws) for name in ASTERO_SCALE_PARAMETERS}
    values.update(
        numax=3100.0 * np.exp(0.025 * latent),
        dnu=135.1 * np.exp(0.018 * latent),
        FWHM_env=950.0 * np.exp(0.08 * latent),
        A_env=2.1 * np.exp(0.12 * latent),
        A_gran=55.0 * np.exp(-0.10 * latent),
        b_gran_low=760.0 * np.exp(0.025 * latent),
        b_gran_high=2850.0 * np.exp(0.025 * latent),
    )
    return AsteroScaleSamples(values)


def run_study(repeats: int, seed: int) -> dict[str, object]:
    samples = solar_like_samples(seed=seed)
    factory = AstrophysicalInjectionFactory(
        samples, duration_days=27.4, cadence_seconds=120.0
    )
    cases = build_detection_study(
        {"white_noise": [0.1], "duration_days": [27.4], "dilution": [1.0]},
        factory,
        oscillation_amplitudes=[0.3, 1.0],
        repeats=1,
        seed=seed,
    )
    study = run_sensitivity_study(
        cases,
        draw_counts=[128, 512, 2048],
        dnu_scales=[0.5, 1.0, 2.0],
        repeats=repeats,
        seed=seed + 1,
        observation=ObservationModel(integration_time_seconds=120.0),
    )
    return {
        "cases": len(cases),
        "inference_repeats": repeats,
        "summaries": [
            {
                "draws": item.draws,
                "dnu_scale": item.dnu_scale,
                "evaluations": item.evaluations,
                "mean_oscillation_probability": item.mean_oscillation_probability,
                "oscillation_probability_std": item.oscillation_probability_std,
                "classification_accuracy": item.classification_accuracy,
                "minimum_median_ess_fraction": item.minimum_median_ess_fraction,
                "maximum_median_log_evidence_standard_error": (
                    item.maximum_median_log_evidence_standard_error
                ),
            }
            for item in study.summaries()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    rendered = json.dumps(run_study(arguments.repeats, arguments.seed), indent=2)
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
