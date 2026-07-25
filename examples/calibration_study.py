"""Run a small, reproducible completeness and false-positive study.

This is intentionally a demonstration-scale experiment. Increase ``--repeats``
and ``--draws`` for a scientific calibration run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asterodetect import (
    AsteroScaleSamples,
    AstrophysicalInjectionFactory,
    Detector,
    ObservationModel,
    build_detection_study,
    evaluate_injections,
)
from asterodetect.asteroscale import ASTERO_SCALE_PARAMETERS


def solar_like_samples(draws: int = 512, seed: int = 123) -> AsteroScaleSamples:
    """Return correlated demonstration samples around a solar target.

    Parameters
    ----------
    draws
        Number of aligned sample rows.
    seed
        Random seed.

    Returns
    -------
    AsteroScaleSamples
        Synthetic correlated sample cloud.
    """

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


def run_study(repeats: int, draws: int, seed: int) -> dict[str, object]:
    """Run the demonstration completeness study.

    Parameters
    ----------
    repeats
        Stochastic injections per grid coordinate.
    draws
        Prior samples per detector run.
    seed
        Root random seed.

    Returns
    -------
    dict
        JSON-serializable calibration summary.
    """

    samples = solar_like_samples(max(draws, 128), seed)
    factory = AstrophysicalInjectionFactory(
        samples,
        duration_days=27.4,
        cadence_seconds=120.0,
    )
    cases = build_detection_study(
        {
            "white_noise": [0.05, 0.2],
            "duration_days": [27.4],
            "dilution": [1.0],
        },
        factory,
        oscillation_amplitudes=[0.1, 0.3, 1.0],
        repeats=repeats,
        seed=seed,
    )
    detector = Detector(
        draws=draws,
        observation=ObservationModel(integration_time_seconds=120.0),
    )
    calibration = evaluate_injections(detector, cases, seed=seed + 1)
    metrics = calibration.detection_metrics(threshold=0.5)
    by_amplitude = {}
    for amplitude, subset in calibration.group_by("amplitude_scale").items():
        amplitude_metrics = subset.detection_metrics(threshold=0.5)
        by_amplitude[str(amplitude)] = {
            "cases": len(subset.recoveries),
            "completeness": _finite_or_none(amplitude_metrics.completeness),
            "false_positive_rate": _finite_or_none(
                amplitude_metrics.false_positive_rate
            ),
            "binary_brier_score": amplitude_metrics.binary_brier_score,
        }
    return {
        "cases": len(cases),
        "draws": draws,
        "repeats": repeats,
        "seed": seed,
        "confusion_matrix": calibration.confusion_matrix.tolist(),
        "accuracy": calibration.accuracy,
        "multiclass_brier_score": calibration.multiclass_brier_score,
        "oscillation_detection": {
            "threshold": metrics.threshold,
            "completeness": metrics.completeness,
            "false_positive_rate": metrics.false_positive_rate,
            "precision": metrics.precision,
            "binary_brier_score": metrics.binary_brier_score,
            "expected_calibration_error": metrics.expected_calibration_error,
        },
        "by_amplitude_scale": by_amplitude,
    }


def _finite_or_none(value: float) -> float | None:
    """Convert a finite value to float and non-finite values to ``None``."""

    return float(value) if np.isfinite(value) else None


def main() -> None:
    """Run the command-line calibration-study example."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    summary = run_study(arguments.repeats, arguments.draws, arguments.seed)
    rendered = json.dumps(summary, indent=2)
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
