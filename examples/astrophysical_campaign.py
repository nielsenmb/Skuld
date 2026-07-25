"""Run a multi-regime astrophysical injection-recovery campaign.

The default checkpoint profile is large enough to expose regime-dependent
behavior while remaining practical on a laptop. The standard profile adds
dilution and granulation-relation offsets and is intended for longer runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from asterodetect import (
    AsteroScaleSamples,
    AstrophysicalInjectionFactory,
    CalibrationResult,
    Detector,
    NuisancePrior,
    ObservationModel,
    build_regime_detection_study,
    evaluate_injections,
    select_detection_threshold,
    split_injections,
)
from asterodetect.asteroscale import ASTERO_SCALE_PARAMETERS


REGIMES = {
    "dwarf": {
        "numax": 3100.0,
        "dnu": 135.1,
        "FWHM_env": 950.0,
        "A_env": 2.1,
        "A_gran": 55.0,
        "b_gran_low": 760.0,
        "b_gran_high": 2850.0,
        "white_noise": (0.05, 0.2),
    },
    "subgiant": {
        "numax": 800.0,
        "dnu": 50.0,
        "FWHM_env": 350.0,
        "A_env": 5.0,
        "A_gran": 150.0,
        "b_gran_low": 190.0,
        "b_gran_high": 720.0,
        "white_noise": (0.2, 0.8),
    },
    "low_luminosity_rgb": {
        "numax": 100.0,
        "dnu": 9.0,
        "FWHM_env": 40.0,
        "A_env": 30.0,
        "A_gran": 600.0,
        "b_gran_low": 25.0,
        "b_gran_high": 90.0,
        "white_noise": (2.0, 8.0),
    },
}

SPLIT_FIELDS = (
    "truth",
    "stellar_regime",
    "duration_days",
    "white_noise",
    "amplitude_scale",
    "dilution",
    "granulation_scale",
)

AMPLITUDE_PRIOR_CANDIDATES = {
    "single": NuisancePrior(),
    "mixture-0.10-at-0.3": NuisancePrior(
        envelope_suppressed_fraction=0.10,
        envelope_suppression_factor=0.3,
    ),
    "mixture-0.25-at-0.3": NuisancePrior(
        envelope_suppressed_fraction=0.25,
        envelope_suppression_factor=0.3,
    ),
    "mixture-0.50-at-0.3": NuisancePrior(
        envelope_suppressed_fraction=0.50,
        envelope_suppression_factor=0.3,
    ),
    "mixture-0.25-at-0.5": NuisancePrior(
        envelope_suppressed_fraction=0.25,
        envelope_suppression_factor=0.5,
    ),
}


def regime_samples(
    parameters: dict[str, Any],
    *,
    draws: int = 2048,
    seed: int = 123,
) -> AsteroScaleSamples:
    """Construct a correlated AsteroScale-like sample cloud for one regime.

    Parameters
    ----------
    parameters
        Regime medians, including all spectral quantities used below.
    draws
        Number of intact correlated rows.
    seed
        Random seed.

    Returns
    -------
    AsteroScaleSamples
        Synthetic joint sample cloud for recovery.
    """

    rng = np.random.default_rng(seed)
    stellar_latent = rng.normal(size=draws)
    amplitude_latent = (
        -0.45 * stellar_latent
        + np.sqrt(1.0 - 0.45**2) * rng.normal(size=draws)
    )
    values = {name: np.ones(draws) for name in ASTERO_SCALE_PARAMETERS}
    values.update(
        numax=parameters["numax"] * np.exp(0.04 * stellar_latent),
        dnu=parameters["dnu"] * np.exp(0.025 * stellar_latent),
        FWHM_env=parameters["FWHM_env"] * np.exp(0.10 * stellar_latent),
        A_env=parameters["A_env"] * np.exp(0.16 * amplitude_latent),
        A_gran=parameters["A_gran"] * np.exp(-0.12 * stellar_latent),
        b_gran_low=parameters["b_gran_low"] * np.exp(0.04 * stellar_latent),
        b_gran_high=parameters["b_gran_high"] * np.exp(0.04 * stellar_latent),
    )
    return AsteroScaleSamples(values)


def build_campaign(
    *,
    profile: str,
    repeats: int,
    seed: int,
) -> tuple:
    """Build the configured multi-regime injection population.

    Parameters
    ----------
    profile
        ``"checkpoint"`` or ``"standard"`` campaign size.
    repeats
        Stochastic realizations per grid cell.
    seed
        Root random seed.

    Returns
    -------
    tuple
        Injection cases spanning all configured regimes.
    """

    if profile not in {"checkpoint", "standard"}:
        raise ValueError("profile must be checkpoint or standard")
    factories = {}
    axes = {}
    for index, (name, parameters) in enumerate(REGIMES.items()):
        samples = regime_samples(parameters, seed=seed + index)
        factories[name] = AstrophysicalInjectionFactory(
            samples,
            duration_days=27.4,
            cadence_seconds=120.0,
            white_noise=parameters["white_noise"][0],
        )
        axes[name] = {
            "duration_days": [27.4, 90.0],
            "white_noise": list(parameters["white_noise"]),
            "dilution": [1.0],
            "granulation_scale": [1.0],
        }
        if profile == "standard":
            axes[name]["dilution"] = [0.7, 1.0]
            axes[name]["granulation_scale"] = [0.7, 1.3]
    amplitudes = [0.3, 1.0]
    if profile == "standard":
        amplitudes = [0.3, 0.5, 1.0]
    return build_regime_detection_study(
        factories,
        axes,
        oscillation_amplitudes=amplitudes,
        repeats=repeats,
        seed=seed + 10,
    )


def _metrics(
    calibration: CalibrationResult,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return a JSON-safe detection summary."""

    metrics = calibration.detection_metrics(threshold=threshold)
    return {
        "cases": len(calibration.recoveries),
        "threshold": threshold,
        "accuracy": calibration.accuracy,
        "binary_accuracy": (
            metrics.true_positives + metrics.true_negatives
        ) / len(calibration.recoveries),
        "multiclass_brier_score": calibration.multiclass_brier_score,
        "completeness": _finite_or_none(metrics.completeness),
        "false_positive_rate": _finite_or_none(metrics.false_positive_rate),
        "precision": _finite_or_none(metrics.precision),
        "binary_brier_score": metrics.binary_brier_score,
        "expected_calibration_error": metrics.expected_calibration_error,
    }


def _group_metrics(
    calibration: CalibrationResult,
    metadata_key: str,
    *,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Summarize calibration separately for one metadata coordinate."""

    return {
        str(value): _metrics(subset, threshold=threshold)
        for value, subset in calibration.group_by(metadata_key).items()
    }


def _cross_group_metrics(
    calibration: CalibrationResult,
    first_key: str,
    second_key: str,
    *,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Summarize every populated pair of two metadata coordinates."""

    result = {}
    for first_value, first_subset in calibration.group_by(first_key).items():
        for second_value, subset in first_subset.group_by(second_key).items():
            label = f"{first_key}={first_value},{second_key}={second_value}"
            result[label] = _metrics(subset, threshold=threshold)
    return result


def _finite_or_none(value: float) -> float | None:
    """Convert finite values to floats and non-finite values to ``None``."""

    return float(value) if np.isfinite(value) else None


def _detector(draws: int, prior: NuisancePrior) -> Detector:
    """Construct the common adaptive detector for one amplitude prior."""

    return Detector(
        draws=draws,
        estimator="adaptive",
        nuisance_prior=prior,
        observation=ObservationModel(integration_time_seconds=120.0),
        dnu_scale=1.0,
    )


def _prior_description(prior: NuisancePrior) -> dict[str, float]:
    """Return JSON-safe amplitude-mixture hyperparameters."""

    return {
        "normal_log_scatter": prior.envelope_log_scatter,
        "suppressed_fraction": prior.envelope_suppressed_fraction,
        "suppression_factor": prior.envelope_suppression_factor,
        "suppressed_log_scatter": prior.envelope_suppressed_log_scatter,
    }


def run_campaign(
    *,
    profile: str,
    repeats: int,
    draws: int,
    seed: int,
    maximum_false_positive_rate: float = 0.05,
) -> dict[str, Any]:
    """Run a multi-regime adaptive injection-recovery campaign.

    Parameters
    ----------
    profile
        Campaign size, either ``"checkpoint"`` or ``"standard"``.
    repeats
        Stochastic injections per grid coordinate.
    draws
        Final adaptive importance draws per model and case.
    seed
        Root random seed.
    maximum_false_positive_rate
        Tuning-population constraint used to select the decision threshold.

    Returns
    -------
    dict
        JSON-serializable overall and grouped calibration metrics.
    """

    cases = build_campaign(profile=profile, repeats=repeats, seed=seed)
    split = split_injections(
        cases,
        validation_fraction=0.5,
        stratify_by=SPLIT_FIELDS,
        seed=seed + 30,
    )
    candidates = {}
    candidate_calibrations = {}
    candidate_metrics = {}
    thresholds = np.linspace(0.05, 1.0, 20)
    candidate_order = {
        label: index for index, label in enumerate(AMPLITUDE_PRIOR_CANDIDATES)
    }
    for label, prior in AMPLITUDE_PRIOR_CANDIDATES.items():
        calibration = evaluate_injections(
            _detector(draws, prior),
            split.tuning,
            seed=seed + 100,
        )
        selected = select_detection_threshold(
            calibration,
            thresholds,
            maximum_false_positive_rate=maximum_false_positive_rate,
        )
        candidate_calibrations[label] = calibration
        candidate_metrics[label] = selected
        candidates[label] = {
            "prior": _prior_description(prior),
            "tuning": _metrics(calibration, threshold=selected.threshold),
        }

    def candidate_score(label: str) -> tuple[float, float, float, int]:
        """Rank candidate priors using tuning results only."""

        metrics = candidate_metrics[label]
        return (
            metrics.completeness,
            -metrics.binary_brier_score,
            -metrics.false_positive_rate,
            -candidate_order[label],
        )

    selected_label = max(candidate_metrics, key=candidate_score)
    selected_prior = AMPLITUDE_PRIOR_CANDIDATES[selected_label]
    threshold = candidate_metrics[selected_label].threshold
    tuning = candidate_calibrations[selected_label]
    validation = evaluate_injections(
        _detector(draws, selected_prior),
        split.validation,
        seed=seed + 1000,
    )
    return {
        "profile": profile,
        "estimator": "adaptive",
        "draws": draws,
        "repeats": repeats,
        "seed": seed,
        "maximum_false_positive_rate": maximum_false_positive_rate,
        "amplitude_prior_candidates": candidates,
        "selected_amplitude_prior": selected_label,
        "selected_amplitude_prior_parameters": _prior_description(selected_prior),
        "tuning": _metrics(tuning, threshold=threshold),
        "validation": _metrics(validation, threshold=threshold),
        "selected_threshold": threshold,
        "validation_at_default_threshold": _metrics(validation),
        "validation_at_default_by_amplitude_scale": _group_metrics(
            validation, "amplitude_scale"
        ),
        "validation_by_stellar_regime": _group_metrics(
            validation, "stellar_regime", threshold=threshold
        ),
        "validation_by_duration_days": _group_metrics(
            validation, "duration_days", threshold=threshold
        ),
        "validation_by_amplitude_scale": _group_metrics(
            validation, "amplitude_scale", threshold=threshold
        ),
        "validation_by_regime_and_amplitude": _cross_group_metrics(
            validation,
            "stellar_regime",
            "amplitude_scale",
            threshold=threshold,
        ),
        "validation_by_dilution": _group_metrics(
            validation, "dilution", threshold=threshold
        ),
        "validation_by_granulation_scale": _group_metrics(
            validation, "granulation_scale", threshold=threshold
        ),
    }


def main() -> None:
    """Run the command-line astrophysical campaign."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("checkpoint", "standard"), default="checkpoint"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--draws", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--maximum-false-positive-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_campaign(
        profile=arguments.profile,
        repeats=arguments.repeats,
        draws=arguments.draws,
        seed=arguments.seed,
        maximum_false_positive_rate=arguments.maximum_false_positive_rate,
    )
    rendered = json.dumps(result, indent=2)
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
