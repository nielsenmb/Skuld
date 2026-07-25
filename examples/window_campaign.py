"""Compare uninterrupted and TESS-like observing windows.

Each pair shares the same stochastic stellar realization and inference seed.
The decision threshold is selected using only uninterrupted tuning cases, then
applied unchanged to held-out uninterrupted and gapped validation cases.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from asterodetect import (
    AstrophysicalInjectionFactory,
    CalibrationResult,
    Detector,
    InjectionCase,
    NuisancePrior,
    ObservationModel,
    Recovery,
    select_detection_threshold,
    summarize_recoveries,
)
from astrophysical_campaign import REGIMES, regime_samples


def build_window_campaign(
    *,
    profile: str,
    repeats: int,
    seed: int,
) -> tuple[InjectionCase, ...]:
    """Build paired continuous and gapped multi-regime injections.

    Parameters
    ----------
    profile
        ``"checkpoint"`` uses one noise level and amplitudes 0.3 and 1.0.
        ``"standard"`` uses both noise levels and also includes amplitude 0.5.
    repeats
        Independent stochastic realizations per exact astrophysical cell.
    seed
        Root simulation seed.

    Returns
    -------
    tuple
        Paired injections distinguished by ``window_profile`` metadata.
    """

    if profile not in {"checkpoint", "standard"}:
        raise ValueError("profile must be checkpoint or standard")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("repeats must be an integer of at least two")

    amplitudes = (0.3, 1.0) if profile == "checkpoint" else (0.3, 0.5, 1.0)
    classes = (("noise", 0.0), ("granulation", 0.0)) + tuple(
        ("oscillation", amplitude) for amplitude in amplitudes
    )
    cells = []
    for regime, parameters in REGIMES.items():
        noise_levels = (
            parameters["white_noise"][:1]
            if profile == "checkpoint"
            else parameters["white_noise"]
        )
        cells.extend(
            (regime, duration, noise, truth, amplitude)
            for duration, noise, (truth, amplitude) in product(
                (27.4, 90.0),
                noise_levels,
                classes,
            )
        )

    child_seeds = np.random.SeedSequence(seed).spawn(len(cells) * repeats)
    factories = {
        regime: AstrophysicalInjectionFactory(
            regime_samples(parameters, seed=seed + index),
            cadence_seconds=120.0,
        )
        for index, (regime, parameters) in enumerate(REGIMES.items())
    }
    cases = []
    seed_index = 0
    for cell_index, (regime, duration, noise, truth, amplitude) in enumerate(cells):
        for repeat in range(repeats):
            pair_id = f"cell={cell_index}-repeat={repeat}"
            parameters = {
                "truth": truth,
                "duration_days": duration,
                "white_noise": noise,
                "amplitude_scale": amplitude,
                "window_seed": seed + cell_index,
            }
            child_seed = child_seeds[seed_index]
            seed_index += 1
            for window_profile in ("continuous", "tess-like"):
                generated = factories[regime](
                    f"{pair_id}-window={window_profile}",
                    {
                        **parameters,
                        "window_profile": window_profile,
                    },
                    np.random.default_rng(child_seed),
                )
                metadata = dict(generated.metadata or {})
                metadata.update(
                    stellar_regime=regime,
                    repeat=repeat,
                    pair_id=pair_id,
                )
                cases.append(
                    InjectionCase(
                        name=generated.name,
                        truth=generated.truth,
                        spectrum=generated.spectrum,
                        stellar_constraints=generated.stellar_constraints,
                        metadata=MappingProxyType(metadata),
                    )
                )
    return tuple(cases)


def evaluate_paired_windows(
    detector: Detector,
    cases: tuple[InjectionCase, ...],
    *,
    seed: int,
) -> CalibrationResult:
    """Evaluate each window pair with a shared inference random seed."""

    pair_ids = tuple(
        dict.fromkeys(str(case.metadata["pair_id"]) for case in cases)
    )
    child_seeds = np.random.SeedSequence(seed).spawn(len(pair_ids))
    seeds = dict(zip(pair_ids, child_seeds, strict=True))
    recoveries = []
    for case in cases:
        pair_id = str(case.metadata["pair_id"])
        result = detector.run(
            case.spectrum,
            case.stellar_constraints,
            rng=np.random.default_rng(seeds[pair_id]),
        )
        recoveries.append(Recovery(case, result))
    return summarize_recoveries(recoveries)


def _metrics(
    calibration: CalibrationResult,
    threshold: float,
) -> dict[str, Any]:
    """Return JSON-safe binary detection metrics."""

    metrics = calibration.detection_metrics(threshold=threshold)
    return {
        "cases": len(calibration.recoveries),
        "threshold": threshold,
        "true_positive_rate": _finite_or_none(metrics.completeness),
        "false_positive_rate": _finite_or_none(metrics.false_positive_rate),
        "precision": _finite_or_none(metrics.precision),
        "binary_brier_score": metrics.binary_brier_score,
    }


def _finite_or_none(value: float) -> float | None:
    """Convert finite values to floats and non-finite values to ``None``."""

    return float(value) if np.isfinite(value) else None


def _group_metrics(
    calibration: CalibrationResult,
    key: str,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Return metrics grouped by one injection metadata coordinate."""

    return {
        str(value): _metrics(group, threshold)
        for value, group in calibration.group_by(key).items()
    }


def _paired_probability_summary(
    calibration: CalibrationResult,
    threshold: float,
) -> dict[str, float | int]:
    """Summarize gapped-minus-continuous probability changes."""

    pairs: dict[str, dict[str, Recovery]] = {}
    for recovery in calibration.recoveries:
        metadata = recovery.case.metadata
        pairs.setdefault(str(metadata["pair_id"]), {})[
            str(metadata["window_profile"])
        ] = recovery
    shifts = []
    detections_lost = 0
    detections_gained = 0
    for pair in pairs.values():
        continuous = pair["continuous"].result.probabilities["oscillation"]
        gapped = pair["tess-like"].result.probabilities["oscillation"]
        shifts.append(gapped - continuous)
        detections_lost += continuous >= threshold and gapped < threshold
        detections_gained += continuous < threshold and gapped >= threshold
    values = np.asarray(shifts, dtype=float)
    return {
        "pairs": len(values),
        "mean_probability_shift": float(np.mean(values)),
        "median_probability_shift": float(np.median(values)),
        "mean_absolute_probability_shift": float(np.mean(np.abs(values))),
        "detections_lost": int(detections_lost),
        "detections_gained": int(detections_gained),
    }


def run_campaign(
    *,
    profile: str,
    repeats: int,
    draws: int,
    seed: int,
    maximum_false_positive_rate: float = 0.05,
) -> dict[str, Any]:
    """Run leakage-safe paired observing-window calibration."""

    cases = build_window_campaign(profile=profile, repeats=repeats, seed=seed)
    tuning_repeats = set(range(repeats // 2))
    tuning = tuple(
        case
        for case in cases
        if case.metadata["repeat"] in tuning_repeats
    )
    validation = tuple(
        case
        for case in cases
        if case.metadata["repeat"] not in tuning_repeats
    )
    detector = Detector(
        draws=draws,
        estimator="adaptive",
        nuisance_prior=NuisancePrior(),
        observation=ObservationModel(integration_time_seconds=120.0),
    )
    tuning_result = evaluate_paired_windows(detector, tuning, seed=seed + 100)
    continuous_tuning = tuning_result.subset(window_profile="continuous")
    selected = select_detection_threshold(
        continuous_tuning,
        np.linspace(0.05, 1.0, 20),
        maximum_false_positive_rate=maximum_false_positive_rate,
    )
    threshold = selected.threshold

    validation_result = evaluate_paired_windows(
        detector,
        validation,
        seed=seed + 1000,
    )
    continuous = validation_result.subset(window_profile="continuous")
    gapped = validation_result.subset(window_profile="tess-like")
    return {
        "profile": profile,
        "draws": draws,
        "repeats": repeats,
        "seed": seed,
        "selected_threshold": threshold,
        "maximum_false_positive_rate": maximum_false_positive_rate,
        "tuning_continuous": _metrics(continuous_tuning, threshold),
        "validation": {
            "continuous": _metrics(continuous, threshold),
            "tess-like": _metrics(gapped, threshold),
        },
        "paired_probability_change": _paired_probability_summary(
            validation_result, threshold
        ),
        "validation_by_amplitude_scale": {
            "continuous": _group_metrics(continuous, "amplitude_scale", threshold),
            "tess-like": _group_metrics(gapped, "amplitude_scale", threshold),
        },
        "validation_by_stellar_regime": {
            "continuous": _group_metrics(continuous, "stellar_regime", threshold),
            "tess-like": _group_metrics(gapped, "stellar_regime", threshold),
        },
        "validation_by_duration_days": {
            "continuous": _group_metrics(continuous, "duration_days", threshold),
            "tess-like": _group_metrics(gapped, "duration_days", threshold),
        },
        "duty_cycle": {
            "continuous": float(
                np.mean(
                    [
                        recovery.case.metadata["duty_cycle"]
                        for recovery in continuous.recoveries
                    ]
                )
            ),
            "tess-like": float(
                np.mean(
                    [
                        recovery.case.metadata["duty_cycle"]
                        for recovery in gapped.recoveries
                    ]
                )
            ),
        },
    }


def main() -> None:
    """Run the command-line observing-window campaign."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("checkpoint", "standard"), default="checkpoint"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--draws", type=int, default=256)
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
