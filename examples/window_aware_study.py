"""Test a target-specific window-aware forward model on noisy RGB stars."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable

import numpy as np

from asterodetect import (
    AstrophysicalInjectionFactory,
    CalibrationResult,
    Detector,
    InjectionCase,
    NuisancePrior,
    ObservationModel,
    Recovery,
    summarize_recoveries,
    tess_observing_window,
)
from astrophysical_campaign import REGIMES, regime_samples


EVALUATION_PROFILES = (
    "continuous-current",
    "gapped-current",
    "gapped-window-aware",
)


def build_window_aware_campaign(
    *,
    repeats: int,
    seed: int,
    truths: Iterable[str] = ("noise", "granulation", "oscillation"),
) -> tuple[InjectionCase, ...]:
    """Build matched continuous and 2.5-day-gapped noisy RGB spectra."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    truth_labels = tuple(str(truth) for truth in truths)
    allowed_truths = {"noise", "granulation", "oscillation"}
    if (
        not truth_labels
        or len(set(truth_labels)) != len(truth_labels)
        or not set(truth_labels).issubset(allowed_truths)
    ):
        raise ValueError(
            "truths must be a non-empty unique subset of noise, "
            "granulation, and oscillation"
        )
    regime = "low_luminosity_rgb"
    parameters = REGIMES[regime]
    factory = AstrophysicalInjectionFactory(
        regime_samples(parameters, seed=seed),
        cadence_seconds=120.0,
    )
    cells = tuple(
        (duration, noise, truth)
        for duration, noise, truth in product(
            (27.4, 90.0),
            parameters["white_noise"],
            truth_labels,
        )
    )
    realization_seeds = np.random.SeedSequence(seed).spawn(
        len(cells) * repeats
    )
    cases: list[InjectionCase] = []
    seed_index = 0
    for cell_index, (duration, noise, truth) in enumerate(cells):
        for repeat in range(repeats):
            pair_id = f"cell={cell_index}-repeat={repeat}"
            realization_seed = realization_seeds[seed_index]
            seed_index += 1
            common = {
                "truth": truth,
                "duration_days": duration,
                "white_noise": noise,
                "amplitude_scale": 1.0,
                "window_seed": seed + cell_index * repeats + repeat,
                "momentum_dump_interval_days": 2.5,
                "momentum_dump_duration_minutes": 30.0,
            }
            for profile in ("continuous", "momentum-dumps"):
                generated = factory(
                    f"{pair_id}-window={profile}",
                    {**common, "window_profile": profile},
                    np.random.default_rng(realization_seed),
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


def _window_for_case(case: InjectionCase):
    """Reconstruct the exact deterministic window recorded for one case."""

    metadata = case.metadata
    if metadata is None:
        raise ValueError("case metadata is required")
    return tess_observing_window(
        metadata["duration_days"],
        metadata["cadence_seconds"],
        profile=metadata["window_profile"],
        momentum_dump_interval_days=metadata[
            "momentum_dump_interval_days"
        ],
        momentum_dump_duration_minutes=metadata[
            "momentum_dump_duration_minutes"
        ],
        rng=metadata["window_seed"],
    )


def evaluate_window_aware_campaign(
    detector: Detector,
    cases: tuple[InjectionCase, ...],
    *,
    seed: int,
) -> CalibrationResult:
    """Evaluate current and window-aware models with paired inference seeds."""

    pair_ids = tuple(
        dict.fromkeys(str(case.metadata["pair_id"]) for case in cases)
    )
    child_seeds = np.random.SeedSequence(seed).spawn(len(pair_ids))
    inference_seeds = dict(zip(pair_ids, child_seeds, strict=True))
    recoveries: list[Recovery] = []
    for case in cases:
        metadata = case.metadata
        if metadata is None:
            raise ValueError("case metadata is required")
        pair_id = str(metadata["pair_id"])
        if metadata["window_profile"] == "continuous":
            evaluations = (("continuous-current", None),)
        else:
            evaluations = (
                ("gapped-current", None),
                ("gapped-window-aware", _window_for_case(case)),
            )
        for profile, window in evaluations:
            result = detector.run(
                case.spectrum,
                case.stellar_constraints,
                rng=np.random.default_rng(inference_seeds[pair_id]),
                observing_window=window,
                window_fft_workers=-1,
            )
            output_metadata = dict(metadata)
            output_metadata["evaluation_profile"] = profile
            output_case = InjectionCase(
                name=f"{case.name}-evaluation={profile}",
                truth=case.truth,
                spectrum=case.spectrum,
                stellar_constraints=case.stellar_constraints,
                metadata=MappingProxyType(output_metadata),
            )
            recoveries.append(Recovery(output_case, result))
    return summarize_recoveries(recoveries)


def _metrics(
    calibration: CalibrationResult,
    threshold: float,
) -> dict[str, Any]:
    """Return JSON-safe binary metrics and median model probabilities."""

    metrics = calibration.detection_metrics(threshold=threshold)
    probabilities = {
        label: np.asarray(
            [
                recovery.result.probabilities[label]
                for recovery in calibration.recoveries
            ]
        )
        for label in ("noise", "granulation", "oscillation")
    }
    return {
        "cases": len(calibration.recoveries),
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "true_negatives": metrics.true_negatives,
        "false_negatives": metrics.false_negatives,
        "true_positive_rate": metrics.completeness,
        "false_positive_rate": metrics.false_positive_rate,
        "binary_brier_score": metrics.binary_brier_score,
        "median_model_probability": {
            label: float(np.median(values))
            for label, values in probabilities.items()
        },
    }


def _paired_model_shift(
    calibration: CalibrationResult,
    *,
    truth: str,
) -> dict[str, float | int]:
    """Compare window-aware and current probabilities on identical PSDs."""

    recoveries = tuple(
        recovery
        for recovery in calibration.recoveries
        if recovery.case.truth == truth
    )
    paired: dict[str, dict[str, Recovery]] = {}
    for recovery in recoveries:
        metadata = recovery.case.metadata
        if metadata is None:
            continue
        profile = str(metadata["evaluation_profile"])
        if profile == "continuous-current":
            continue
        paired.setdefault(str(metadata["pair_id"]), {})[profile] = recovery
    shifts = []
    for pair in paired.values():
        current = pair["gapped-current"].result.probabilities["oscillation"]
        aware = pair[
            "gapped-window-aware"
        ].result.probabilities["oscillation"]
        shifts.append(aware - current)
    values = np.asarray(shifts, dtype=float)
    return {
        "pairs": int(values.size),
        "mean_oscillation_probability_shift": float(np.mean(values)),
        "median_oscillation_probability_shift": float(np.median(values)),
        "mean_absolute_oscillation_probability_shift": float(
            np.mean(np.abs(values))
        ),
    }


def summarize_window_aware_study(
    calibration: CalibrationResult,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Summarize current and window-aware performance."""

    return {
        "threshold": float(threshold),
        "profiles": {
            profile: _metrics(
                calibration.subset(evaluation_profile=profile),
                threshold,
            )
            for profile in EVALUATION_PROFILES
        },
        "window_aware_minus_current_probability": {
            truth: _paired_model_shift(calibration, truth=truth)
            for truth in ("noise", "granulation", "oscillation")
        },
        "by_duration_days": {
            str(duration): {
                profile: _metrics(
                    group.subset(evaluation_profile=profile),
                    threshold,
                )
                for profile in EVALUATION_PROFILES
            }
            for duration, group in calibration.group_by(
                "duration_days"
            ).items()
        },
        "by_white_noise": {
            str(noise): {
                profile: _metrics(
                    group.subset(evaluation_profile=profile),
                    threshold,
                )
                for profile in EVALUATION_PROFILES
            }
            for noise, group in calibration.group_by("white_noise").items()
        },
    }


def run_window_aware_study(
    *,
    repeats: int,
    draws: int,
    seed: int,
    threshold: float = 0.45,
) -> dict[str, Any]:
    """Run the paired noisy-RGB window-aware experiment."""

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    cases = build_window_aware_campaign(repeats=repeats, seed=seed)
    detector = Detector(
        draws=draws,
        estimator="adaptive",
        nuisance_prior=NuisancePrior(),
        observation=ObservationModel(integration_time_seconds=120.0),
    )
    calibration = evaluate_window_aware_campaign(
        detector,
        cases,
        seed=seed + 1000,
    )
    return {
        "draws": draws,
        "repeats": repeats,
        "seed": seed,
        **summarize_window_aware_study(
            calibration,
            threshold=threshold,
        ),
    }


def run_oscillation_replay(
    *,
    repeats: int = 8,
    draws: int = 256,
    seed: int = 2321,
    threshold: float = 0.45,
) -> dict[str, Any]:
    """Replay PR #10's difficult full-amplitude RGB population."""

    cases = build_window_aware_campaign(
        repeats=repeats,
        seed=seed,
        truths=("oscillation",),
    )
    detector = Detector(
        draws=draws,
        estimator="adaptive",
        nuisance_prior=NuisancePrior(),
        observation=ObservationModel(integration_time_seconds=120.0),
    )
    calibration = evaluate_window_aware_campaign(
        detector,
        cases,
        seed=seed + 1000,
    )
    profiles = {}
    for profile in EVALUATION_PROFILES:
        group = calibration.subset(evaluation_profile=profile)
        probabilities = np.asarray(
            [
                recovery.result.probabilities["oscillation"]
                for recovery in group.recoveries
            ]
        )
        detections = int(np.count_nonzero(probabilities >= threshold))
        profiles[profile] = {
            "cases": int(probabilities.size),
            "detections": detections,
            "true_positive_rate": detections / probabilities.size,
            "mean_oscillation_probability": float(
                np.mean(probabilities)
            ),
        }
    return {
        "draws": draws,
        "repeats": repeats,
        "seed": seed,
        "threshold": threshold,
        "profiles": profiles,
    }


def main() -> None:
    """Run the study from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--seed", type=int, default=411)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--oscillation-replay", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    runner = (
        run_oscillation_replay
        if arguments.oscillation_replay
        else run_window_aware_study
    )
    result = runner(
        repeats=arguments.repeats,
        draws=arguments.draws,
        seed=arguments.seed,
        threshold=arguments.threshold,
    )
    rendered = json.dumps(result, indent=2)
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
