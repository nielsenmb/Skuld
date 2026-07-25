"""Compare alternative Bayesian model sets on identical evidence estimates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from asterodetect import (
    CalibrationResult,
    Detector,
    NuisancePrior,
    ObservationModel,
    reweight_calibration_models,
    summarize_recoveries,
)
from astrophysical_campaign import REGIMES
from window_campaign import build_window_campaign, evaluate_paired_windows


def metrics(
    calibration: CalibrationResult,
    threshold: float,
) -> dict[str, object]:
    """Return binary detection counts for one model set and threshold."""

    recoveries = calibration.recoveries
    truth = np.asarray([item.case.truth == "oscillation" for item in recoveries])
    probability = np.asarray(
        [item.result.probabilities["oscillation"] for item in recoveries]
    )
    detected = probability >= threshold
    positive = int(np.sum(truth))
    negative = int(np.sum(~truth))
    return {
        "cases": len(recoveries),
        "threshold": float(threshold),
        "true_positive_rate": (
            float(np.sum(detected & truth) / positive) if positive else None
        ),
        "false_positive_rate": (
            float(np.sum(detected & ~truth) / negative) if negative else None
        ),
        "false_positives_by_truth": {
            label: int(
                sum(
                    item.case.truth == label
                    and item.result.probabilities["oscillation"] >= threshold
                    for item in recoveries
                )
            )
            for label in ("noise", "granulation")
        },
        "mean_probability": float(np.mean(probability)),
        "binary_brier_score": float(np.mean((probability - truth) ** 2)),
    }


def select_threshold(
    calibration: CalibrationResult,
    maximum_fpr: float = 0.05,
) -> dict[str, object]:
    """Select the highest-TPR operating point under an FPR constraint."""

    candidates = [
        metrics(calibration, threshold)
        for threshold in np.linspace(0.05, 1.0, 20)
    ]
    feasible = [
        item
        for item in candidates
        if item["false_positive_rate"] <= maximum_fpr
    ]
    pool = feasible or candidates
    selected = max(
        pool,
        key=lambda item: (
            item["true_positive_rate"]
            if feasible
            else -item["false_positive_rate"],
            -item["binary_brier_score"],
            -item["threshold"],
        ),
    )
    selected["false_positive_constraint_satisfied"] = bool(feasible)
    return selected


def run_study(
    *,
    profile: str,
    repeats: int,
    draws: int,
    seed: int,
    maximum_false_positive_rate: float = 0.05,
) -> dict[str, object]:
    """Run a paired comparison without repeating evidence calculations."""

    cases = build_window_campaign(profile=profile, repeats=repeats, seed=seed)
    tuning_repeats = set(range(repeats // 2))
    tuning = tuple(
        case for case in cases if case.metadata["repeat"] in tuning_repeats
    )
    validation = tuple(
        case for case in cases if case.metadata["repeat"] not in tuning_repeats
    )
    detector = Detector(
        draws=draws,
        estimator="adaptive",
        nuisance_prior=NuisancePrior(),
        observation=ObservationModel(integration_time_seconds=120.0),
    )
    tuning_result = evaluate_paired_windows(detector, tuning, seed=seed + 100)
    validation_result = evaluate_paired_windows(
        detector,
        validation,
        seed=seed + 1000,
    )
    continuous_tuning = tuning_result.subset(window_profile="continuous")
    model_sets = {
        "three": None,
        "noise-oscillation": {"noise": 0.5, "oscillation": 0.5},
        "granulation-oscillation": {
            "granulation": 0.5,
            "oscillation": 0.5,
        },
    }
    report: dict[str, object] = {
        "profile": profile,
        "draws": draws,
        "repeats": repeats,
        "seed": seed,
        "maximum_false_positive_rate": maximum_false_positive_rate,
        "regimes": list(REGIMES),
        "model_sets": {},
    }
    model_reports: dict[str, object] = {}
    for model_set, model_priors in model_sets.items():
        tuning_candidate = (
            continuous_tuning
            if model_priors is None
            else reweight_calibration_models(continuous_tuning, model_priors)
        )
        selected = select_threshold(
            tuning_candidate,
            maximum_fpr=maximum_false_positive_rate,
        )
        threshold = selected["threshold"]
        entry = {"tuning_continuous": selected, "validation": {}}
        for window in ("continuous", "tess-like"):
            subset = validation_result.subset(window_profile=window)
            candidate = (
                subset
                if model_priors is None
                else reweight_calibration_models(subset, model_priors)
            )
            entry["validation"][window] = metrics(candidate, threshold)
            by_amplitude = defaultdict(list)
            for recovery in candidate.recoveries:
                if recovery.case.truth == "oscillation":
                    by_amplitude[str(recovery.case.metadata["amplitude_scale"])].append(
                        recovery
                    )
            entry["validation"][window]["true_positive_rate_by_amplitude"] = {
                amplitude: metrics(
                    summarize_recoveries(items),
                    threshold,
                )[
                    "true_positive_rate"
                ]
                for amplitude, items in by_amplitude.items()
            }
        model_reports[model_set] = entry
    report["model_sets"] = model_reports

    three_threshold = model_reports["three"]["tuning_continuous"][
        "threshold"
    ]
    false_negatives = []
    for recovery in validation_result.recoveries:
        if (
            recovery.case.truth == "oscillation"
            and recovery.result.probabilities["oscillation"] < three_threshold
        ):
            probabilities = recovery.result.probabilities
            false_negatives.append(
                {
                    "window": recovery.case.metadata["window_profile"],
                    "amplitude": recovery.case.metadata["amplitude_scale"],
                    "regime": recovery.case.metadata["stellar_regime"],
                    "duration": recovery.case.metadata["duration_days"],
                    "largest_competitor": max(
                        ("noise", "granulation"),
                        key=probabilities.__getitem__,
                    ),
                    "p_noise": probabilities["noise"],
                    "p_granulation": probabilities["granulation"],
                    "p_oscillation": probabilities["oscillation"],
                }
            )
    report["three_model_false_negatives"] = {
        "count": len(false_negatives),
        "largest_competitor": {
            label: sum(
                item["largest_competitor"] == label for item in false_negatives
            )
            for label in ("noise", "granulation")
        },
        "median_probabilities": {
            label: float(
                np.median([item[f"p_{label}"] for item in false_negatives])
            )
            for label in ("noise", "granulation", "oscillation")
        },
    }
    return report


def main() -> None:
    """Run the command-line model-set sensitivity study."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("checkpoint", "standard"),
        default="checkpoint",
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--maximum-false-positive-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_study(
        profile=arguments.profile,
        repeats=arguments.repeats,
        draws=arguments.draws,
        seed=arguments.seed,
        maximum_false_positive_rate=arguments.maximum_false_positive_rate,
    )
    rendered = json.dumps(report, indent=2)
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
