"""Isolate observing-window effects for full-amplitude oscillators.

Every window profile is applied to the same stochastic stellar realization,
and every member of a matched set uses the same inference seed. The detector,
amplitude prior, and probability threshold remain frozen.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest

from asterodetect import (
    AstrophysicalInjectionFactory,
    CalibrationResult,
    Detector,
    InjectionCase,
    NuisancePrior,
    ObservationModel,
    Recovery,
    TESS_WINDOW_PROFILES,
    summarize_recoveries,
)
from astrophysical_campaign import REGIMES, regime_samples
from window_campaign import evaluate_paired_windows


def build_focused_window_campaign(
    *,
    repeats: int,
    seed: int,
    window_profiles: Iterable[str] = TESS_WINDOW_PROFILES,
) -> tuple[InjectionCase, ...]:
    """Build matched full-amplitude injections across TESS window profiles.

    Parameters
    ----------
    repeats
        Independent stochastic realizations per astrophysical grid cell.
    seed
        Root seed for stellar spectra and observing masks.
    window_profiles
        Window profiles applied to every stochastic realization.

    Returns
    -------
    tuple
        Full-amplitude oscillation cases with pairing metadata.
    """

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    profiles = tuple(str(profile) for profile in window_profiles)
    if not profiles or "continuous" not in profiles:
        raise ValueError("window_profiles must include continuous")
    unknown = set(profiles) - set(TESS_WINDOW_PROFILES)
    if unknown:
        raise ValueError(f"unsupported window profiles: {sorted(unknown)}")
    if len(set(profiles)) != len(profiles):
        raise ValueError("window_profiles must not contain duplicates")

    cells = tuple(
        (regime, duration, noise)
        for regime, parameters in REGIMES.items()
        for duration, noise in product(
            (27.4, 90.0),
            parameters["white_noise"],
        )
    )
    factories = {
        regime: AstrophysicalInjectionFactory(
            regime_samples(parameters, seed=seed + index),
            cadence_seconds=120.0,
        )
        for index, (regime, parameters) in enumerate(REGIMES.items())
    }
    realization_seeds = np.random.SeedSequence(seed).spawn(len(cells) * repeats)
    cases: list[InjectionCase] = []
    seed_index = 0
    for cell_index, (regime, duration, noise) in enumerate(cells):
        for repeat in range(repeats):
            pair_id = f"cell={cell_index}-repeat={repeat}"
            realization_seed = realization_seeds[seed_index]
            seed_index += 1
            parameters = {
                "truth": "oscillation",
                "duration_days": duration,
                "white_noise": noise,
                "amplitude_scale": 1.0,
                "window_seed": seed + cell_index * repeats + repeat,
            }
            for window_profile in profiles:
                generated = factories[regime](
                    f"{pair_id}-window={window_profile}",
                    {**parameters, "window_profile": window_profile},
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


def build_momentum_interval_campaign(
    *,
    repeats: int,
    seed: int,
    intervals_days: Iterable[float] = (2.5, 4.0, 6.75),
    match_duty_cycle: bool = False,
) -> tuple[InjectionCase, ...]:
    """Build a targeted low-luminosity-RGB momentum-dump interval sweep.

    When ``match_duty_cycle`` is true, gap durations scale with their
    recurrence intervals. Each periodic profile then removes approximately
    the same fraction of scheduled cadences.
    """

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    intervals = tuple(float(value) for value in intervals_days)
    if not intervals or any(not np.isfinite(value) or value <= 0 for value in intervals):
        raise ValueError("intervals_days must contain positive finite values")
    if len(set(intervals)) != len(intervals):
        raise ValueError("intervals_days must not contain duplicates")

    regime = "low_luminosity_rgb"
    parameters = REGIMES[regime]
    factory = AstrophysicalInjectionFactory(
        regime_samples(parameters, seed=seed),
        cadence_seconds=120.0,
    )
    cells = tuple(
        (duration, noise)
        for duration, noise in product(
            (27.4, 90.0),
            parameters["white_noise"],
        )
    )
    realization_seeds = np.random.SeedSequence(seed).spawn(len(cells) * repeats)
    cases: list[InjectionCase] = []
    seed_index = 0
    for cell_index, (duration, noise) in enumerate(cells):
        for repeat in range(repeats):
            pair_id = f"rgb-cell={cell_index}-repeat={repeat}"
            realization_seed = realization_seeds[seed_index]
            seed_index += 1
            common = {
                "truth": "oscillation",
                "duration_days": duration,
                "white_noise": noise,
                "amplitude_scale": 1.0,
                "window_seed": seed + cell_index * repeats + repeat,
            }
            configurations = (("continuous", None),) + tuple(
                ("momentum-dumps", interval) for interval in intervals
            )
            for profile, interval in configurations:
                overrides = {**common, "window_profile": profile}
                gap_duration_minutes = None
                if interval is not None:
                    overrides["momentum_dump_interval_days"] = interval
                    gap_duration_minutes = (
                        30.0 * interval / 2.5 if match_duty_cycle else 30.0
                    )
                    overrides["momentum_dump_duration_minutes"] = (
                        gap_duration_minutes
                    )
                generated = factory(
                    f"{pair_id}-window={profile}-interval={interval}",
                    overrides,
                    np.random.default_rng(realization_seed),
                )
                metadata = dict(generated.metadata or {})
                metadata.update(
                    stellar_regime=regime,
                    repeat=repeat,
                    pair_id=pair_id,
                    gap_interval_days=interval,
                    gap_duration_minutes=gap_duration_minutes,
                    matched_duty_cycle=match_duty_cycle,
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


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float]:
    """Return a two-sided Wilson binomial interval."""

    if trials <= 0:
        return [float("nan"), float("nan")]
    fraction = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (fraction + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            fraction * (1.0 - fraction) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return [float(centre - half_width), float(centre + half_width)]


def _profile_metrics(
    calibration: CalibrationResult,
    threshold: float,
) -> dict[str, Any]:
    """Summarize full-amplitude detections for one window profile."""

    probabilities = np.asarray(
        [
            recovery.result.probabilities["oscillation"]
            for recovery in calibration.recoveries
        ],
        dtype=float,
    )
    detections = probabilities >= threshold
    detected = int(np.count_nonzero(detections))
    cases = int(probabilities.size)
    metadata = [recovery.case.metadata for recovery in calibration.recoveries]
    return {
        "cases": cases,
        "detections": detected,
        "true_positive_rate": detected / cases,
        "true_positive_rate_wilson_95": _wilson_interval(detected, cases),
        "mean_probability": float(np.mean(probabilities)),
        "median_probability": float(np.median(probabilities)),
        "mean_duty_cycle": float(
            np.mean([item["duty_cycle"] for item in metadata])
        ),
        "mean_gap_count": float(
            np.mean([item["gap_count"] for item in metadata])
        ),
        "mean_maximum_gap_hours": float(
            np.mean([item["maximum_gap_hours"] for item in metadata])
        ),
        "mean_peak_sidelobe_power": float(
            np.mean([item["peak_sidelobe_power"] for item in metadata])
        ),
    }


def _envelope_power(
    recovery: Recovery,
) -> float:
    """Integrate the recovered binned PSD across one envelope FWHM."""

    metadata = recovery.case.metadata
    spectrum = recovery.result.binned_spectrum
    half_width = float(metadata["fwhm_envelope"]) / 2.0
    selected = np.abs(spectrum.frequency - float(metadata["numax"])) <= half_width
    if np.count_nonzero(selected) < 2:
        return float("nan")
    return float(np.trapz(spectrum.power[selected], spectrum.frequency[selected]))


def _paired_profile_metrics(
    calibration: CalibrationResult,
    profile: str,
    threshold: float,
) -> dict[str, Any]:
    """Compare one window profile with its matched continuous cases."""

    pairs: dict[str, dict[str, Recovery]] = defaultdict(dict)
    for recovery in calibration.recoveries:
        metadata = recovery.case.metadata
        pairs[str(metadata["pair_id"])][str(metadata["window_profile"])] = recovery
    probability_shifts = []
    envelope_power_ratios = []
    lost = 0
    gained = 0
    for pair in pairs.values():
        continuous = pair["continuous"]
        comparison = pair[profile]
        continuous_probability = continuous.result.probabilities["oscillation"]
        comparison_probability = comparison.result.probabilities["oscillation"]
        probability_shifts.append(comparison_probability - continuous_probability)
        lost += continuous_probability >= threshold > comparison_probability
        gained += comparison_probability >= threshold > continuous_probability
        continuous_power = _envelope_power(continuous)
        comparison_power = _envelope_power(comparison)
        envelope_power_ratios.append(comparison_power / continuous_power)

    shifts = np.asarray(probability_shifts, dtype=float)
    ratios = np.asarray(envelope_power_ratios, dtype=float)
    discordant = lost + gained
    paired_p = (
        float(binomtest(min(lost, gained), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "pairs": int(shifts.size),
        "mean_probability_shift": float(np.mean(shifts)),
        "median_probability_shift": float(np.median(shifts)),
        "mean_absolute_probability_shift": float(np.mean(np.abs(shifts))),
        "detections_lost": int(lost),
        "detections_gained": int(gained),
        "exact_paired_p_value": paired_p,
        "median_envelope_power_ratio": float(np.nanmedian(ratios)),
    }


def _grouped_metrics(
    calibration: CalibrationResult,
    *,
    key: str,
    threshold: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Group profile metrics by one astrophysical coordinate."""

    output: dict[str, dict[str, dict[str, Any]]] = {}
    for value, group in calibration.group_by(key).items():
        output[str(value)] = {
            profile: _profile_metrics(
                group.subset(window_profile=profile),
                threshold,
            )
            for profile in TESS_WINDOW_PROFILES
        }
    return output


def summarize_focused_window_study(
    calibration: CalibrationResult,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Summarize paired full-amplitude window sensitivity."""

    profiles = {
        profile: _profile_metrics(
            calibration.subset(window_profile=profile),
            threshold,
        )
        for profile in TESS_WINDOW_PROFILES
    }
    return {
        "threshold": float(threshold),
        "profiles": profiles,
        "paired_vs_continuous": {
            profile: _paired_profile_metrics(calibration, profile, threshold)
            for profile in TESS_WINDOW_PROFILES
            if profile != "continuous"
        },
        "by_stellar_regime": _grouped_metrics(
            calibration, key="stellar_regime", threshold=threshold
        ),
        "by_duration_days": _grouped_metrics(
            calibration, key="duration_days", threshold=threshold
        ),
        "by_white_noise": _grouped_metrics(
            calibration, key="white_noise", threshold=threshold
        ),
        "by_numax": _grouped_metrics(
            calibration, key="numax", threshold=threshold
        ),
    }


def summarize_momentum_interval_study(
    calibration: CalibrationResult,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Summarize the targeted RGB momentum-dump interval sweep."""

    continuous = calibration.subset(window_profile="continuous")
    intervals = sorted(
        {
            float(recovery.case.metadata["gap_interval_days"])
            for recovery in calibration.recoveries
            if recovery.case.metadata["gap_interval_days"] is not None
        }
    )
    output: dict[str, Any] = {
        "continuous": _profile_metrics(continuous, threshold),
        "intervals_days": {},
    }
    for interval in intervals:
        comparison = calibration.subset(gap_interval_days=interval)
        paired = summarize_recoveries(
            (*continuous.recoveries, *comparison.recoveries)
        )
        output["intervals_days"][str(interval)] = {
            "profile": _profile_metrics(comparison, threshold),
            "paired_vs_continuous": _paired_profile_metrics(
                paired, "momentum-dumps", threshold
            ),
            "by_duration_days": {
                str(value): _profile_metrics(group, threshold)
                for value, group in comparison.group_by("duration_days").items()
            },
            "by_white_noise": {
                str(value): _profile_metrics(group, threshold)
                for value, group in comparison.group_by("white_noise").items()
            },
        }
    return output


def run_focused_window_study(
    *,
    repeats: int,
    draws: int,
    seed: int,
    threshold: float = 0.45,
    run_interval_sweep: bool = True,
) -> dict[str, Any]:
    """Run the focused paired observing-window experiment."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    cases = build_focused_window_campaign(repeats=repeats, seed=seed)
    detector = Detector(
        draws=draws,
        estimator="adaptive",
        nuisance_prior=NuisancePrior(),
        observation=ObservationModel(integration_time_seconds=120.0),
    )
    calibration = evaluate_paired_windows(detector, cases, seed=seed + 1000)
    output = {
        "draws": draws,
        "repeats": repeats,
        "seed": seed,
        **summarize_focused_window_study(calibration, threshold=threshold),
    }
    if run_interval_sweep:
        interval_cases = build_momentum_interval_campaign(
            repeats=repeats,
            seed=seed + 2000,
        )
        interval_calibration = evaluate_paired_windows(
            detector,
            interval_cases,
            seed=seed + 3000,
        )
        output["momentum_dump_interval_sweep"] = (
            summarize_momentum_interval_study(
                interval_calibration,
                threshold=threshold,
            )
        )
    return output


def run_momentum_interval_study(
    *,
    repeats: int,
    draws: int,
    seed: int,
    threshold: float = 0.45,
    match_duty_cycle: bool = False,
) -> dict[str, Any]:
    """Run only the targeted low-luminosity-RGB interval sweep."""

    detector = Detector(
        draws=draws,
        estimator="adaptive",
        nuisance_prior=NuisancePrior(),
        observation=ObservationModel(integration_time_seconds=120.0),
    )
    cases = build_momentum_interval_campaign(
        repeats=repeats,
        seed=seed,
        match_duty_cycle=match_duty_cycle,
    )
    calibration = evaluate_paired_windows(detector, cases, seed=seed + 1000)
    return {
        "draws": draws,
        "repeats": repeats,
        "seed": seed,
        "threshold": threshold,
        "matched_duty_cycle": match_duty_cycle,
        **summarize_momentum_interval_study(
            calibration,
            threshold=threshold,
        ),
    }


def main() -> None:
    """Run the focused window study from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--skip-interval-sweep", action="store_true")
    parser.add_argument("--interval-sweep-only", action="store_true")
    parser.add_argument("--match-interval-duty-cycle", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.interval_sweep_only:
        result = run_momentum_interval_study(
            repeats=arguments.repeats,
            draws=arguments.draws,
            seed=arguments.seed,
            threshold=arguments.threshold,
            match_duty_cycle=arguments.match_interval_duty_cycle,
        )
    else:
        result = run_focused_window_study(
            repeats=arguments.repeats,
            draws=arguments.draws,
            seed=arguments.seed,
            threshold=arguments.threshold,
            run_interval_sweep=not arguments.skip_interval_sweep,
        )
    rendered = json.dumps(result, indent=2)
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
