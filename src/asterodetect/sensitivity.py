"""Sensitivity studies for prior-draw convergence and PSD binning."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .calibration import InjectionCase
from .detector import Detector
from .nuisance import NuisancePrior
from .observation import ObservationModel


@dataclass(frozen=True, slots=True)
class SensitivityRun:
    """One detector evaluation within a paired sensitivity study."""

    case_name: str
    truth: str
    draws: int
    dnu_scale: float
    estimator: str
    repeat: int
    probabilities: Mapping[str, float]
    log_evidences: Mapping[str, float]
    effective_sample_sizes: Mapping[str, float]
    log_evidence_standard_errors: Mapping[str, float]
    bin_width: float
    binned_points: int


@dataclass(frozen=True, slots=True)
class SensitivitySummary:
    """Aggregate diagnostics for one draw-count and bin-scale combination."""

    draws: int
    dnu_scale: float
    estimator: str
    evaluations: int
    mean_oscillation_probability: float
    oscillation_probability_std: float
    classification_accuracy: float
    minimum_median_ess_fraction: float
    maximum_median_log_evidence_standard_error: float


@dataclass(frozen=True, slots=True)
class SensitivityStudy:
    """Results of a paired detector sensitivity experiment."""

    runs: tuple[SensitivityRun, ...]

    def summaries(self) -> tuple[SensitivitySummary, ...]:
        """Aggregate results by draw count and bin scale."""

        groups: dict[tuple[int, float, str], list[SensitivityRun]] = {}
        for run in self.runs:
            groups.setdefault((run.draws, run.dnu_scale, run.estimator), []).append(run)

        summaries = []
        for (draws, dnu_scale, estimator), runs in sorted(groups.items()):
            probabilities_by_case: dict[str, list[float]] = {}
            for run in runs:
                probabilities_by_case.setdefault(run.case_name, []).append(
                    run.probabilities["oscillation"]
                )
            case_probability_stds = [
                np.std(values, ddof=1) if len(values) > 1 else 0.0
                for values in probabilities_by_case.values()
            ]
            probabilities = np.asarray(
                [run.probabilities["oscillation"] for run in runs]
            )
            correct = [
                max(run.probabilities, key=run.probabilities.__getitem__) == run.truth
                for run in runs
            ]
            median_ess_fractions = [
                np.median(
                    [
                        run.effective_sample_sizes[label] / draws
                        for run in runs
                    ]
                )
                for label in DetectorLabels
            ]
            median_standard_errors = [
                np.median(
                    [
                        run.log_evidence_standard_errors[label]
                        for run in runs
                    ]
                )
                for label in DetectorLabels
            ]
            summaries.append(
                SensitivitySummary(
                    draws=draws,
                    dnu_scale=dnu_scale,
                    estimator=estimator,
                    evaluations=len(runs),
                    mean_oscillation_probability=float(np.mean(probabilities)),
                    oscillation_probability_std=float(
                        np.mean(case_probability_stds)
                    ),
                    classification_accuracy=float(np.mean(correct)),
                    minimum_median_ess_fraction=float(
                        np.min(median_ess_fractions)
                    ),
                    maximum_median_log_evidence_standard_error=float(
                        np.max(median_standard_errors)
                    ),
                )
            )
        return tuple(summaries)


DetectorLabels = ("noise", "granulation", "oscillation")


def run_sensitivity_study(
    cases: Iterable[InjectionCase],
    *,
    draw_counts: Sequence[int] = (128, 512, 2048),
    dnu_scales: Sequence[float] = (0.5, 1.0, 2.0),
    repeats: int = 4,
    seed: int | None = None,
    observation: ObservationModel | None = None,
    nuisance_prior: NuisancePrior | None = None,
    model_probabilities: Mapping[str, float] | None = None,
    minimum_envelope_bins: int = 5,
    estimators: Sequence[str] = ("prior",),
    pilot_draws: int = 256,
    defensive_fraction: float = 0.2,
) -> SensitivityStudy:
    """Compare Monte Carlo convergence and fixed binning choices.

    Every injected spectrum is held fixed across all configurations. Each
    configuration is then evaluated with independent, reproducible inference
    seeds. Consequently, variation between configurations is not contaminated
    by different stochastic periodogram realizations.
    """

    case_tuple = tuple(cases)
    if not case_tuple:
        raise ValueError("at least one injection case is required")
    draws_tuple = _positive_integer_sequence(draw_counts, "draw_counts")
    scales_tuple = _positive_float_sequence(dnu_scales, "dnu_scales")
    estimators_tuple = tuple(estimators)
    if not estimators_tuple or any(
        estimator not in {"prior", "adaptive"} for estimator in estimators_tuple
    ):
        raise ValueError("estimators must contain 'prior' and/or 'adaptive'")
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)):
        raise TypeError("repeats must be an integer")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    coordinates = tuple(
        product(draws_tuple, scales_tuple, estimators_tuple, range(repeats), case_tuple)
    )
    child_seeds = np.random.SeedSequence(seed).spawn(len(coordinates))
    runs = []
    for (draws, dnu_scale, estimator, repeat, case), child_seed in zip(
        coordinates, child_seeds, strict=True
    ):
        detector = Detector(
            draws=draws,
            observation=observation,
            nuisance_prior=nuisance_prior,
            model_probabilities=model_probabilities,
            dnu_scale=dnu_scale,
            minimum_envelope_bins=minimum_envelope_bins,
            estimator=estimator,
            pilot_draws=pilot_draws,
            defensive_fraction=defensive_fraction,
        )
        result = detector.run(
            case.spectrum,
            case.stellar_constraints,
            rng=np.random.default_rng(child_seed),
        )
        diagnostics = result.evaluation.diagnostics
        runs.append(
            SensitivityRun(
                case_name=case.name,
                truth=case.truth,
                draws=draws,
                dnu_scale=dnu_scale,
                estimator=estimator,
                repeat=repeat,
                probabilities=MappingProxyType(dict(result.probabilities)),
                log_evidences=MappingProxyType(
                    dict(result.evaluation.log_evidences)
                ),
                effective_sample_sizes=MappingProxyType(
                    {
                        label: diagnostics[label].effective_sample_size
                        for label in DetectorLabels
                    }
                ),
                log_evidence_standard_errors=MappingProxyType(
                    {
                        label: diagnostics[label].log_evidence_standard_error
                        for label in DetectorLabels
                    }
                ),
                bin_width=result.bin_width,
                binned_points=result.binned_spectrum.frequency.size,
            )
        )
    return SensitivityStudy(tuple(runs))


def _positive_integer_sequence(
    values: Sequence[int], name: str
) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        for value in result
    ):
        raise TypeError(f"{name} must contain integers")
    if any(value < 1 for value in result):
        raise ValueError(f"{name} must contain only positive values")
    return tuple(int(value) for value in result)


def _positive_float_sequence(
    values: Sequence[float], name: str
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(result)) or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result
