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
    """One detector evaluation within a paired sensitivity study.

    Attributes
    ----------
    case_name, truth
        Injection identifier and generating class.
    draws, dnu_scale, estimator, repeat
        Inference configuration.
    probabilities, log_evidences
        Three-model inference outputs.
    effective_sample_sizes, log_evidence_standard_errors
        Model-specific Monte Carlo diagnostics.
    bin_width, binned_points
        Fixed-binning summary.
    """

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
    """Aggregate diagnostics for one inference configuration.

    Attributes
    ----------
    draws, dnu_scale, estimator
        Inference configuration.
    evaluations
        Number of detector runs in the aggregate.
    mean_oscillation_probability, oscillation_probability_std
        Mean probability and mean fixed-case repeat scatter.
    classification_accuracy
        Fraction of maximum-probability classifications that are correct.
    minimum_median_ess_fraction
        Worst model-specific median ESS divided by draw count.
    maximum_median_log_evidence_standard_error
        Worst model-specific median log-evidence uncertainty.
    minimum_truth_model_median_ess_fraction
        Worst median ESS fraction when each injection is evaluated using its
        known generating model.
    """

    draws: int
    dnu_scale: float
    estimator: str
    evaluations: int
    mean_oscillation_probability: float
    oscillation_probability_std: float
    classification_accuracy: float
    minimum_median_ess_fraction: float
    maximum_median_log_evidence_standard_error: float
    minimum_truth_model_median_ess_fraction: float


@dataclass(frozen=True, slots=True)
class EstimatorComparison:
    """Adaptive-estimator changes relative to prior sampling.

    Attributes
    ----------
    draws, dnu_scale
        Matched inference configuration.
    evaluations_per_estimator
        Number of paired detector runs.
    mean_absolute_oscillation_probability_difference
        Mean absolute probability shift between estimators.
    classification_accuracy_difference
        Adaptive accuracy minus prior-sampling accuracy.
    minimum_median_ess_fraction_ratio
        Adaptive-to-prior ratio for the worst median ESS fraction.
    maximum_median_log_evidence_standard_error_ratio
        Adaptive-to-prior ratio for the worst evidence uncertainty.
    minimum_truth_model_median_ess_fraction_ratio
        Adaptive-to-prior ratio for the worst generating-model ESS fraction.
    """

    draws: int
    dnu_scale: float
    evaluations_per_estimator: int
    mean_absolute_oscillation_probability_difference: float
    classification_accuracy_difference: float
    minimum_median_ess_fraction_ratio: float
    maximum_median_log_evidence_standard_error_ratio: float
    minimum_truth_model_median_ess_fraction_ratio: float


@dataclass(frozen=True, slots=True)
class SensitivityStudy:
    """Results of a paired detector sensitivity experiment.

    Parameters
    ----------
    runs
        Individual detector evaluations.
    """

    runs: tuple[SensitivityRun, ...]

    def summaries(self) -> tuple[SensitivitySummary, ...]:
        """Aggregate results by draw count, bin scale, and estimator.

        Returns
        -------
        tuple
            One summary per unique inference configuration.
        """

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
            truth_model_medians = [
                np.median(
                    [
                        run.effective_sample_sizes[run.truth] / draws
                        for run in runs
                        if run.truth == label
                    ]
                )
                for label in DetectorLabels
                if any(run.truth == label for run in runs)
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
                    minimum_truth_model_median_ess_fraction=float(
                        np.min(truth_model_medians)
                    ),
                )
            )
        return tuple(summaries)

    def estimator_comparisons(self) -> tuple[EstimatorComparison, ...]:
        """Compare adaptive and prior estimators for matched configurations.

        Returns
        -------
        tuple
            Paired summaries where both estimators are available.
        """

        grouped: dict[tuple[int, float, str], SensitivitySummary] = {
            (summary.draws, summary.dnu_scale, summary.estimator): summary
            for summary in self.summaries()
        }
        comparisons = []
        configurations = sorted(
            {(run.draws, run.dnu_scale) for run in self.runs}
        )
        for draws, dnu_scale in configurations:
            prior = grouped.get((draws, dnu_scale, "prior"))
            adaptive = grouped.get((draws, dnu_scale, "adaptive"))
            if prior is None or adaptive is None:
                continue

            prior_runs = {
                (run.case_name, run.repeat): run
                for run in self.runs
                if run.draws == draws
                and run.dnu_scale == dnu_scale
                and run.estimator == "prior"
            }
            adaptive_runs = {
                (run.case_name, run.repeat): run
                for run in self.runs
                if run.draws == draws
                and run.dnu_scale == dnu_scale
                and run.estimator == "adaptive"
            }
            matched = sorted(prior_runs.keys() & adaptive_runs.keys())
            probability_difference = np.mean(
                [
                    abs(
                        adaptive_runs[key].probabilities["oscillation"]
                        - prior_runs[key].probabilities["oscillation"]
                    )
                    for key in matched
                ]
            )
            comparisons.append(
                EstimatorComparison(
                    draws=draws,
                    dnu_scale=dnu_scale,
                    evaluations_per_estimator=len(matched),
                    mean_absolute_oscillation_probability_difference=float(
                        probability_difference
                    ),
                    classification_accuracy_difference=(
                        adaptive.classification_accuracy
                        - prior.classification_accuracy
                    ),
                    minimum_median_ess_fraction_ratio=_safe_ratio(
                        adaptive.minimum_median_ess_fraction,
                        prior.minimum_median_ess_fraction,
                    ),
                    maximum_median_log_evidence_standard_error_ratio=_safe_ratio(
                        adaptive.maximum_median_log_evidence_standard_error,
                        prior.maximum_median_log_evidence_standard_error,
                    ),
                    minimum_truth_model_median_ess_fraction_ratio=_safe_ratio(
                        adaptive.minimum_truth_model_median_ess_fraction,
                        prior.minimum_truth_model_median_ess_fraction,
                    ),
                )
            )
        return tuple(comparisons)


DetectorLabels = ("noise", "granulation", "oscillation")


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a stable ratio for non-negative diagnostics."""

    if denominator == 0:
        return float("inf") if numerator > 0 else 1.0
    return float(numerator / denominator)


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
    pilot_ess_fraction: float = 0.1,
    proposal_degrees_of_freedom: float = 5.0,
    stellar_draws_per_nuisance: int = 8,
) -> SensitivityStudy:
    """Compare Monte Carlo convergence and fixed binning choices.

    Every injected spectrum is held fixed across all configurations. Each
    configuration is then evaluated with independent, reproducible inference
    seeds. Consequently, variation between configurations is not contaminated
    by different stochastic periodogram realizations.

    Parameters
    ----------
    cases
        Fixed injected spectra.
    draw_counts
        Final Monte Carlo sample counts to compare.
    dnu_scales
        Fixed bin widths in units of predicted large separation.
    repeats
        Independent inference repeats per configuration and case.
    seed
        Root random seed.
    observation, nuisance_prior, model_probabilities
        Detector configuration shared across runs.
    minimum_envelope_bins
        Minimum bins retained across the predicted envelope FWHM.
    estimators
        Evidence estimators to compare.
    pilot_draws, defensive_fraction
        Adaptive-estimator controls.
    pilot_ess_fraction
        Minimum pilot ESS fraction enforced through likelihood tempering.
    proposal_degrees_of_freedom
        Degrees of freedom of the adaptive Student proposal.
    stellar_draws_per_nuisance
        Intact AsteroScale rows averaged per adaptive nuisance point.

    Returns
    -------
    SensitivityStudy
        Paired per-run outputs and aggregation methods.
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
            pilot_ess_fraction=pilot_ess_fraction,
            proposal_degrees_of_freedom=proposal_degrees_of_freedom,
            stellar_draws_per_nuisance=stellar_draws_per_nuisance,
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
    """Validate a non-empty sequence of positive integers."""

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
    """Validate a non-empty sequence of positive finite values."""

    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(result)) or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result
