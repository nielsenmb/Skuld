"""Small injection-recovery bookkeeping framework."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .asteroscale import AsteroScaleSamples
from .data import PowerSpectrum
from .detector import DetectionResult, Detector


MODEL_LABELS = ("noise", "granulation", "oscillation")


@dataclass(frozen=True, slots=True)
class InjectionCase:
    """One simulated spectrum and its known generating class."""

    name: str
    truth: str
    spectrum: PowerSpectrum
    stellar_constraints: Mapping[str, Any] | AsteroScaleSamples
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.truth not in MODEL_LABELS:
            raise ValueError(f"truth must be one of {MODEL_LABELS}")


@dataclass(frozen=True, slots=True)
class Recovery:
    case: InjectionCase
    result: DetectionResult


@dataclass(frozen=True, slots=True)
class ProbabilityBin:
    """One bin in a binary probability-reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_probability: float
    observed_frequency: float


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """Binary oscillation-detection metrics at a chosen threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    completeness: float
    false_positive_rate: float
    precision: float
    binary_brier_score: float
    expected_calibration_error: float
    reliability: tuple[ProbabilityBin, ...]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    recoveries: tuple[Recovery, ...]
    confusion_matrix: NDArray[np.int64]
    accuracy: float
    multiclass_brier_score: float

    def detection_metrics(
        self,
        *,
        threshold: float = 0.5,
        probability_bins: int | Sequence[float] = 10,
    ) -> DetectionMetrics:
        """Treat oscillation as the positive class and summarize detection.

        Noise and granulation are both negative cases. This is the most useful
        view for reporting completeness and false-positive rates while the
        three-way confusion matrix retains the failure mode.
        """

        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and between zero and one")
        probabilities = np.asarray(
            [
                recovery.result.probabilities["oscillation"]
                for recovery in self.recoveries
            ],
            dtype=float,
        )
        truth = np.asarray(
            [recovery.case.truth == "oscillation" for recovery in self.recoveries]
        )
        predicted = probabilities >= threshold
        true_positives = int(np.sum(predicted & truth))
        false_positives = int(np.sum(predicted & ~truth))
        true_negatives = int(np.sum(~predicted & ~truth))
        false_negatives = int(np.sum(~predicted & truth))
        reliability = probability_reliability(
            probabilities, truth, bins=probability_bins
        )
        expected_calibration_error = float(
            sum(
                item.count
                * abs(item.mean_probability - item.observed_frequency)
                for item in reliability
            )
            / probabilities.size
        )
        return DetectionMetrics(
            threshold=threshold,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
            completeness=_safe_ratio(
                true_positives, true_positives + false_negatives
            ),
            false_positive_rate=_safe_ratio(
                false_positives, false_positives + true_negatives
            ),
            precision=_safe_ratio(true_positives, true_positives + false_positives),
            binary_brier_score=float(np.mean((probabilities - truth) ** 2)),
            expected_calibration_error=expected_calibration_error,
            reliability=reliability,
        )

    def subset(self, **metadata: Any) -> "CalibrationResult":
        """Return metrics for recoveries matching all supplied metadata."""

        selected = tuple(
            recovery
            for recovery in self.recoveries
            if recovery.case.metadata is not None
            and all(
                recovery.case.metadata.get(name) == value
                for name, value in metadata.items()
            )
        )
        if not selected:
            raise ValueError("no recoveries match the requested metadata")
        return summarize_recoveries(selected)

    def group_by(self, metadata_key: str) -> Mapping[Any, "CalibrationResult"]:
        """Calculate separate calibration summaries for one metadata field."""

        groups: dict[Any, list[Recovery]] = {}
        for recovery in self.recoveries:
            metadata = recovery.case.metadata
            if metadata is None or metadata_key not in metadata:
                continue
            groups.setdefault(metadata[metadata_key], []).append(recovery)
        if not groups:
            raise ValueError(f"no recoveries contain metadata key {metadata_key!r}")
        return {
            value: summarize_recoveries(group)
            for value, group in groups.items()
        }


def build_injection_grid(
    axes: Mapping[str, Sequence[Any]],
    factory: Callable[[str, Mapping[str, Any], np.random.Generator], InjectionCase],
    *,
    repeats: int = 1,
    seed: int | None = None,
) -> tuple[InjectionCase, ...]:
    """Construct reproducible injections over a Cartesian parameter grid.

    The factory owns the astrophysical simulation. It receives a unique name,
    a read-only coordinate mapping, and an independent random generator.
    """

    if not axes:
        raise ValueError("at least one grid axis is required")
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)):
        raise TypeError("repeats must be an integer")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    names = tuple(axes)
    values = tuple(tuple(axes[name]) for name in names)
    if any(not axis for axis in values):
        raise ValueError("grid axes cannot be empty")

    combinations = tuple(product(*values))
    seeds = np.random.SeedSequence(seed).spawn(len(combinations) * repeats)
    cases: list[InjectionCase] = []
    seed_index = 0
    for coordinates in combinations:
        parameters = dict(zip(names, coordinates, strict=True))
        for repeat in range(repeats):
            labels = "-".join(f"{name}={parameters[name]}" for name in names)
            case = factory(
                f"{labels}-repeat={repeat}",
                MappingProxyType(dict(parameters)),
                np.random.default_rng(seeds[seed_index]),
            )
            seed_index += 1
            if not isinstance(case, InjectionCase):
                raise TypeError("factory must return an InjectionCase")
            metadata = dict(case.metadata or {})
            metadata.update(parameters)
            metadata["repeat"] = repeat
            cases.append(
                InjectionCase(
                    name=case.name,
                    truth=case.truth,
                    spectrum=case.spectrum,
                    stellar_constraints=case.stellar_constraints,
                    metadata=MappingProxyType(metadata),
                )
            )
    return tuple(cases)


def build_detection_study(
    axes: Mapping[str, Sequence[Any]],
    factory: Callable[[str, Mapping[str, Any], np.random.Generator], InjectionCase],
    *,
    oscillation_amplitudes: Sequence[float] = (0.1, 0.3, 1.0),
    repeats: int = 20,
    seed: int | None = None,
) -> tuple[InjectionCase, ...]:
    """Build a balanced null/background/detection study.

    Noise and granulation cases are generated once per coordinate, while the
    oscillation class is generated at every requested amplitude. This avoids
    accidentally duplicating the two negative classes across an irrelevant
    amplitude axis.
    """

    if "truth" in axes or "amplitude_scale" in axes:
        raise ValueError("axes must not contain truth or amplitude_scale")
    amplitudes = tuple(float(value) for value in oscillation_amplitudes)
    if (
        not amplitudes
        or not np.all(np.isfinite(amplitudes))
        or any(value < 0 for value in amplitudes)
    ):
        raise ValueError("oscillation_amplitudes must be finite and non-negative")
    class_axis = ("noise", "granulation") + tuple(
        f"oscillation:{value:g}" for value in amplitudes
    )

    def study_factory(
        name: str,
        parameters: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> InjectionCase:
        class_value = str(parameters["study_class"])
        forwarded = {
            key: value for key, value in parameters.items() if key != "study_class"
        }
        if class_value.startswith("oscillation:"):
            forwarded["truth"] = "oscillation"
            forwarded["amplitude_scale"] = float(class_value.split(":", 1)[1])
        else:
            forwarded["truth"] = class_value
            forwarded["amplitude_scale"] = 0.0
        return factory(name, MappingProxyType(forwarded), rng)

    study_axes = dict(axes)
    study_axes["study_class"] = class_axis
    return build_injection_grid(
        study_axes, study_factory, repeats=repeats, seed=seed
    )


def summarize_recoveries(recoveries: Iterable[Recovery]) -> CalibrationResult:
    """Calculate classification and probability metrics for recoveries."""

    recovery_tuple = tuple(recoveries)
    if not recovery_tuple:
        raise ValueError("at least one recovery is required")

    confusion = np.zeros((3, 3), dtype=np.int64)
    squared_errors = []
    for recovery in recovery_tuple:
        truth_index = MODEL_LABELS.index(recovery.case.truth)
        predicted_index = MODEL_LABELS.index(recovery.result.classification)
        confusion[truth_index, predicted_index] += 1
        probabilities = np.asarray(
            [recovery.result.probabilities[label] for label in MODEL_LABELS]
        )
        target = np.zeros(3)
        target[truth_index] = 1.0
        squared_errors.append(np.sum((probabilities - target) ** 2))
    confusion.setflags(write=False)
    return CalibrationResult(
        recoveries=recovery_tuple,
        confusion_matrix=confusion,
        accuracy=float(np.trace(confusion) / np.sum(confusion)),
        multiclass_brier_score=float(np.mean(squared_errors)),
    )


def probability_reliability(
    probabilities: Sequence[float] | NDArray[np.floating],
    truth: Sequence[bool] | NDArray[np.bool_],
    *,
    bins: int | Sequence[float] = 10,
) -> tuple[ProbabilityBin, ...]:
    """Bin predicted probabilities and compare them with observed frequency."""

    probability_array = np.asarray(probabilities, dtype=float)
    truth_array = np.asarray(truth, dtype=bool)
    if (
        probability_array.ndim != 1
        or truth_array.ndim != 1
        or probability_array.size != truth_array.size
        or probability_array.size == 0
    ):
        raise ValueError("probabilities and truth must be non-empty 1D arrays")
    if (
        not np.all(np.isfinite(probability_array))
        or np.any(probability_array < 0)
        or np.any(probability_array > 1)
    ):
        raise ValueError("probabilities must be finite and between zero and one")
    if isinstance(bins, bool):
        raise TypeError("bins must be an integer or sequence of edges")
    if isinstance(bins, (int, np.integer)):
        if bins < 1:
            raise ValueError("bins must be positive")
        edges = np.linspace(0.0, 1.0, int(bins) + 1)
    else:
        edges = np.asarray(tuple(bins), dtype=float)
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.all(np.isfinite(edges))
            or edges[0] != 0
            or edges[-1] != 1
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError(
                "bin edges must increase strictly from zero to one"
            )

    indices = np.searchsorted(edges, probability_array, side="right") - 1
    indices = np.minimum(indices, edges.size - 2)
    result = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = indices == index
        count = int(np.sum(selected))
        if count == 0:
            continue
        result.append(
            ProbabilityBin(
                lower=float(lower),
                upper=float(upper),
                count=count,
                mean_probability=float(np.mean(probability_array[selected])),
                observed_frequency=float(np.mean(truth_array[selected])),
            )
        )
    return tuple(result)


def threshold_curve(
    calibration: CalibrationResult,
    thresholds: Sequence[float] | NDArray[np.floating] = np.linspace(0, 1, 101),
) -> tuple[DetectionMetrics, ...]:
    """Return completeness and false-positive rates over many thresholds."""

    return tuple(
        calibration.detection_metrics(threshold=float(threshold))
        for threshold in thresholds
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def evaluate_injections(
    detector: Detector,
    cases: Iterable[InjectionCase],
    *,
    seed: int | None = None,
) -> CalibrationResult:
    """Run a reproducible injection set and calculate basic diagnostics."""

    case_tuple = tuple(cases)
    if not case_tuple:
        raise ValueError("at least one injection case is required")
    child_seeds = np.random.SeedSequence(seed).spawn(len(case_tuple))
    recoveries = tuple(
        Recovery(
            case,
            detector.run(
                case.spectrum,
                case.stellar_constraints,
                rng=np.random.default_rng(child_seed),
            ),
        )
        for case, child_seed in zip(case_tuple, child_seeds, strict=True)
    )

    return summarize_recoveries(recoveries)
