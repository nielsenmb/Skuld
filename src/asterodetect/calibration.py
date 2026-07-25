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
class CalibrationResult:
    recoveries: tuple[Recovery, ...]
    confusion_matrix: NDArray[np.int64]
    accuracy: float
    multiclass_brier_score: float

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
