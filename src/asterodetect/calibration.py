"""Small injection-recovery bookkeeping framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

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

    confusion = np.zeros((3, 3), dtype=np.int64)
    squared_errors = []
    for recovery in recoveries:
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
        recoveries=recoveries,
        confusion_matrix=confusion,
        accuracy=float(np.trace(confusion) / np.sum(confusion)),
        multiclass_brier_score=float(np.mean(squared_errors)),
    )
