"""Interfaces for target-specific joint priors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.typing import ArrayLike, NDArray
import numpy as np


@runtime_checkable
class JointPrior(Protocol):
    """Protocol to be implemented by the future AsteroScale adapter."""

    @property
    def ndim(self) -> int:
        """Dimensionality of the joint prior."""

    def transform(self, unit_cube: ArrayLike) -> NDArray[np.float64]:
        """Map a point from the unit cube into physical parameters."""

    def logpdf(self, parameters: ArrayLike) -> float:
        """Evaluate the normalized joint log-density."""

