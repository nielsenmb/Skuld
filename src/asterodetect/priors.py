"""Interfaces for target-specific joint priors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.typing import ArrayLike, NDArray
import numpy as np


@runtime_checkable
class JointPrior(Protocol):
    """Protocol for normalized target-specific joint priors."""

    @property
    def ndim(self) -> int:
        """Return the dimensionality of the joint prior."""

    def transform(self, unit_cube: ArrayLike) -> NDArray[np.float64]:
        """Map a point from the unit cube into physical parameters.

        Parameters
        ----------
        unit_cube
            Coordinates on the unit hypercube.

        Returns
        -------
        numpy.ndarray
            Corresponding physical parameters.
        """

    def logpdf(self, parameters: ArrayLike) -> float:
        """Evaluate the normalized joint log density.

        Parameters
        ----------
        parameters
            Physical parameter vector.

        Returns
        -------
        float
            Normalized log density.
        """
