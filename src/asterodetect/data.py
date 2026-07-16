"""Validated power-spectrum data structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class PowerSpectrum:
    """A power-density spectrum and its Gamma shape parameter.

    Parameters
    ----------
    frequency
        Strictly increasing frequencies.  The model is unit-agnostic, but all
        frequency-valued parameters must use the same unit.
    power
        Positive power-density values.
    bins_averaged
        Number of independent raw periodogram bins averaged into each value.
        One gives the exponential likelihood.  An array allows differently
        binned sections of a spectrum.
    """

    frequency: NDArray[np.float64]
    power: NDArray[np.float64]
    bins_averaged: NDArray[np.float64]

    def __init__(
        self,
        frequency: ArrayLike,
        power: ArrayLike,
        bins_averaged: ArrayLike = 1.0,
    ) -> None:
        frequency_array = np.asarray(frequency, dtype=float)
        power_array = np.asarray(power, dtype=float)

        if frequency_array.ndim != 1 or power_array.ndim != 1:
            raise ValueError("frequency and power must be one-dimensional")
        if frequency_array.shape != power_array.shape:
            raise ValueError("frequency and power must have matching shapes")
        if frequency_array.size == 0:
            raise ValueError("the power spectrum must contain at least one bin")
        if not np.all(np.isfinite(frequency_array)):
            raise ValueError("frequency must contain only finite values")
        if not np.all(np.isfinite(power_array)) or np.any(power_array <= 0):
            raise ValueError("power must contain only finite, positive values")
        if np.any(frequency_array < 0) or np.any(np.diff(frequency_array) <= 0):
            raise ValueError("frequency must be non-negative and strictly increasing")

        shape_array = np.asarray(bins_averaged, dtype=float)
        try:
            shape_array = np.broadcast_to(shape_array, frequency_array.shape).copy()
        except ValueError as error:
            raise ValueError(
                "bins_averaged must be scalar or broadcast to the spectrum shape"
            ) from error
        if not np.all(np.isfinite(shape_array)) or np.any(shape_array < 1):
            raise ValueError("bins_averaged must be finite and at least one")

        frequency_array.setflags(write=False)
        power_array.setflags(write=False)
        shape_array.setflags(write=False)
        object.__setattr__(self, "frequency", frequency_array)
        object.__setattr__(self, "power", power_array)
        object.__setattr__(self, "bins_averaged", shape_array)

    def select(self, minimum: float, maximum: float) -> "PowerSpectrum":
        """Return a frequency interval, including both endpoints."""

        if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
            raise ValueError("minimum and maximum must be finite with minimum < maximum")
        mask = (self.frequency >= minimum) & (self.frequency <= maximum)
        if not np.any(mask):
            raise ValueError("the selected interval contains no frequency bins")
        return PowerSpectrum(
            self.frequency[mask], self.power[mask], self.bins_averaged[mask]
        )

    def bin_regular(self, factor: int) -> "PowerSpectrum":
        """Average consecutive independent bins by an integer factor.

        Any incomplete group at the high-frequency edge is discarded.  This
        method assumes the input bins are mutually independent.
        """

        if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)):
            raise TypeError("factor must be an integer")
        if factor < 1:
            raise ValueError("factor must be at least one")
        if factor == 1:
            return self

        groups = self.frequency.size // factor
        if groups == 0:
            raise ValueError("factor is larger than the number of frequency bins")
        stop = groups * factor
        frequency_groups = self.frequency[:stop].reshape(groups, factor)
        power_groups = self.power[:stop].reshape(groups, factor)
        shape_groups = self.bins_averaged[:stop].reshape(groups, factor)
        shapes = shape_groups.sum(axis=1)
        # Weight by the number of raw bins represented by each input estimate.
        # This preserves the Gamma sufficient statistic when already-binned
        # sections are combined and their local limit spectrum is constant.
        frequency = np.sum(frequency_groups * shape_groups, axis=1) / shapes
        power = np.sum(power_groups * shape_groups, axis=1) / shapes
        return PowerSpectrum(frequency, power, shapes)
