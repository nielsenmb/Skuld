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
    bin_lower: NDArray[np.float64]
    bin_upper: NDArray[np.float64]

    def __init__(
        self,
        frequency: ArrayLike,
        power: ArrayLike,
        bins_averaged: ArrayLike = 1.0,
        *,
        bin_lower: ArrayLike | None = None,
        bin_upper: ArrayLike | None = None,
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

        if (bin_lower is None) != (bin_upper is None):
            raise ValueError("bin_lower and bin_upper must be supplied together")
        if bin_lower is None:
            if frequency_array.size == 1:
                lower_array = frequency_array.copy()
                upper_array = frequency_array.copy()
            else:
                midpoints = 0.5 * (frequency_array[1:] + frequency_array[:-1])
                lower_array = np.concatenate(
                    ([frequency_array[0] - (midpoints[0] - frequency_array[0])], midpoints)
                )
                upper_array = np.concatenate(
                    (midpoints, [frequency_array[-1] + (frequency_array[-1] - midpoints[-1])])
                )
                lower_array = np.maximum(lower_array, 0.0)
        else:
            lower_array = np.asarray(bin_lower, dtype=float)
            upper_array = np.asarray(bin_upper, dtype=float)
            if lower_array.shape != frequency_array.shape or upper_array.shape != frequency_array.shape:
                raise ValueError("bin boundaries must match the spectrum shape")
            if (
                not np.all(np.isfinite(lower_array))
                or not np.all(np.isfinite(upper_array))
                or np.any(lower_array < 0)
                or np.any(upper_array <= lower_array)
                or np.any(frequency_array < lower_array)
                or np.any(frequency_array > upper_array)
            ):
                raise ValueError("bin boundaries must be finite and enclose each frequency")

        frequency_array.setflags(write=False)
        power_array.setflags(write=False)
        shape_array.setflags(write=False)
        lower_array.setflags(write=False)
        upper_array.setflags(write=False)
        object.__setattr__(self, "frequency", frequency_array)
        object.__setattr__(self, "power", power_array)
        object.__setattr__(self, "bins_averaged", shape_array)
        object.__setattr__(self, "bin_lower", lower_array)
        object.__setattr__(self, "bin_upper", upper_array)

    def select(self, minimum: float, maximum: float) -> "PowerSpectrum":
        """Return a frequency interval, including both endpoints."""

        if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
            raise ValueError("minimum and maximum must be finite with minimum < maximum")
        mask = (self.frequency >= minimum) & (self.frequency <= maximum)
        if not np.any(mask):
            raise ValueError("the selected interval contains no frequency bins")
        return PowerSpectrum(
            self.frequency[mask],
            self.power[mask],
            self.bins_averaged[mask],
            bin_lower=self.bin_lower[mask],
            bin_upper=self.bin_upper[mask],
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
        lower = self.bin_lower[:stop].reshape(groups, factor)[:, 0]
        upper = self.bin_upper[:stop].reshape(groups, factor)[:, -1]
        return PowerSpectrum(
            frequency, power, shapes, bin_lower=lower, bin_upper=upper
        )

    def bin_by_width(
        self,
        width: float,
        *,
        origin: float | None = None,
        drop_incomplete: bool = True,
    ) -> "PowerSpectrum":
        """Average the spectrum into fixed physical-frequency bins.

        Parameters
        ----------
        width
            Bin width in the same units as ``frequency``.
        origin
            Left edge of the first bin. By default the left edge of the
            input spectrum is used.
        drop_incomplete
            Discard a final bin that is truncated by the input range.

        Notes
        -----
        Input values are weighted by ``bins_averaged`` so rebinning an
        already averaged spectrum preserves its Gamma sufficient statistic.
        The bin edges are retained for averaging the spectral model over the
        same intervals.
        """

        width = float(width)
        if not np.isfinite(width) or width <= 0:
            raise ValueError("width must be finite and positive")
        if origin is None:
            origin = float(self.bin_lower[0])
        else:
            origin = float(origin)
        if not np.isfinite(origin) or origin < 0:
            raise ValueError("origin must be finite and non-negative")
        if origin > self.frequency[-1]:
            raise ValueError("origin lies above the spectrum")

        group_index = np.floor((self.frequency - origin) / width).astype(int)
        valid = group_index >= 0
        if not np.any(valid):
            raise ValueError("no frequencies fall within the requested bins")

        maximum_group = int(group_index[valid].max())
        if drop_incomplete:
            available_upper = float(self.bin_upper[-1])
            while (
                maximum_group >= 0
                and origin + (maximum_group + 1) * width > available_upper + 1e-12 * width
            ):
                maximum_group -= 1
        if maximum_group < 0:
            raise ValueError("the spectrum does not contain one complete bin")

        frequencies = []
        powers = []
        shapes = []
        lowers = []
        uppers = []
        for group in range(maximum_group + 1):
            mask = valid & (group_index == group)
            if not np.any(mask):
                continue
            weights = self.bins_averaged[mask]
            total_weight = float(np.sum(weights))
            frequencies.append(float(np.sum(self.frequency[mask] * weights) / total_weight))
            powers.append(float(np.sum(self.power[mask] * weights) / total_weight))
            shapes.append(total_weight)
            lowers.append(origin + group * width)
            uppers.append(origin + (group + 1) * width)

        return PowerSpectrum(
            frequencies,
            powers,
            shapes,
            bin_lower=lowers,
            bin_upper=uppers,
        )
