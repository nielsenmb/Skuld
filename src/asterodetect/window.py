"""Spectral-window operators for target-specific forward models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import irfft, rfft

from .data import PowerSpectrum


class ObservingWindowLike(Protocol):
    """Structural type required by :class:`SpectralWindowOperator`."""

    observed: NDArray[np.bool_]
    cadence_seconds: float


@dataclass(frozen=True, slots=True)
class SpectralWindowOperator:
    r"""Pass expected Fourier power through a regularly sampled time mask.

    The operator implements the expectation

    .. math::

        E[|Y_k|^2] =
        \frac{1}{N^2 d}
        \sum_j |W_{k-j}|^2 E[|X_j|^2],

    where ``N`` is the scheduled sample count, ``d`` is the duty cycle, and
    ``W`` is the discrete Fourier transform of the boolean observing mask.
    The normalization matches :func:`simulate_windowed_periodogram`.

    Parameters
    ----------
    observed
        Boolean mask on a regular time grid.
    cadence_seconds
        Sampling interval.
    row_batch_size
        Maximum number of expected spectra transformed together.
    fft_workers
        Worker count passed to SciPy's FFT implementation. Negative values
        count backwards from the available CPU count.
    """

    observed: NDArray[np.bool_]
    cadence_seconds: float
    row_batch_size: int = 32
    fft_workers: int = 1
    _frequency: NDArray[np.float64] = field(init=False, repr=False)
    _window_transfer: NDArray[np.complex128] = field(
        init=False,
        repr=False,
    )

    def __init__(
        self,
        observed: ArrayLike,
        cadence_seconds: float,
        *,
        row_batch_size: int = 32,
        fft_workers: int = 1,
    ) -> None:
        """Validate and retain an immutable observing mask."""

        mask = np.asarray(observed)
        if mask.ndim != 1 or mask.size < 4:
            raise ValueError(
                "observed must be one-dimensional with at least four samples"
            )
        if mask.dtype.kind != "b":
            raise TypeError("observed must contain boolean values")
        if np.count_nonzero(mask) < 4:
            raise ValueError("observing window must retain at least four samples")
        cadence = float(cadence_seconds)
        if not np.isfinite(cadence) or cadence <= 0:
            raise ValueError("cadence_seconds must be finite and positive")
        if (
            isinstance(row_batch_size, bool)
            or not isinstance(row_batch_size, (int, np.integer))
            or row_batch_size < 1
        ):
            raise ValueError("row_batch_size must be a positive integer")
        if (
            isinstance(fft_workers, bool)
            or not isinstance(fft_workers, (int, np.integer))
            or fft_workers == 0
        ):
            raise ValueError("fft_workers must be a non-zero integer")
        mask = mask.copy()
        mask.setflags(write=False)
        frequency = (
            np.fft.rfftfreq(mask.size, cadence)[1:] * 1.0e6
        )
        frequency.setflags(write=False)
        window_power = np.abs(np.fft.fft(mask.astype(float))) ** 2
        transfer = rfft(window_power, workers=int(fft_workers))
        transfer.setflags(write=False)
        object.__setattr__(self, "observed", mask)
        object.__setattr__(self, "cadence_seconds", cadence)
        object.__setattr__(self, "row_batch_size", int(row_batch_size))
        object.__setattr__(self, "fft_workers", int(fft_workers))
        object.__setattr__(self, "_frequency", frequency)
        object.__setattr__(self, "_window_transfer", transfer)

    @classmethod
    def from_observing_window(
        cls,
        window: ObservingWindowLike,
        *,
        row_batch_size: int = 32,
        fft_workers: int = 1,
    ) -> "SpectralWindowOperator":
        """Construct an operator from an observing-window-like object."""

        try:
            observed = window.observed
            cadence_seconds = window.cadence_seconds
        except AttributeError as error:
            raise TypeError(
                "window must provide observed and cadence_seconds"
            ) from error
        return cls(
            observed,
            cadence_seconds,
            row_batch_size=row_batch_size,
            fft_workers=fft_workers,
        )

    @property
    def duty_cycle(self) -> float:
        """Return the fraction of scheduled cadences retained."""

        return float(np.mean(self.observed))

    @property
    def frequency(self) -> NDArray[np.float64]:
        """Return positive Fourier frequencies in microhertz."""

        return self._frequency

    def convolve(self, expected_power: ArrayLike) -> NDArray[np.float64]:
        """Convolve one or more positive-frequency expected spectra.

        Parameters
        ----------
        expected_power
            Array whose final dimension matches :attr:`frequency`. A
            one-dimensional input returns a one-dimensional result.

        Returns
        -------
        numpy.ndarray
            Window-convolved expected power with the same shape as the input.
        """

        expected = np.asarray(expected_power, dtype=float)
        one_dimensional = expected.ndim == 1
        if one_dimensional:
            expected = expected[None, :]
        if expected.ndim != 2 or expected.shape[1] != self.frequency.size:
            raise ValueError(
                "expected_power must be one- or two-dimensional and match "
                "the positive frequencies of the window"
            )
        if not np.all(np.isfinite(expected)) or np.any(expected < 0):
            raise ValueError(
                "expected_power must contain finite non-negative values"
            )

        sample_count = self.observed.size
        positive_count = self.frequency.size
        output = np.empty_like(expected)
        for start in range(0, expected.shape[0], self.row_batch_size):
            stop = min(start + self.row_batch_size, expected.shape[0])
            positive = expected[start:stop]
            two_sided = np.zeros((stop - start, sample_count), dtype=float)
            two_sided[:, 1 : positive_count + 1] = positive
            if sample_count % 2 == 0:
                two_sided[:, positive_count + 1 :] = positive[:, :-1][:, ::-1]
            else:
                two_sided[:, positive_count + 1 :] = positive[:, ::-1]
            convolved = irfft(
                rfft(
                    two_sided,
                    axis=1,
                    workers=self.fft_workers,
                )
                * self._window_transfer[None, :],
                n=sample_count,
                axis=1,
                workers=self.fft_workers,
            )
            convolved /= sample_count**2 * self.duty_cycle
            output[start:stop] = np.maximum(
                convolved[:, 1 : positive_count + 1],
                0.0,
            )
        return output[0] if one_dimensional else output

    def convolve_and_bin(
        self,
        expected_power: ArrayLike,
        spectrum: PowerSpectrum,
    ) -> NDArray[np.float64]:
        """Convolve expected power and average it over observed PSD bins.

        Parameters
        ----------
        expected_power
            One or more spectra on :attr:`frequency`.
        spectrum
            Binned observed spectrum defining the output intervals.

        Returns
        -------
        numpy.ndarray
            Expected bin means. The leading dimensions match
            ``expected_power``.
        """

        if not isinstance(spectrum, PowerSpectrum):
            raise TypeError("spectrum must be a PowerSpectrum")
        expected = np.asarray(expected_power, dtype=float)
        one_dimensional = expected.ndim == 1
        convolved = self.convolve(expected)
        if one_dimensional:
            convolved = convolved[None, :]
        frequency = self.frequency
        starts = np.searchsorted(frequency, spectrum.bin_lower, side="left")
        stops = np.searchsorted(frequency, spectrum.bin_upper, side="left")
        if (
            np.any(starts >= stops)
            or np.any(starts < 0)
            or np.any(stops > frequency.size)
        ):
            raise ValueError(
                "each observed bin must contain at least one window frequency"
            )
        cumulative = np.pad(
            np.cumsum(convolved, axis=1),
            ((0, 0), (1, 0)),
        )
        result = (cumulative[:, stops] - cumulative[:, starts]) / (
            stops - starts
        )[None, :]
        return result[0] if one_dimensional else result
