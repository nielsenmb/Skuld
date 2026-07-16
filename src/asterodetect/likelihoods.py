"""Periodogram likelihoods."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln

from .data import PowerSpectrum
from .models import SpectralModel


def gamma_log_likelihood(
    power: ArrayLike,
    mean_power: ArrayLike,
    shape: ArrayLike = 1.0,
) -> float:
    """Return the independent-bin Gamma log-likelihood.

    The distribution is parameterized to have expectation ``mean_power`` and
    variance ``mean_power**2 / shape``.  ``shape=1`` recovers the exponential
    likelihood of an unbinned periodogram.
    """

    observed = np.asarray(power, dtype=float)
    expected = np.asarray(mean_power, dtype=float)
    gamma_shape = np.asarray(shape, dtype=float)
    try:
        observed, expected, gamma_shape = np.broadcast_arrays(
            observed, expected, gamma_shape
        )
    except ValueError as error:
        raise ValueError("power, mean_power, and shape must broadcast together") from error

    if (
        not np.all(np.isfinite(observed))
        or not np.all(np.isfinite(expected))
        or not np.all(np.isfinite(gamma_shape))
        or np.any(observed <= 0)
        or np.any(expected <= 0)
        or np.any(gamma_shape < 1)
    ):
        raise ValueError(
            "power and mean_power must be positive; shape must be at least one"
        )

    terms = (
        gamma_shape * np.log(gamma_shape)
        - gammaln(gamma_shape)
        + (gamma_shape - 1.0) * np.log(observed)
        - gamma_shape * np.log(expected)
        - gamma_shape * observed / expected
    )
    return float(np.sum(terms))


def model_log_likelihood(spectrum: PowerSpectrum, model: SpectralModel) -> float:
    """Evaluate a complete spectral model against a power spectrum."""

    expected = model.mean_spectrum(spectrum.frequency)
    return gamma_log_likelihood(
        spectrum.power, expected, spectrum.bins_averaged
    )

