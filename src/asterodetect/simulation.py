"""Simulation helpers for calibration and injection-recovery tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import SpectralModel


def _as_generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def simulate_periodogram(
    frequency: ArrayLike,
    model: SpectralModel,
    *,
    bins_averaged: ArrayLike = 1.0,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    """Draw independent Gamma-distributed periodogram powers."""

    frequency_array = np.asarray(frequency, dtype=float)
    expected = model.mean_spectrum(frequency_array)
    shapes = np.asarray(bins_averaged, dtype=float)
    try:
        shapes = np.broadcast_to(shapes, expected.shape)
    except ValueError as error:
        raise ValueError("bins_averaged must broadcast to frequency") from error
    if not np.all(np.isfinite(shapes)) or np.any(shapes < 1):
        raise ValueError("bins_averaged must be finite and at least one")
    return _as_generator(rng).gamma(shape=shapes, scale=expected / shapes)

