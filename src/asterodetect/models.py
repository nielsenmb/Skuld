"""Complete expected power-spectrum models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .components import GaussianEnvelope, HarveyComponent, _positive_finite


@dataclass(frozen=True, slots=True)
class SpectralModel:
    """White noise plus optional Harvey components and oscillation envelope."""

    white_noise: float
    harvey_components: tuple[HarveyComponent, ...] = ()
    envelope: GaussianEnvelope | None = None

    def __init__(
        self,
        white_noise: float,
        harvey_components: Iterable[HarveyComponent] = (),
        envelope: GaussianEnvelope | None = None,
    ) -> None:
        components = tuple(harvey_components)
        if not all(isinstance(component, HarveyComponent) for component in components):
            raise TypeError("harvey_components must contain only HarveyComponent values")
        if envelope is not None and not isinstance(envelope, GaussianEnvelope):
            raise TypeError("envelope must be a GaussianEnvelope or None")
        object.__setattr__(
            self, "white_noise", _positive_finite(white_noise, "white_noise")
        )
        object.__setattr__(self, "harvey_components", components)
        object.__setattr__(self, "envelope", envelope)

    def mean_spectrum(self, frequency: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the positive limit spectrum."""

        frequency_array = np.asarray(frequency, dtype=float)
        if frequency_array.ndim != 1:
            raise ValueError("frequency must be one-dimensional")
        if not np.all(np.isfinite(frequency_array)) or np.any(frequency_array < 0):
            raise ValueError("frequency must be finite and non-negative")

        result = np.full(frequency_array.shape, self.white_noise, dtype=float)
        for component in self.harvey_components:
            result += component(frequency_array)
        if self.envelope is not None:
            result += self.envelope(frequency_array)
        return result

