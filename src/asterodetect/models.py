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
    overdispersion: float = 1.0

    def __init__(
        self,
        white_noise: float,
        harvey_components: Iterable[HarveyComponent] = (),
        envelope: GaussianEnvelope | None = None,
        overdispersion: float = 1.0,
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
        overdispersion = _positive_finite(overdispersion, "overdispersion")
        if overdispersion < 1:
            raise ValueError("overdispersion must be at least one")
        object.__setattr__(self, "overdispersion", overdispersion)

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

    def mean_binned_spectrum(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        *,
        quadrature_order: int = 16,
    ) -> NDArray[np.float64]:
        """Average the limit spectrum over fixed frequency intervals."""

        from scipy.special import erf

        lower_array = np.asarray(lower, dtype=float)
        upper_array = np.asarray(upper, dtype=float)
        if lower_array.shape != upper_array.shape or lower_array.ndim != 1:
            raise ValueError("lower and upper must be matching one-dimensional arrays")
        if (
            not np.all(np.isfinite(lower_array))
            or not np.all(np.isfinite(upper_array))
            or np.any(lower_array < 0)
            or np.any(upper_array <= lower_array)
        ):
            raise ValueError("each bin must have finite bounds with 0 <= lower < upper")
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")

        widths = upper_array - lower_array
        result = np.full(lower_array.shape, self.white_noise, dtype=float)
        if self.harvey_components:
            nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
            samples = (
                0.5 * widths[:, None] * nodes[None, :]
                + 0.5 * (upper_array + lower_array)[:, None]
            )
            for component in self.harvey_components:
                result += 0.5 * np.sum(
                    component(samples) * weights[None, :], axis=1
                )
        if self.envelope is not None:
            scale = np.sqrt(2.0) * self.envelope.sigma
            probability = 0.5 * (
                erf((upper_array - self.envelope.numax) / scale)
                - erf((lower_array - self.envelope.numax) / scale)
            )
            result += self.envelope.integrated_power * probability / widths
        return result
