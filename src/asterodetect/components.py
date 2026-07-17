"""Components of the expected (limit) power-density spectrum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True, slots=True)
class HarveyComponent:
    """A Harvey-like background component.

    ``power`` is the zero-frequency power-density level.  This explicit
    parameterization avoids silently mixing PSD height and integrated RMS.
    """

    power: float
    characteristic_frequency: float
    exponent: float = 4.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "power", _positive_finite(self.power, "power"))
        object.__setattr__(
            self,
            "characteristic_frequency",
            _positive_finite(
                self.characteristic_frequency, "characteristic_frequency"
            ),
        )
        object.__setattr__(self, "exponent", _positive_finite(self.exponent, "exponent"))

    @classmethod
    def from_rms_amplitude(
        cls,
        amplitude: float,
        characteristic_frequency: float,
        exponent: float = 4.0,
    ) -> "HarveyComponent":
        """Construct a normalized Harvey profile from its integrated RMS.

        The resulting one-sided profile integrates from zero to infinity to
        ``amplitude**2``.  For an exponent of four this is the normalized
        Kallinger super-Lorentzian used by AsteroScale.
        """

        amplitude = _positive_finite(amplitude, "amplitude")
        characteristic_frequency = _positive_finite(
            characteristic_frequency, "characteristic_frequency"
        )
        exponent = _positive_finite(exponent, "exponent")
        if exponent <= 1:
            raise ValueError("exponent must exceed one for finite integrated power")
        normalization = exponent * np.sin(np.pi / exponent) / np.pi
        power = (
            normalization
            * amplitude**2
            / characteristic_frequency
        )
        return cls(power, characteristic_frequency, exponent)

    def __call__(self, frequency: ArrayLike) -> NDArray[np.float64]:
        frequency_array = np.asarray(frequency, dtype=float)
        if np.any(frequency_array < 0) or not np.all(np.isfinite(frequency_array)):
            raise ValueError("frequency must be finite and non-negative")
        ratio = frequency_array / self.characteristic_frequency
        return self.power / (1.0 + ratio**self.exponent)


@dataclass(frozen=True, slots=True)
class GaussianEnvelope:
    """A Gaussian power envelope parameterized by integrated power."""

    integrated_power: float
    numax: float
    sigma: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "integrated_power",
            _positive_finite(self.integrated_power, "integrated_power"),
        )
        object.__setattr__(self, "numax", _positive_finite(self.numax, "numax"))
        object.__setattr__(self, "sigma", _positive_finite(self.sigma, "sigma"))

    @property
    def height(self) -> float:
        """Peak power density implied by the integrated power."""

        return self.integrated_power / (np.sqrt(2.0 * np.pi) * self.sigma)

    @property
    def fwhm(self) -> float:
        """Full width at half maximum."""

        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * self.sigma

    def __call__(self, frequency: ArrayLike) -> NDArray[np.float64]:
        frequency_array = np.asarray(frequency, dtype=float)
        if np.any(frequency_array < 0) or not np.all(np.isfinite(frequency_array)):
            raise ValueError("frequency must be finite and non-negative")
        offset = (frequency_array - self.numax) / self.sigma
        return self.height * np.exp(-0.5 * offset**2)
