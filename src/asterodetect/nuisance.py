"""Target-specific nuisance priors for the end-to-end detector."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .data import PowerSpectrum


@dataclass(frozen=True, slots=True)
class NuisancePrior:
    """Weakly informative priors around observable spectral quantities.

    Scatter parameters are standard deviations in natural-log space.  The
    white-noise prior is centred on a robust estimate from the highest
    frequency fraction of the supplied spectrum.
    """

    white_noise_log_scatter: float = 0.35
    envelope_log_scatter: float = 0.35
    granulation_log_scatter: float = 0.25
    granulation_split_concentration: float = 20.0
    overdispersion_log_scatter: float = 0.25
    high_frequency_fraction: float = 0.2

    def __post_init__(self) -> None:
        for name in (
            "white_noise_log_scatter",
            "envelope_log_scatter",
            "granulation_log_scatter",
            "overdispersion_log_scatter",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        concentration = float(self.granulation_split_concentration)
        if not np.isfinite(concentration) or concentration <= 0:
            raise ValueError(
                "granulation_split_concentration must be finite and positive"
            )
        object.__setattr__(self, "granulation_split_concentration", concentration)
        fraction = float(self.high_frequency_fraction)
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("high_frequency_fraction must be in (0, 1]")
        object.__setattr__(self, "high_frequency_fraction", fraction)

    def estimate_white_noise(self, spectrum: PowerSpectrum) -> float:
        """Estimate the white floor from the upper end of a PSD."""

        count = max(1, int(np.ceil(len(spectrum.power) * self.high_frequency_fraction)))
        return float(np.median(spectrum.power[-count:]))

    def sample(
        self,
        spectrum: PowerSpectrum,
        size: int,
        *,
        rng: np.random.Generator | int | None = None,
        white_noise_centre: float | None = None,
    ) -> Mapping[str, NDArray[np.float64]]:
        """Draw nuisance quantities used by one prior-predictive integral."""

        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise TypeError("size must be an integer")
        if size < 1:
            raise ValueError("size must be positive")
        generator = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )
        centre = (
            self.estimate_white_noise(spectrum)
            if white_noise_centre is None
            else float(white_noise_centre)
        )
        if not np.isfinite(centre) or centre <= 0:
            raise ValueError("white_noise_centre must be finite and positive")

        def lognormal(scatter: float) -> NDArray[np.float64]:
            # Mean-one multiplier, so widening the prior does not move its mean.
            return generator.lognormal(-0.5 * scatter**2, scatter, size)

        alpha = 0.5 * self.granulation_split_concentration
        draws = {
            "white_noise": centre * lognormal(self.white_noise_log_scatter),
            "envelope_scale": lognormal(self.envelope_log_scatter),
            "granulation_scale": lognormal(self.granulation_log_scatter),
            "granulation_variance_fraction_low": generator.beta(alpha, alpha, size),
            "overdispersion": np.exp(
                np.abs(
                    generator.normal(
                        0.0, self.overdispersion_log_scatter, size
                    )
                )
            ),
        }
        for values in draws.values():
            values.setflags(write=False)
        return MappingProxyType(draws)
