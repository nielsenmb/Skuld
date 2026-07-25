"""Target-specific nuisance priors for the end-to-end detector."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta, norm

from .data import PowerSpectrum


@dataclass(frozen=True, slots=True)
class NuisancePrior:
    """Weakly informative priors around observable spectral quantities.

    Scatter parameters are standard deviations in natural-log space.  The
    white-noise prior is centred on a robust estimate from the highest
    frequency fraction of the supplied spectrum.

    Parameters
    ----------
    white_noise_log_scatter
        Natural-log scatter of the white-noise level.
    envelope_log_scatter
        Natural-log scatter of the oscillation amplitude multiplier.
    granulation_log_scatter
        Natural-log scatter of the granulation amplitude multiplier.
    granulation_split_concentration
        Symmetric Beta concentration for the Harvey variance split.
    overdispersion_log_scatter
        Natural-log scatter controlling overdispersion.
    high_frequency_fraction
        Fraction of high-frequency PSD bins used to estimate white noise.
    """

    white_noise_log_scatter: float = 0.35
    envelope_log_scatter: float = 0.35
    granulation_log_scatter: float = 0.25
    granulation_split_concentration: float = 20.0
    overdispersion_log_scatter: float = 0.25
    high_frequency_fraction: float = 0.2

    latent_names = (
        "white_noise",
        "envelope_scale",
        "granulation_scale",
        "granulation_variance_fraction_low",
        "overdispersion",
    )

    def __post_init__(self) -> None:
        """Validate nuisance-prior hyperparameters."""

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
        """Estimate the white floor from the upper end of a PSD.

        Parameters
        ----------
        spectrum
            Observed power spectrum.

        Returns
        -------
        float
            Median power in the configured high-frequency fraction.
        """

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
        """Draw nuisance quantities used by one prior-predictive integral.

        Parameters
        ----------
        spectrum
            Observed power spectrum used for the default noise centre.
        size
            Number of aligned nuisance rows.
        rng
            Random generator or seed.
        white_noise_centre
            Optional positive noise-prior centre.

        Returns
        -------
        mapping
            Immutable transformed nuisance arrays.
        """

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

        latent = generator.normal(size=(size, len(self.latent_names)))
        draws = dict(self.transform_latent(latent, white_noise_centre=centre))
        for values in draws.values():
            values.setflags(write=False)
        return MappingProxyType(draws)

    def transform_latent(
        self,
        latent: NDArray[np.float64],
        *,
        white_noise_centre: float,
    ) -> Mapping[str, NDArray[np.float64]]:
        """Transform standard-Normal coordinates to nuisance values.

        Parameters
        ----------
        latent
            Array with one column per name in ``latent_names``.
        white_noise_centre
            Positive centre of the white-noise prior.

        Returns
        -------
        mapping
            Transformed nuisance arrays aligned with input rows.
        """

        values = np.asarray(latent, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.latent_names)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                f"latent must be finite with shape (n, {len(self.latent_names)})"
            )
        centre = float(white_noise_centre)
        if not np.isfinite(centre) or centre <= 0:
            raise ValueError("white_noise_centre must be finite and positive")

        def lognormal(z: NDArray[np.float64], scatter: float) -> NDArray[np.float64]:
            """Return a mean-one lognormal multiplier."""

            # Mean-one multiplier, so widening the prior does not move its mean.
            return np.exp(-0.5 * scatter**2 + scatter * z)

        alpha = 0.5 * self.granulation_split_concentration
        probability = np.clip(
            norm.cdf(values[:, 3]),
            np.finfo(float).eps,
            1 - np.finfo(float).eps,
        )
        return MappingProxyType(
            {
                "white_noise": centre
                * lognormal(values[:, 0], self.white_noise_log_scatter),
                "envelope_scale": lognormal(
                    values[:, 1], self.envelope_log_scatter
                ),
                "granulation_scale": lognormal(
                    values[:, 2], self.granulation_log_scatter
                ),
                "granulation_variance_fraction_low": beta.ppf(
                    probability, alpha, alpha
                ),
                "overdispersion": np.exp(
                    self.overdispersion_log_scatter * np.abs(values[:, 4])
                ),
            }
        )
