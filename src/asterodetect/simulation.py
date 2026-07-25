"""Simulation helpers for calibration and injection-recovery tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .asteroscale import AsteroScaleSamples
from .calibration import InjectionCase
from .components import HarveyComponent
from .data import PowerSpectrum
from .models import SpectralModel
from .observation import ObservationModel, cadence_amplitude_response


def _as_generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Normalize a generator, integer seed, or ``None`` to a generator."""

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
    """Draw independent Gamma-distributed periodogram powers.

    Parameters
    ----------
    frequency
        Frequency grid.
    model
        Complete expected spectral model.
    bins_averaged
        Gamma shape or number of averaged raw bins.
    rng
        Random generator or seed.

    Returns
    -------
    numpy.ndarray
        Simulated positive periodogram powers.
    """

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


DEFAULT_MODE_VISIBILITIES = MappingProxyType(
    {0: 1.0, 1: 1.5, 2: 0.5, 3: 0.04}
)


def regular_frequency_grid(
    duration_days: float,
    cadence_seconds: float,
) -> NDArray[np.float64]:
    """Return positive Fourier frequencies in microhertz up to Nyquist.

    Parameters
    ----------
    duration_days
        Observation duration in days.
    cadence_seconds
        Regular sampling cadence in seconds.

    Returns
    -------
    numpy.ndarray
        Positive real-FFT frequency grid.
    """

    duration_days = float(duration_days)
    cadence_seconds = float(cadence_seconds)
    if not np.isfinite(duration_days) or duration_days <= 0:
        raise ValueError("duration_days must be finite and positive")
    if not np.isfinite(cadence_seconds) or cadence_seconds <= 0:
        raise ValueError("cadence_seconds must be finite and positive")
    samples = int(np.floor(duration_days * 86400.0 / cadence_seconds))
    if samples < 4:
        raise ValueError("duration and cadence must provide at least four samples")
    return np.fft.rfftfreq(samples, cadence_seconds)[1:] * 1.0e6


def lorentzian_mode_comb(
    frequency: ArrayLike,
    *,
    numax: float,
    dnu: float,
    fwhm_envelope: float,
    radial_mode_rms_amplitude: float,
    linewidth: float,
    mode_visibilities: Mapping[int, float] = DEFAULT_MODE_VISIBILITIES,
) -> NDArray[np.float64]:
    """Construct an approximate resolved stochastic-mode limit spectrum.

    Parameters
    ----------
    frequency
        Non-negative frequency grid.
    numax, dnu, fwhm_envelope
        Global seismic and envelope parameters.
    radial_mode_rms_amplitude
        Maximum radial-mode RMS amplitude.
    linewidth
        Mode FWHM.
    mode_visibilities
        Relative integrated powers keyed by angular degree.

    Returns
    -------
    numpy.ndarray
        Expected Lorentzian mode-comb power density.
    """

    frequency_array = np.asarray(frequency, dtype=float)
    scalars = {
        "numax": numax,
        "dnu": dnu,
        "fwhm_envelope": fwhm_envelope,
        "radial_mode_rms_amplitude": radial_mode_rms_amplitude,
        "linewidth": linewidth,
    }
    for name, value in scalars.items():
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        scalars[name] = value
    if (
        frequency_array.ndim != 1
        or not np.all(np.isfinite(frequency_array))
        or np.any(frequency_array < 0)
    ):
        raise ValueError("frequency must be finite, non-negative, and one-dimensional")
    if not mode_visibilities or 0 not in mode_visibilities:
        raise ValueError("mode_visibilities must include radial modes")
    if any(
        not isinstance(degree, int)
        or not np.isfinite(visibility)
        or visibility < 0
        for degree, visibility in mode_visibilities.items()
    ):
        raise ValueError(
            "mode visibilities must have integer degrees and non-negative values"
        )

    numax = scalars["numax"]
    dnu = scalars["dnu"]
    fwhm_envelope = scalars["fwhm_envelope"]
    amplitude = scalars["radial_mode_rms_amplitude"]
    half_width = 0.5 * scalars["linewidth"]
    sigma = fwhm_envelope / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    offsets = {0: 0.0, 1: 0.5, 2: -0.12, 3: 0.80}
    margin = 4.0 * fwhm_envelope
    n_min = int(np.floor((max(0.0, frequency_array[0] - margin) - numax) / dnu))
    n_max = int(np.ceil((frequency_array[-1] + margin - numax) / dnu))
    result = np.zeros_like(frequency_array)
    for order in range(n_min, n_max + 1):
        radial_frequency = numax + order * dnu
        for degree, visibility in mode_visibilities.items():
            if visibility == 0:
                continue
            mode_frequency = radial_frequency + offsets.get(degree, 0.0) * dnu
            if mode_frequency <= 0:
                continue
            envelope_weight = np.exp(
                -0.5 * ((mode_frequency - numax) / sigma) ** 2
            )
            integrated_power = amplitude**2 * visibility * envelope_weight
            result += (
                integrated_power
                * half_width
                / np.pi
                / ((frequency_array - mode_frequency) ** 2 + half_width**2)
            )
    return result


@dataclass(frozen=True, slots=True)
class AstrophysicalInjectionFactory:
    """Create regular-sampling injections from joint AsteroScale samples.

    Parameters
    ----------
    stellar_samples
        Joint AsteroScale samples defining the injected star.
    duration_days, cadence_seconds
        Default observing configuration.
    white_noise
        Default white-noise power density.
    amplitude_scale, granulation_scale
        Multipliers relative to AsteroScale predictions.
    dilution
        Target fraction of aperture flux.
    linewidth
        Optional fixed mode linewidth.
    bolometric_correction
        Granulation bolometric correction.
    """

    stellar_samples: AsteroScaleSamples
    duration_days: float = 27.4
    cadence_seconds: float = 120.0
    white_noise: float = 0.1
    amplitude_scale: float = 1.0
    granulation_scale: float = 1.0
    dilution: float = 1.0
    linewidth: float | None = None
    bolometric_correction: float = 1.0

    def __post_init__(self) -> None:
        """Validate the AsteroScale sample container."""

        if not isinstance(self.stellar_samples, AsteroScaleSamples):
            raise TypeError("stellar_samples must be AsteroScaleSamples")

    def __call__(
        self,
        name: str,
        parameters: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> InjectionCase:
        """Generate one stochastic astrophysical injection.

        Parameters
        ----------
        name
            Unique case identifier.
        parameters
            Overrides for truth, observing setup, and amplitude scales.
        rng
            Random generator for the periodogram realization.

        Returns
        -------
        InjectionCase
            Simulated spectrum with recovery constraints and metadata.
        """

        truth = str(parameters.get("truth", "oscillation"))
        if truth not in {"noise", "granulation", "oscillation"}:
            raise ValueError("truth must be noise, granulation, or oscillation")
        duration = _positive_parameter(
            parameters.get("duration_days", self.duration_days), "duration_days"
        )
        cadence = _positive_parameter(
            parameters.get("cadence_seconds", self.cadence_seconds),
            "cadence_seconds",
        )
        white_noise = _positive_parameter(
            parameters.get("white_noise", self.white_noise), "white_noise"
        )
        amplitude_scale = _nonnegative_parameter(
            parameters.get("amplitude_scale", self.amplitude_scale),
            "amplitude_scale",
        )
        granulation_scale = _nonnegative_parameter(
            parameters.get("granulation_scale", self.granulation_scale),
            "granulation_scale",
        )
        dilution = _positive_parameter(
            parameters.get("dilution", self.dilution), "dilution"
        )
        if dilution > 1:
            raise ValueError("dilution must not exceed one")
        bolometric_correction = _positive_parameter(
            parameters.get("bolometric_correction", self.bolometric_correction),
            "bolometric_correction",
        )

        medians = {
            key: float(np.median(value))
            for key, value in self.stellar_samples.values.items()
        }
        linewidth_default = max(
            1.0 / (duration * 86400.0) * 1.0e6,
            0.001 * medians["numax"],
        )
        linewidth = _positive_parameter(
            parameters.get(
                "linewidth",
                self.linewidth if self.linewidth is not None else linewidth_default,
            ),
            "linewidth",
        )
        frequency = regular_frequency_grid(duration, cadence)
        mean = np.full(frequency.shape, white_noise)
        response_squared = cadence_amplitude_response(frequency, cadence) ** 2

        if truth != "noise" and granulation_scale > 0:
            observation = ObservationModel(
                dilution=dilution,
                bolometric_correction=bolometric_correction,
            )
            low_amp, high_amp = observation.granulation_amplitudes(
                medians["A_gran"] * granulation_scale
            )
            for amplitude, characteristic_frequency in zip(
                (float(low_amp), float(high_amp)),
                (medians["b_gran_low"], medians["b_gran_high"]),
                strict=True,
            ):
                mean += (
                    HarveyComponent.from_rms_amplitude(
                        amplitude, characteristic_frequency
                    )(frequency)
                    * response_squared
                )

        if truth == "oscillation" and amplitude_scale > 0:
            mean += (
                lorentzian_mode_comb(
                    frequency,
                    numax=medians["numax"],
                    dnu=medians["dnu"],
                    fwhm_envelope=medians["FWHM_env"],
                    radial_mode_rms_amplitude=(
                        medians["A_env"] * amplitude_scale * dilution
                    ),
                    linewidth=linewidth,
                )
                * response_squared
            )

        power = rng.exponential(mean)
        metadata = {
            "truth": truth,
            "duration_days": duration,
            "cadence_seconds": cadence,
            "white_noise": white_noise,
            "amplitude_scale": amplitude_scale,
            "granulation_scale": granulation_scale,
            "dilution": dilution,
            "linewidth": linewidth,
            "numax": medians["numax"],
            "dnu": medians["dnu"],
        }
        return InjectionCase(
            name=name,
            truth=truth,
            spectrum=PowerSpectrum(frequency, power),
            stellar_constraints=self.stellar_samples,
            metadata=MappingProxyType(metadata),
        )


def _positive_parameter(value: Any, name: str) -> float:
    """Validate and return a positive finite scalar parameter."""

    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_parameter(value: Any, name: str) -> float:
    """Validate and return a non-negative finite scalar parameter."""

    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result
