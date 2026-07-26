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

TESS_WINDOW_PROFILES = (
    "continuous",
    "momentum-dumps",
    "downlinks",
    "random-loss-matched",
    "tess-like",
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


@dataclass(frozen=True, slots=True)
class ObservingWindow:
    """A regularly sampled observing window.

    Parameters
    ----------
    observed
        Boolean mask whose true entries are retained observations.
    cadence_seconds
        Time between adjacent samples.
    label
        Short description stored in injection metadata.
    """

    observed: NDArray[np.bool_]
    cadence_seconds: float
    label: str = "custom"

    def __init__(
        self,
        observed: ArrayLike,
        cadence_seconds: float,
        label: str = "custom",
    ) -> None:
        """Validate and store an immutable observing window."""

        mask = np.asarray(observed)
        if mask.ndim != 1 or mask.size < 4:
            raise ValueError(
                "observed must be one-dimensional with at least four samples"
            )
        if mask.dtype.kind != "b":
            raise TypeError("observed must contain boolean values")
        if np.count_nonzero(mask) < 4:
            raise ValueError("observing window must retain at least four samples")
        cadence = _positive_parameter(cadence_seconds, "cadence_seconds")
        label = str(label)
        if not label:
            raise ValueError("label must not be empty")
        mask = mask.copy()
        mask.setflags(write=False)
        object.__setattr__(self, "observed", mask)
        object.__setattr__(self, "cadence_seconds", cadence)
        object.__setattr__(self, "label", label)

    @property
    def duty_cycle(self) -> float:
        """Return the fraction of scheduled samples that are observed."""

        return float(np.mean(self.observed))

    @property
    def duration_days(self) -> float:
        """Return the scheduled baseline in days."""

        return self.observed.size * self.cadence_seconds / 86400.0


@dataclass(frozen=True, slots=True)
class ObservingWindowDiagnostic:
    """Compact diagnostics for an observing mask.

    Attributes
    ----------
    duty_cycle
        Fraction of scheduled cadences retained.
    gap_count
        Number of distinct runs of missing cadences.
    maximum_gap_hours
        Duration of the longest uninterrupted gap.
    peak_sidelobe_power
        Largest non-zero-frequency spectral-window power relative to the
        zero-frequency peak.
    """

    duty_cycle: float
    gap_count: int
    maximum_gap_hours: float
    peak_sidelobe_power: float


def continuous_observing_window(
    duration_days: float,
    cadence_seconds: float,
) -> ObservingWindow:
    """Construct an uninterrupted regular observing window.

    Parameters
    ----------
    duration_days
        Scheduled observing baseline.
    cadence_seconds
        Regular sampling cadence.

    Returns
    -------
    ObservingWindow
        All-true observing mask.
    """

    samples = _sample_count(duration_days, cadence_seconds)
    return ObservingWindow(
        np.ones(samples, dtype=bool),
        cadence_seconds,
        label="continuous",
    )


def tess_like_observing_window(
    duration_days: float,
    cadence_seconds: float,
    *,
    downlink_interval_days: float = 13.7,
    downlink_duration_hours: float = 16.0,
    momentum_dump_interval_days: float = 2.5,
    momentum_dump_duration_minutes: float = 30.0,
    random_loss_fraction: float = 0.005,
    rng: np.random.Generator | int | None = None,
) -> ObservingWindow:
    """Construct a configurable TESS-like regular-cadence window.

    The mask combines periodic long downlink gaps, shorter momentum-dump gaps,
    and independent missing cadences. It is a controlled approximation for
    model-misspecification studies rather than a reconstruction of a specific
    target's quality flags.

    Parameters
    ----------
    duration_days, cadence_seconds
        Scheduled observing baseline and cadence.
    downlink_interval_days, downlink_duration_hours
        Spacing and duration of the longer periodic gaps.
    momentum_dump_interval_days, momentum_dump_duration_minutes
        Spacing and duration of shorter periodic interruptions.
    random_loss_fraction
        Independent probability that any otherwise observed cadence is lost.
    rng
        Random generator or seed controlling independent losses.

    Returns
    -------
    ObservingWindow
        Boolean observing mask labelled ``"tess-like"``.
    """

    samples = _sample_count(duration_days, cadence_seconds)
    downlink_interval = _positive_parameter(
        downlink_interval_days, "downlink_interval_days"
    )
    downlink_duration = _nonnegative_parameter(
        downlink_duration_hours, "downlink_duration_hours"
    ) / 24.0
    dump_interval = _positive_parameter(
        momentum_dump_interval_days, "momentum_dump_interval_days"
    )
    dump_duration = _nonnegative_parameter(
        momentum_dump_duration_minutes, "momentum_dump_duration_minutes"
    ) / 1440.0
    loss = float(random_loss_fraction)
    if not np.isfinite(loss) or not 0 <= loss < 1:
        raise ValueError("random_loss_fraction must be finite and in [0, 1)")

    time_days = np.arange(samples, dtype=float) * cadence_seconds / 86400.0
    observed = np.ones(samples, dtype=bool)
    _mask_periodic_gaps(
        observed,
        time_days,
        interval_days=downlink_interval,
        duration_days=downlink_duration,
    )
    _mask_periodic_gaps(
        observed,
        time_days,
        interval_days=dump_interval,
        duration_days=dump_duration,
    )
    if loss > 0:
        observed &= _as_generator(rng).random(samples) >= loss
    return ObservingWindow(observed, cadence_seconds, label="tess-like")


def tess_observing_window(
    duration_days: float,
    cadence_seconds: float,
    *,
    profile: str,
    momentum_dump_interval_days: float = 2.5,
    momentum_dump_duration_minutes: float = 30.0,
    rng: np.random.Generator | int | None = None,
) -> ObservingWindow:
    """Construct one component or control profile of a TESS-like window.

    Parameters
    ----------
    duration_days, cadence_seconds
        Scheduled observing baseline and cadence.
    profile
        One of ``"continuous"``, ``"momentum-dumps"``, ``"downlinks"``,
        ``"random-loss-matched"``, or ``"tess-like"``.
    momentum_dump_interval_days, momentum_dump_duration_minutes
        Spacing and duration of momentum-dump gaps. These affect profiles
        containing momentum dumps and their matched random-loss controls.
    rng
        Random generator or seed controlling independent cadence losses.

    Returns
    -------
    ObservingWindow
        Requested immutable observing mask.

    Notes
    -----
    The random-loss control removes exactly as many cadences as the combined
    TESS-like profile, but places them independently. It therefore matches
    duty cycle while removing the long, coherent gap structure.
    """

    profile = str(profile)
    if profile not in TESS_WINDOW_PROFILES:
        raise ValueError(
            "profile must be one of " + ", ".join(TESS_WINDOW_PROFILES)
        )
    if profile == "continuous":
        return continuous_observing_window(duration_days, cadence_seconds)
    if profile == "momentum-dumps":
        window = tess_like_observing_window(
            duration_days,
            cadence_seconds,
            downlink_duration_hours=0.0,
            momentum_dump_interval_days=momentum_dump_interval_days,
            momentum_dump_duration_minutes=momentum_dump_duration_minutes,
            random_loss_fraction=0.0,
            rng=rng,
        )
        return ObservingWindow(
            window.observed, cadence_seconds, label="momentum-dumps"
        )
    if profile == "downlinks":
        window = tess_like_observing_window(
            duration_days,
            cadence_seconds,
            momentum_dump_duration_minutes=0.0,
            random_loss_fraction=0.0,
            rng=rng,
        )
        return ObservingWindow(window.observed, cadence_seconds, label="downlinks")

    combined = tess_like_observing_window(
        duration_days,
        cadence_seconds,
        momentum_dump_interval_days=momentum_dump_interval_days,
        momentum_dump_duration_minutes=momentum_dump_duration_minutes,
        rng=rng,
    )
    if profile == "tess-like":
        return combined

    generator = _as_generator(rng)
    missing = int(np.count_nonzero(~combined.observed))
    observed = np.ones(combined.observed.size, dtype=bool)
    if missing:
        observed[generator.choice(observed.size, missing, replace=False)] = False
    return ObservingWindow(
        observed,
        cadence_seconds,
        label="random-loss-matched",
    )


def observing_window_diagnostics(
    window: ObservingWindow,
) -> ObservingWindowDiagnostic:
    """Measure duty cycle, gap structure, and the strongest window sidelobe.

    Parameters
    ----------
    window
        Regularly sampled observing mask.

    Returns
    -------
    ObservingWindowDiagnostic
        Scalar diagnostics describing missing time and spectral leakage.
    """

    if not isinstance(window, ObservingWindow):
        raise TypeError("window must be an ObservingWindow")
    missing = ~window.observed
    gap_starts = missing & np.concatenate(([True], ~missing[:-1]))
    gap_count = int(np.count_nonzero(gap_starts))
    maximum_gap = 0
    if gap_count:
        padded = np.concatenate(([False], missing, [False])).astype(np.int8)
        changes = np.flatnonzero(np.diff(padded))
        maximum_gap = int(np.max(changes[1::2] - changes[::2]))
    transform = np.fft.rfft(window.observed.astype(float))
    sidelobes = np.abs(transform[1:]) ** 2 / np.abs(transform[0]) ** 2
    return ObservingWindowDiagnostic(
        duty_cycle=window.duty_cycle,
        gap_count=gap_count,
        maximum_gap_hours=maximum_gap * window.cadence_seconds / 3600.0,
        peak_sidelobe_power=float(np.max(sidelobes, initial=0.0)),
    )


def simulate_windowed_periodogram(
    expected_power: ArrayLike,
    window: ObservingWindow,
    *,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    """Simulate a stochastic light curve and apply an observing window.

    A complex Gaussian Fourier realization is transformed to the time domain,
    multiplied by the observing mask, and transformed back. Dividing by the
    duty cycle preserves the mean white-noise power while gaps redistribute
    coherent spectral power and introduce correlations between Fourier bins.

    Parameters
    ----------
    expected_power
        Positive one-sided power density on the positive real-FFT frequencies,
        excluding zero frequency.
    window
        Regularly sampled observing mask.
    rng
        Random generator or seed.

    Returns
    -------
    numpy.ndarray
        Window-convolved positive-frequency periodogram.
    """

    if not isinstance(window, ObservingWindow):
        raise TypeError("window must be an ObservingWindow")
    expected = np.asarray(expected_power, dtype=float)
    expected_size = np.fft.rfftfreq(
        window.observed.size, window.cadence_seconds
    )[1:].size
    if expected.ndim != 1 or expected.size != expected_size:
        raise ValueError(
            "expected_power must match the positive frequencies of the window"
        )
    if not np.all(np.isfinite(expected)) or np.any(expected <= 0):
        raise ValueError("expected_power must contain finite positive values")

    generator = _as_generator(rng)
    realized_power = generator.exponential(expected)
    phase = generator.uniform(0.0, 2.0 * np.pi, expected.size)
    coefficients = np.zeros(expected.size + 1, dtype=complex)
    coefficients[1:] = np.sqrt(realized_power) * np.exp(1j * phase)
    if window.observed.size % 2 == 0:
        coefficients[-1] = (
            np.sqrt(realized_power[-1])
            * (1.0 if generator.random() >= 0.5 else -1.0)
        )

    light_curve = np.fft.irfft(coefficients, n=window.observed.size)
    if np.any(window.observed):
        light_curve = light_curve - np.mean(light_curve[window.observed])
    windowed = np.where(window.observed, light_curve, 0.0)
    transformed = np.fft.rfft(windowed)[1:]
    power = np.abs(transformed) ** 2 / window.duty_cycle
    return np.maximum(power, np.finfo(float).tiny)


def _sample_count(duration_days: float, cadence_seconds: float) -> int:
    """Validate an observing setup and return its number of samples."""

    duration = _positive_parameter(duration_days, "duration_days")
    cadence = _positive_parameter(cadence_seconds, "cadence_seconds")
    samples = int(np.floor(duration * 86400.0 / cadence))
    if samples < 4:
        raise ValueError("duration and cadence must provide at least four samples")
    return samples


def _mask_periodic_gaps(
    observed: NDArray[np.bool_],
    time_days: NDArray[np.float64],
    *,
    interval_days: float,
    duration_days: float,
) -> None:
    """Apply periodic gaps in place, beginning after one full interval."""

    if duration_days == 0:
        return
    gap_start = interval_days
    while gap_start < time_days[-1]:
        observed[
            (time_days >= gap_start) & (time_days < gap_start + duration_days)
        ] = False
        gap_start += interval_days


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
    window_profile
        ``"independent"`` for the original independent-bin simulation,
        ``"continuous"`` for a time-domain uninterrupted simulation, or one
        of the component, matched-random, or combined TESS window profiles in
        ``TESS_WINDOW_PROFILES``.
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
    window_profile: str = "independent"

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
        window_profile = str(
            parameters.get("window_profile", self.window_profile)
        )
        allowed_profiles = {"independent", *TESS_WINDOW_PROFILES}
        if window_profile not in allowed_profiles:
            raise ValueError(
                "window_profile must be independent or a supported TESS "
                "window profile"
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

        if window_profile == "independent":
            power = rng.exponential(mean)
            duty_cycle = 1.0
            gap_count = 0
            maximum_gap_hours = 0.0
            peak_sidelobe_power = 0.0
        else:
            window = tess_observing_window(
                duration,
                cadence,
                profile=window_profile,
                momentum_dump_interval_days=parameters.get(
                    "momentum_dump_interval_days", 2.5
                ),
                momentum_dump_duration_minutes=parameters.get(
                    "momentum_dump_duration_minutes", 30.0
                ),
                rng=parameters.get("window_seed", 0),
            )
            power = simulate_windowed_periodogram(mean, window, rng=rng)
            diagnostics = observing_window_diagnostics(window)
            duty_cycle = diagnostics.duty_cycle
            gap_count = diagnostics.gap_count
            maximum_gap_hours = diagnostics.maximum_gap_hours
            peak_sidelobe_power = diagnostics.peak_sidelobe_power
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
            "fwhm_envelope": medians["FWHM_env"],
            "window_profile": window_profile,
            "window_seed": int(parameters.get("window_seed", 0)),
            "duty_cycle": duty_cycle,
            "gap_count": gap_count,
            "maximum_gap_hours": maximum_gap_hours,
            "peak_sidelobe_power": peak_sidelobe_power,
            "momentum_dump_interval_days": float(
                parameters.get("momentum_dump_interval_days", 2.5)
            ),
            "momentum_dump_duration_minutes": float(
                parameters.get("momentum_dump_duration_minutes", 30.0)
            ),
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
