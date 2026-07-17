"""Map intrinsic AsteroScale amplitudes onto an observed power spectrum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


# Effective sum of relative mode powers in one radial order.  This value also
# maps AsteroScale's 2.1 ppm solar TESS radial-mode RMS amplitude to the
# approximately 0.1 ppm^2/microhertz envelope height used by Nielsen et al.
DEFAULT_TOTAL_MODE_VISIBILITY = 3.04


def _positive_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return array


def cadence_amplitude_response(
    frequency: ArrayLike,
    integration_time_seconds: float,
) -> NDArray[np.float64]:
    """Return finite-integration-time attenuation of sinusoid amplitude.

    Frequencies are in microhertz.  ``numpy.sinc`` uses the normalized sinc,
    so this is ``sin(pi * nu * dt) / (pi * nu * dt)``.  When integration time
    equals sampling cadence, it is equivalent to Eq. B.10 of Nielsen et al.
    (2022).  Power is attenuated by the square of this response.
    """

    frequency_array = np.asarray(frequency, dtype=float)
    if not np.all(np.isfinite(frequency_array)) or np.any(frequency_array < 0):
        raise ValueError("frequency must be finite and non-negative")
    integration_time_seconds = float(integration_time_seconds)
    if not np.isfinite(integration_time_seconds) or integration_time_seconds <= 0:
        raise ValueError("integration_time_seconds must be finite and positive")
    return np.sinc(frequency_array * integration_time_seconds * 1.0e-6)


def envelope_integrated_power(
    radial_mode_rms_amplitude: ArrayLike,
    dnu: ArrayLike,
    fwhm: ArrayLike,
    *,
    numax: ArrayLike | None = None,
    integration_time_seconds: float | None = None,
    dilution: ArrayLike = 1.0,
    total_mode_visibility: ArrayLike = DEFAULT_TOTAL_MODE_VISIBILITY,
) -> NDArray[np.float64]:
    """Convert radial-mode RMS amplitude to Gaussian integrated power.

    The conversion treats ``total_mode_visibility`` as the effective sum of
    relative mode powers per radial order.  The number of contributing orders
    is the area of a unit-height Gaussian divided by ``dnu``.  Therefore

    ``P_env = V_tot A_rms^2 (FWHM / dnu) sqrt(pi / (4 ln 2))``.

    Dilution and the cadence response act on amplitude and are consequently
    squared.  Following the original detector, cadence attenuation is
    evaluated at ``numax``; this approximation is explicit here so it can be
    replaced by a frequency-dependent response near the Nyquist frequency.
    """

    amplitude = _positive_array(
        radial_mode_rms_amplitude, "radial_mode_rms_amplitude"
    )
    dnu_array = _positive_array(dnu, "dnu")
    fwhm_array = _positive_array(fwhm, "fwhm")
    visibility = _positive_array(total_mode_visibility, "total_mode_visibility")
    dilution_array = _positive_array(dilution, "dilution")
    if np.any(dilution_array > 1):
        raise ValueError("dilution must not exceed one")

    try:
        amplitude, dnu_array, fwhm_array, visibility, dilution_array = (
            np.broadcast_arrays(
                amplitude,
                dnu_array,
                fwhm_array,
                visibility,
                dilution_array,
            )
        )
    except ValueError as error:
        raise ValueError("envelope parameters must be broadcast-compatible") from error

    response = 1.0
    if integration_time_seconds is not None:
        if numax is None:
            raise ValueError("numax is required when cadence attenuation is enabled")
        response = cadence_amplitude_response(numax, integration_time_seconds)
        try:
            response = np.broadcast_to(response, amplitude.shape)
        except ValueError as error:
            raise ValueError("numax must be broadcast-compatible") from error

    gaussian_area_in_fwhm = np.sqrt(np.pi) / (2.0 * np.sqrt(np.log(2.0)))
    return (
        visibility
        * amplitude**2
        * (fwhm_array / dnu_array)
        * gaussian_area_in_fwhm
        * (dilution_array * response) ** 2
    )


@dataclass(frozen=True, slots=True)
class ObservationModel:
    """Instrumental terms needed to observe an oscillation envelope.

    ``integration_time_seconds=None`` disables cadence attenuation, which is
    useful for an intrinsic or already-corrected spectrum.  ``dilution`` is
    the fraction of aperture flux supplied by the target star.
    """

    integration_time_seconds: float | None = None
    dilution: float = 1.0
    total_mode_visibility: float = DEFAULT_TOTAL_MODE_VISIBILITY

    def __post_init__(self) -> None:
        if self.integration_time_seconds is not None:
            integration_time = float(self.integration_time_seconds)
            if not np.isfinite(integration_time) or integration_time <= 0:
                raise ValueError(
                    "integration_time_seconds must be finite and positive"
                )
            object.__setattr__(self, "integration_time_seconds", integration_time)
        dilution = float(self.dilution)
        if not np.isfinite(dilution) or not 0 < dilution <= 1:
            raise ValueError("dilution must be finite and in (0, 1]")
        object.__setattr__(self, "dilution", dilution)
        visibility = float(self.total_mode_visibility)
        if not np.isfinite(visibility) or visibility <= 0:
            raise ValueError("total_mode_visibility must be finite and positive")
        object.__setattr__(self, "total_mode_visibility", visibility)

    def envelope_power(
        self,
        radial_mode_rms_amplitude: ArrayLike,
        dnu: ArrayLike,
        fwhm: ArrayLike,
        *,
        numax: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Return observed Gaussian-envelope integrated power in ppm squared."""

        return envelope_integrated_power(
            radial_mode_rms_amplitude,
            dnu,
            fwhm,
            numax=numax,
            integration_time_seconds=self.integration_time_seconds,
            dilution=self.dilution,
            total_mode_visibility=self.total_mode_visibility,
        )
