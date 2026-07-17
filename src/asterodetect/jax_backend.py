"""Pure JAX kernels for the performance-sensitive numerical core.

The public object-oriented API performs validation and remains NumPy based.
These lower-level functions contain no Python-side data validation, scalar
casts, or mutable state, making them suitable for :func:`jax.jit`,
:func:`jax.vmap`, and automatic differentiation.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import erf, gammaln, logsumexp


_GAUSS_LEGENDRE_NODES = jnp.asarray(
    [-0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
      0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565]
)
_GAUSS_LEGENDRE_WEIGHTS = jnp.asarray(
    [0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
     0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363]
)


def cadence_amplitude_response(frequency, integration_time_seconds):
    """JAX kernel for finite-integration-time amplitude attenuation."""

    return jnp.sinc(frequency * integration_time_seconds * 1.0e-6)


def envelope_integrated_power(
    radial_mode_rms_amplitude,
    dnu,
    fwhm,
    numax,
    integration_time_seconds,
    dilution=1.0,
    total_mode_visibility=3.04,
):
    """JAX kernel mapping radial-mode RMS amplitude to observed power."""

    response = cadence_amplitude_response(numax, integration_time_seconds)
    gaussian_area_in_fwhm = jnp.sqrt(jnp.pi) / (2.0 * jnp.sqrt(jnp.log(2.0)))
    return (
        total_mode_visibility
        * radial_mode_rms_amplitude**2
        * (fwhm / dnu)
        * gaussian_area_in_fwhm
        * (dilution * response) ** 2
    )


def harvey_profile(
    frequency,
    power,
    characteristic_frequency,
    exponent=4.0,
):
    """Evaluate a Harvey profile parameterized by zero-frequency power."""

    ratio = frequency / characteristic_frequency
    return power / (1.0 + ratio**exponent)


def harvey_from_rms(
    frequency,
    amplitude,
    characteristic_frequency,
    exponent=4.0,
):
    """Evaluate a one-sided Harvey profile normalized to RMS amplitude."""

    normalization = exponent * jnp.sin(jnp.pi / exponent) / jnp.pi
    power = normalization * amplitude**2 / characteristic_frequency
    return harvey_profile(
        frequency, power, characteristic_frequency, exponent
    )


def gaussian_envelope(frequency, integrated_power, numax, sigma):
    """Evaluate a Gaussian envelope parameterized by integrated power."""

    height = integrated_power / (jnp.sqrt(2.0 * jnp.pi) * sigma)
    offset = (frequency - numax) / sigma
    return height * jnp.exp(-0.5 * offset**2)


def mean_spectrum(
    frequency,
    white_noise,
    harvey_powers,
    harvey_frequencies,
    harvey_exponents,
    envelope_integrated_power=0.0,
    envelope_numax=1.0,
    envelope_sigma=1.0,
):
    """Evaluate a vectorized white-noise, Harvey, and envelope model.

    Harvey parameters are one-dimensional arrays of equal length.  Empty
    arrays represent a model with no Harvey components.  Setting the
    integrated envelope power to zero removes the envelope without changing
    the traced function signature.
    """

    frequency = jnp.asarray(frequency)
    powers = jnp.asarray(harvey_powers)
    characteristic_frequencies = jnp.asarray(harvey_frequencies)
    exponents = jnp.asarray(harvey_exponents)
    profiles = harvey_profile(
        frequency[..., None],
        powers,
        characteristic_frequencies,
        exponents,
    )
    background = white_noise + jnp.sum(profiles, axis=-1)
    envelope = gaussian_envelope(
        frequency,
        envelope_integrated_power,
        envelope_numax,
        envelope_sigma,
    )
    return background + envelope


def mean_binned_spectrum(
    lower,
    upper,
    white_noise,
    harvey_powers,
    harvey_frequencies,
    harvey_exponents,
    envelope_integrated_power=0.0,
    envelope_numax=1.0,
    envelope_sigma=1.0,
):
    """Average a spectral model over bins using fixed JAX operations."""

    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)
    widths = upper - lower
    samples = (
        0.5 * widths[:, None] * _GAUSS_LEGENDRE_NODES[None, :]
        + 0.5 * (upper + lower)[:, None]
    )
    profiles = harvey_profile(
        samples[..., None],
        jnp.asarray(harvey_powers),
        jnp.asarray(harvey_frequencies),
        jnp.asarray(harvey_exponents),
    )
    harvey_average = 0.5 * jnp.sum(
        profiles
        * _GAUSS_LEGENDRE_WEIGHTS[None, :, None],
        axis=(1, 2),
    )
    scale = jnp.sqrt(2.0) * envelope_sigma
    probability = 0.5 * (
        erf((upper - envelope_numax) / scale)
        - erf((lower - envelope_numax) / scale)
    )
    envelope_average = envelope_integrated_power * probability / widths
    return white_noise + harvey_average + envelope_average


def gamma_log_likelihood(power, expected_power, shape=1.0):
    """Independent-bin Gamma log-likelihood with a JAX-traceable body."""

    power = jnp.asarray(power)
    expected_power = jnp.asarray(expected_power)
    shape = jnp.asarray(shape)
    terms = (
        shape * jnp.log(shape)
        - gammaln(shape)
        + (shape - 1.0) * jnp.log(power)
        - shape * jnp.log(expected_power)
        - shape * power / expected_power
    )
    return jnp.sum(terms)


def whole_spectrum_mixture(power, expected_spectra, shape, probabilities):
    """Return mixture log-likelihood and whole-spectrum responsibilities.

    ``expected_spectra`` has shape ``(n_models, n_frequency)``.  This is a
    mixture of complete-spectrum likelihoods, not a per-frequency mixture.
    """

    expected_spectra = jnp.asarray(expected_spectra)
    component_logs = jnp.sum(
        shape * jnp.log(shape)
        - gammaln(shape)
        + (shape - 1.0) * jnp.log(power)
        - shape * jnp.log(expected_spectra)
        - shape * power / expected_spectra,
        axis=-1,
    )
    joint_logs = component_logs + jnp.log(probabilities)
    total = logsumexp(joint_logs)
    return total, jnp.exp(joint_logs - total)
