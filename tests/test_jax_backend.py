import numpy as np
import jax
import jax.numpy as jnp

from asterodetect import (
    GaussianEnvelope,
    HarveyComponent,
    SpectralModel,
    envelope_integrated_power,
    gamma_log_likelihood,
)
from asterodetect.jax_backend import (
    envelope_integrated_power as jax_envelope_integrated_power,
    gamma_log_likelihood as jax_gamma_log_likelihood,
    mean_binned_spectrum,
    mean_spectrum,
    whole_spectrum_mixture,
)


def test_jax_envelope_power_is_jittable_and_differentiable():
    compiled = jax.jit(jax_envelope_integrated_power)
    actual = compiled(2.1, 135.1, 598.0, 3090.0, 120.0, 0.9, 3.04)
    assert np.isfinite(actual)
    gradient = jax.grad(jax_envelope_integrated_power, argnums=0)(
        2.1, 135.1, 598.0, 3090.0, 120.0, 0.9, 3.04
    )
    assert gradient > 0


def test_jax_envelope_power_matches_validated_numpy_conversion():
    arguments = (2.1, 135.1, 598.0, 3090.0, 120.0, 0.9, 3.04)
    actual = jax_envelope_integrated_power(*arguments)
    target = envelope_integrated_power(
        arguments[0],
        arguments[1],
        arguments[2],
        numax=arguments[3],
        integration_time_seconds=arguments[4],
        dilution=arguments[5],
        total_mode_visibility=arguments[6],
    )
    assert np.isclose(actual, target, rtol=1e-6)


def test_jitted_likelihood_matches_validated_numpy_likelihood():
    power = np.array([0.5, 1.5, 3.0])
    expected = np.array([1.0, 2.0, 4.0])
    shape = np.array([1.0, 2.0, 4.0])
    compiled = jax.jit(jax_gamma_log_likelihood)
    actual = compiled(power, expected, shape)
    target = gamma_log_likelihood(power, expected, shape)
    assert np.isclose(actual, target, rtol=1e-6)


def test_mean_spectrum_is_jittable_and_differentiable():
    frequency = jnp.linspace(100.0, 4000.0, 1000)

    def total_power(log_amplitude):
        return jnp.sum(
            mean_spectrum(
                frequency,
                1.0,
                jnp.array([]),
                jnp.array([]),
                jnp.array([]),
                jnp.exp(log_amplitude),
                1800.0,
                250.0,
            )
        )

    value = jax.jit(total_power)(jnp.log(150.0))
    gradient = jax.grad(total_power)(jnp.log(150.0))
    assert np.isfinite(value)
    assert gradient > 0


def test_jax_mean_spectrum_matches_validated_object_model():
    frequency = np.linspace(100.0, 4000.0, 1000)
    components = (HarveyComponent(2.0, 700.0), HarveyComponent(1.0, 300.0))
    envelope = GaussianEnvelope(150.0, 1800.0, 250.0)
    model = SpectralModel(1.0, components, envelope)
    actual = mean_spectrum(
        frequency,
        model.white_noise,
        jnp.array([component.power for component in components]),
        jnp.array(
            [component.characteristic_frequency for component in components]
        ),
        jnp.array([component.exponent for component in components]),
        envelope.integrated_power,
        envelope.numax,
        envelope.sigma,
    )
    np.testing.assert_allclose(actual, model.mean_spectrum(frequency), rtol=1e-6)


def test_jax_binned_model_matches_validated_object_model():
    lower = np.arange(100.0, 4000.0, 100.0)
    upper = lower + 100.0
    components = (HarveyComponent(2.0, 700.0), HarveyComponent(1.0, 300.0))
    envelope = GaussianEnvelope(150.0, 1800.0, 250.0)
    model = SpectralModel(1.0, components, envelope)
    actual = jax.jit(mean_binned_spectrum)(
        lower,
        upper,
        model.white_noise,
        jnp.array([component.power for component in components]),
        jnp.array([component.characteristic_frequency for component in components]),
        jnp.array([component.exponent for component in components]),
        envelope.integrated_power,
        envelope.numax,
        envelope.sigma,
    )
    target = model.mean_binned_spectrum(lower, upper, quadrature_order=64)
    np.testing.assert_allclose(actual, target, rtol=1e-5)


def test_jax_mixture_uses_complete_spectrum_likelihoods():
    power = jnp.ones(100)
    expected = jnp.stack([jnp.ones(100), jnp.full(100, 10.0)])
    total, responsibilities = jax.jit(whole_spectrum_mixture)(
        power, expected, jnp.ones(100), jnp.array([0.5, 0.5])
    )
    assert np.isfinite(total)
    assert responsibilities[0] > 1 - 1e-6
