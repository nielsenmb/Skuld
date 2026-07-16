import numpy as np

from asterodetect import GaussianEnvelope, HarveyComponent, SpectralModel


def test_harvey_component_reaches_half_power_at_characteristic_frequency():
    component = HarveyComponent(power=10, characteristic_frequency=100, exponent=4)
    np.testing.assert_allclose(component([0, 100]), [10, 5])


def test_gaussian_envelope_parameterization():
    envelope = GaussianEnvelope(integrated_power=100, numax=1000, sigma=100)
    assert np.isclose(envelope([1000])[0], envelope.height)
    assert np.isclose(envelope.fwhm, 235.48200450309495)


def test_spectral_model_adds_components():
    frequency = np.array([0.0, 100.0, 1000.0])
    harvey = HarveyComponent(10, 100)
    envelope = GaussianEnvelope(100, 1000, 100)
    model = SpectralModel(white_noise=2, harvey_components=[harvey], envelope=envelope)
    expected = 2 + harvey(frequency) + envelope(frequency)
    np.testing.assert_allclose(model.mean_spectrum(frequency), expected)

