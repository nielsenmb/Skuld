import numpy as np

from asterodetect import SpectralModel, simulate_periodogram


def test_simulated_periodogram_has_expected_moments():
    frequency = np.arange(200_000, dtype=float)
    model = SpectralModel(white_noise=3.0)
    powers = simulate_periodogram(frequency, model, bins_averaged=4, rng=123)
    assert np.isclose(powers.mean(), 3.0, rtol=0.01)
    assert np.isclose(powers.var(), 3.0**2 / 4, rtol=0.02)


def test_simulation_is_reproducible_from_seed():
    frequency = np.arange(10, dtype=float)
    model = SpectralModel(1)
    first = simulate_periodogram(frequency, model, rng=42)
    second = simulate_periodogram(frequency, model, rng=42)
    np.testing.assert_array_equal(first, second)
