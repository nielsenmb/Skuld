import numpy as np
import pytest

from asterodetect import (
    PowerSpectrum,
    SpectralWindowOperator,
    continuous_observing_window,
    simulate_windowed_periodogram,
    tess_observing_window,
)


def test_continuous_operator_is_identity():
    window = continuous_observing_window(2.0, 600.0)
    operator = SpectralWindowOperator.from_observing_window(window)
    expected = 1.0 + np.sin(operator.frequency / 100.0) ** 2
    np.testing.assert_allclose(operator.convolve(expected), expected)


def test_window_operator_preserves_white_power_away_from_zero():
    window = tess_observing_window(
        27.4,
        600.0,
        profile="momentum-dumps",
    )
    operator = SpectralWindowOperator.from_observing_window(window)
    convolved = operator.convolve(np.ones_like(operator.frequency))
    np.testing.assert_allclose(convolved[10:], 1.0, rtol=2e-3, atol=2e-3)


def test_window_operator_matches_monte_carlo_mean():
    window = tess_observing_window(
        4.0,
        1800.0,
        profile="momentum-dumps",
        momentum_dump_interval_days=1.0,
        momentum_dump_duration_minutes=120.0,
    )
    operator = SpectralWindowOperator.from_observing_window(window)
    expected = np.ones_like(operator.frequency)
    expected += 20.0 * np.exp(
        -0.5 * ((operator.frequency - 80.0) / 8.0) ** 2
    )
    target = operator.convolve(expected)
    simulations = np.asarray(
        [
            simulate_windowed_periodogram(expected, window, rng=seed)
            for seed in range(2000)
        ]
    )
    empirical = np.mean(simulations, axis=0)
    selected = operator.frequency > 10.0
    np.testing.assert_allclose(
        empirical[selected],
        target[selected],
        rtol=0.08,
        atol=0.08,
    )


def test_window_operator_convolves_and_averages_bins():
    window = continuous_observing_window(2.0, 600.0)
    operator = SpectralWindowOperator.from_observing_window(window)
    frequency = operator.frequency
    lower = np.asarray([frequency[4], frequency[9]])
    upper = np.asarray([frequency[9], frequency[14]])
    centres = 0.5 * (lower + upper)
    spectrum = PowerSpectrum(
        centres,
        np.ones(2),
        bins_averaged=5,
        bin_lower=lower,
        bin_upper=upper,
    )
    expected = np.arange(frequency.size, dtype=float)
    actual = operator.convolve_and_bin(expected, spectrum)
    np.testing.assert_allclose(actual, [6.0, 11.0])


@pytest.mark.parametrize(
    "observed,cadence",
    [
        (np.ones(3, dtype=bool), 120.0),
        (np.asarray([True, True, True, False]), 120.0),
        (np.ones(4, dtype=bool), 0.0),
    ],
)
def test_window_operator_validates_inputs(observed, cadence):
    with pytest.raises((TypeError, ValueError)):
        SpectralWindowOperator(observed, cadence)
