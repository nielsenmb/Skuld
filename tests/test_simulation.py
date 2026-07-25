import numpy as np
import pytest

from asterodetect import (
    ObservingWindow,
    SpectralModel,
    continuous_observing_window,
    observing_window_diagnostics,
    simulate_periodogram,
    simulate_windowed_periodogram,
    tess_observing_window,
    tess_like_observing_window,
)


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


def test_continuous_time_domain_simulation_recovers_exponential_moments():
    window = continuous_observing_window(20.0, 600.0)
    expected = np.full(np.fft.rfftfreq(window.observed.size)[1:].shape, 3.0)
    powers = simulate_windowed_periodogram(expected, window, rng=21)
    assert window.duty_cycle == 1.0
    assert np.isclose(powers.mean(), 3.0, rtol=0.04)
    assert np.isclose(powers.var(), 9.0, rtol=0.08)


def test_tess_like_window_is_reproducible_and_contains_structured_gaps():
    first = tess_like_observing_window(27.4, 120.0, rng=5)
    second = tess_like_observing_window(27.4, 120.0, rng=5)
    np.testing.assert_array_equal(first.observed, second.observed)
    assert first.label == "tess-like"
    assert 0.9 < first.duty_cycle < 1.0
    downlink_sample = int(13.7 * 86400 / 120)
    assert not first.observed[downlink_sample]


def test_tess_component_profiles_separate_gap_sources():
    momentum = tess_observing_window(
        27.4, 120.0, profile="momentum-dumps", rng=5
    )
    downlinks = tess_observing_window(
        27.4, 120.0, profile="downlinks", rng=5
    )
    combined = tess_observing_window(
        27.4, 120.0, profile="tess-like", rng=5
    )
    assert momentum.label == "momentum-dumps"
    assert downlinks.label == "downlinks"
    assert combined.duty_cycle < momentum.duty_cycle
    assert combined.duty_cycle < downlinks.duty_cycle


def test_random_loss_control_exactly_matches_combined_duty_cycle():
    random_loss = tess_observing_window(
        90.0, 120.0, profile="random-loss-matched", rng=12
    )
    combined = tess_observing_window(
        90.0, 120.0, profile="tess-like", rng=12
    )
    assert random_loss.duty_cycle == combined.duty_cycle
    assert not np.array_equal(random_loss.observed, combined.observed)


def test_window_diagnostics_distinguish_random_and_structured_gaps():
    random_loss = tess_observing_window(
        90.0, 120.0, profile="random-loss-matched", rng=12
    )
    combined = tess_observing_window(
        90.0, 120.0, profile="tess-like", rng=12
    )
    random_diagnostic = observing_window_diagnostics(random_loss)
    combined_diagnostic = observing_window_diagnostics(combined)
    assert random_diagnostic.duty_cycle == combined_diagnostic.duty_cycle
    assert random_diagnostic.gap_count > combined_diagnostic.gap_count
    assert random_diagnostic.maximum_gap_hours < combined_diagnostic.maximum_gap_hours
    assert (
        random_diagnostic.peak_sidelobe_power
        < combined_diagnostic.peak_sidelobe_power
    )


def test_tess_window_profile_rejects_unknown_name():
    with pytest.raises(ValueError, match="profile must be"):
        tess_observing_window(27.4, 120.0, profile="mystery")


def test_gapped_window_preserves_white_noise_level_but_correlates_bins():
    continuous = continuous_observing_window(90.0, 600.0)
    gapped = tess_like_observing_window(
        90.0,
        600.0,
        random_loss_fraction=0,
    )
    expected = np.full(np.fft.rfftfreq(continuous.observed.size)[1:].shape, 2.0)
    complete_power = simulate_windowed_periodogram(expected, continuous, rng=9)
    gapped_power = simulate_windowed_periodogram(expected, gapped, rng=9)
    assert np.isclose(gapped_power.mean(), complete_power.mean(), rtol=0.05)
    assert not np.array_equal(gapped_power, complete_power)


def test_no_gap_window_preserves_the_paired_fourier_realization():
    continuous = continuous_observing_window(20.0, 600.0)
    no_gaps = tess_like_observing_window(
        20.0,
        600.0,
        downlink_duration_hours=0,
        momentum_dump_duration_minutes=0,
        random_loss_fraction=0,
    )
    expected = np.linspace(1.0, 3.0, np.fft.rfftfreq(2880)[1:].size)
    complete_power = simulate_windowed_periodogram(expected, continuous, rng=17)
    paired_power = simulate_windowed_periodogram(expected, no_gaps, rng=17)
    np.testing.assert_allclose(paired_power, complete_power, rtol=1e-12)


def test_observing_window_rejects_non_boolean_masks():
    with pytest.raises(TypeError, match="boolean"):
        ObservingWindow(np.ones(10), 120.0)
