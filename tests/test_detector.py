import numpy as np
import pytest

from asterodetect import AsteroScaleSamples, Detector, NuisancePrior, PowerSpectrum
from asterodetect.asteroscale import ASTERO_SCALE_PARAMETERS


def _samples(n=32):
    values = {name: np.ones(n) for name in ASTERO_SCALE_PARAMETERS}
    values.update(
        numax=np.full(n, 100.0), dnu=np.full(n, 10.0),
        FWHM_env=np.full(n, 50.0), A_env=np.full(n, 3.0),
        A_gran=np.full(n, 8.0), b_gran_low=np.full(n, 30.0),
        b_gran_high=np.full(n, 90.0),
    )
    return AsteroScaleSamples(values)


def _raw_spectrum():
    frequency = np.arange(0.5, 200.0, 0.5)
    return PowerSpectrum(frequency, np.full_like(frequency, 0.1))


def test_detector_runs_complete_fixed_binning_path_reproducibly():
    detector = Detector(draws=24)
    first = detector.run(_raw_spectrum(), _samples(), rng=42)
    second = detector.run(_raw_spectrum(), _samples(), rng=42)
    assert first.bin_width == 10.0
    assert first.binned_spectrum.power.size < _raw_spectrum().power.size
    assert len(first.samples) == 24
    assert first.classification in ("noise", "granulation", "oscillation")
    assert first.probabilities == second.probabilities


def test_nuisance_draws_are_valid_and_scatter_can_be_disabled():
    prior = NuisancePrior(
        white_noise_log_scatter=0, envelope_log_scatter=0,
        granulation_log_scatter=0, overdispersion_log_scatter=0,
    )
    draws = prior.sample(_raw_spectrum(), 16, rng=1)
    np.testing.assert_allclose(draws["white_noise"], 0.1)
    np.testing.assert_allclose(draws["envelope_scale"], 1)
    np.testing.assert_allclose(draws["granulation_scale"], 1)
    np.testing.assert_allclose(draws["overdispersion"], 1)
    split = draws["granulation_variance_fraction_low"]
    assert np.all((split > 0) & (split < 1))


def test_suppressed_amplitude_mixture_has_requested_weight_and_means():
    prior = NuisancePrior(
        envelope_log_scatter=0,
        envelope_suppressed_fraction=0.25,
        envelope_suppression_factor=0.3,
        envelope_suppressed_log_scatter=0,
    )
    draws = prior.sample(_raw_spectrum(), 40_000, rng=14)
    envelope = draws["envelope_scale"]
    suppressed = np.isclose(envelope, 0.3)
    assert np.mean(suppressed) == pytest.approx(0.25, abs=0.01)
    np.testing.assert_allclose(envelope[suppressed], 0.3)
    np.testing.assert_allclose(envelope[~suppressed], 1.0)


def test_zero_suppressed_fraction_recovers_original_lognormal_transform():
    prior = NuisancePrior(
        envelope_log_scatter=0.4,
        envelope_suppressed_fraction=0,
    )
    latent = np.zeros((3, len(prior.latent_names)))
    latent[:, 1] = [-1.0, 0.0, 1.0]
    transformed = prior.transform_latent(latent, white_noise_centre=1.0)
    expected = np.exp(-0.5 * 0.4**2 + 0.4 * latent[:, 1])
    np.testing.assert_allclose(transformed["envelope_scale"], expected)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("envelope_suppressed_fraction", -0.1, "fraction"),
        ("envelope_suppressed_fraction", 1.0, "fraction"),
        ("envelope_suppression_factor", 0.0, "factor"),
        ("envelope_suppression_factor", 1.1, "factor"),
        ("envelope_suppressed_log_scatter", -0.1, "scatter"),
    ],
)
def test_suppressed_amplitude_prior_validates_hyperparameters(
    keyword, value, message
):
    with pytest.raises(ValueError, match=message):
        NuisancePrior(**{keyword: value})


def test_adaptive_detector_is_reproducible_and_improves_noise_ess():
    prior = NuisancePrior(white_noise_log_scatter=1.2)
    plain = Detector(draws=256, nuisance_prior=prior).run(
        _raw_spectrum(), _samples(), rng=42, white_noise_centre=0.1
    )
    detector = Detector(
        draws=256,
        pilot_draws=96,
        estimator="adaptive",
        nuisance_prior=prior,
    )
    first = detector.run(
        _raw_spectrum(), _samples(), rng=42, white_noise_centre=0.1
    )
    second = detector.run(
        _raw_spectrum(), _samples(), rng=42, white_noise_centre=0.1
    )
    assert first.estimator == "adaptive"
    assert first.probabilities == second.probabilities
    assert (
        first.evaluation.diagnostics["noise"].effective_sample_size
        > plain.evaluation.diagnostics["noise"].effective_sample_size
    )
    assert len(first.samples) == 256


def test_adaptive_detector_validates_stellar_average_size():
    with np.testing.assert_raises(ValueError):
        Detector(
            estimator="adaptive",
            stellar_draws_per_nuisance=0,
        ).run(_raw_spectrum(), _samples(), rng=1)
