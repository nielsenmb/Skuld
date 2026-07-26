import numpy as np
import pytest

from asterodetect import (
    AsteroScaleSamples,
    GaussianEnvelope,
    HarveyComponent,
    PowerSpectrum,
    PriorPredictiveMarginalizer,
    SpectralModel,
)
from asterodetect.asteroscale import ASTERO_SCALE_PARAMETERS


def _samples(n=128):
    values = {name: np.full(n, 1.0) for name in ASTERO_SCALE_PARAMETERS}
    values.update(
        numax=np.full(n, 100.0),
        dnu=np.full(n, 10.0),
        FWHM_env=np.full(n, 50.0),
        A_env=np.full(n, 3.0),
        A_gran=np.full(n, 8.0),
        b_gran_low=np.full(n, 30.0),
        b_gran_high=np.full(n, 90.0),
    )
    return AsteroScaleSamples(values)


def _spectrum(model, seed):
    lower = np.arange(10.0, 190.0, 5.0)
    upper = lower + 5.0
    frequency = 0.5 * (lower + upper)
    shape = np.full(frequency.size, 80.0)
    mean = model.mean_binned_spectrum(lower, upper)
    power = np.random.default_rng(seed).gamma(shape, mean / shape)
    return PowerSpectrum(
        frequency, power, shape, bin_lower=lower, bin_upper=upper
    )


def _models():
    low = HarveyComponent.from_rms_amplitude(8 / np.sqrt(2), 30.0)
    high = HarveyComponent.from_rms_amplitude(8 / np.sqrt(2), 90.0)
    envelope_power = 3.04 * 3.0**2 * 5.0 * np.sqrt(np.pi) / (2 * np.sqrt(np.log(2)))
    envelope = GaussianEnvelope(
        envelope_power, 100.0, 50 / (2 * np.sqrt(2 * np.log(2)))
    )
    return (
        SpectralModel(0.05),
        SpectralModel(0.05, [low, high]),
        SpectralModel(0.05, [low, high], envelope),
    )


def test_prior_predictive_marginalizer_recovers_three_generating_models():
    samples = _samples()
    marginalizer = PriorPredictiveMarginalizer()
    for expected_label, model, seed in zip(
        marginalizer.labels, _models(), (1, 2, 3), strict=True
    ):
        result = marginalizer.evaluate(_spectrum(model, seed), samples, 0.05)
        assert max(result.responsibilities, key=result.responsibilities.get) == expected_label
        assert result.diagnostics[expected_label].effective_sample_size == len(samples)
        assert result.diagnostics[expected_label].log_evidence_standard_error == 0.0


def test_white_noise_draws_are_marginalized_and_report_low_ess_when_needed():
    samples = _samples(64)
    white = np.geomspace(0.01, 1.0, len(samples))
    result = PriorPredictiveMarginalizer().evaluate(_spectrum(_models()[0], 4), samples, white)
    diagnostic = result.diagnostics["noise"]
    assert 1 <= diagnostic.effective_sample_size < len(samples)
    assert diagnostic.log_evidence_standard_error > 0
    assert np.isclose(sum(result.responsibilities.values()), 1.0)


def test_log_likelihoods_can_evaluate_one_model():
    likelihoods = PriorPredictiveMarginalizer().log_likelihoods(
        _spectrum(_models()[0], 4),
        _samples(8),
        0.05,
        model_labels=("noise",),
    )
    assert tuple(likelihoods) == ("noise",)
    with pytest.raises(ValueError):
        PriorPredictiveMarginalizer().log_likelihoods(
            _spectrum(_models()[0], 4),
            _samples(8),
            0.05,
            model_labels=("unknown",),
        )


def test_marginal_evaluation_can_reweight_a_model_subset():
    evaluation = PriorPredictiveMarginalizer().from_evidences(
        {"noise": 0.0, "granulation": 1.0, "oscillation": 2.0},
        {},
    )
    reweighted = evaluation.reweight({"noise": 0.5, "oscillation": 0.5})
    expected = 1.0 / (1.0 + np.exp(-2.0))
    assert reweighted.responsibilities["oscillation"] == pytest.approx(expected)
    assert reweighted.responsibilities["granulation"] == 0.0
    assert reweighted.log_evidences is evaluation.log_evidences


@pytest.mark.parametrize(
    "probabilities",
    [
        {"oscillation": 1.0},
        {"noise": 0.6, "oscillation": 0.6},
        {"noise": -0.1, "oscillation": 1.1},
        {"noise": 0.5, "unknown": 0.5},
    ],
)
def test_marginal_reweight_validates_model_probabilities(probabilities):
    evaluation = PriorPredictiveMarginalizer().from_evidences(
        {"noise": 0.0, "granulation": 1.0, "oscillation": 2.0},
        {},
    )
    with pytest.raises(ValueError):
        evaluation.reweight(probabilities)
