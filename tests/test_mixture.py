import numpy as np

from asterodetect import PowerSpectrum, SpectralModel, WholeSpectrumMixture


def test_whole_spectrum_mixture_matches_manual_calculation():
    spectrum = PowerSpectrum([1, 2], [1, 1])
    mixture = WholeSpectrumMixture(
        models={"low": SpectralModel(1), "high": SpectralModel(2)},
        probabilities={"low": 0.25, "high": 0.75},
    )
    result = mixture.evaluate(spectrum)

    log_low = -2.0
    log_high = -2 * np.log(2) - 1.0
    target = np.log(0.25 * np.exp(log_low) + 0.75 * np.exp(log_high))
    assert np.isclose(result.log_likelihood, target)
    assert np.isclose(sum(result.responsibilities.values()), 1)


def test_responsibility_uses_whole_spectrum_likelihood():
    spectrum = PowerSpectrum(np.arange(1, 101), np.ones(100))
    mixture = WholeSpectrumMixture(
        models={"correct": SpectralModel(1), "wrong": SpectralModel(10)},
        probabilities={"correct": 0.5, "wrong": 0.5},
    )
    result = mixture.evaluate(spectrum)
    assert result.responsibilities["correct"] > 1 - 1e-12

