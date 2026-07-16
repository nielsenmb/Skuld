import numpy as np
from scipy.stats import gamma

from asterodetect import gamma_log_likelihood


def test_exponential_log_likelihood():
    observed = np.array([0.5, 1.5, 3.0])
    expected = np.array([1.0, 2.0, 4.0])
    actual = gamma_log_likelihood(observed, expected, shape=1)
    target = np.sum(-np.log(expected) - observed / expected)
    assert np.isclose(actual, target)


def test_gamma_log_likelihood_matches_scipy():
    observed = np.array([0.5, 1.5, 3.0])
    expected = np.array([1.0, 2.0, 4.0])
    shape = np.array([2.0, 4.0, 8.0])
    actual = gamma_log_likelihood(observed, expected, shape)
    target = np.sum(gamma.logpdf(observed, a=shape, scale=expected / shape))
    assert np.isclose(actual, target)

