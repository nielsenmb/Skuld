import numpy as np
from scipy.stats import norm

from asterodetect.importance import AdaptiveImportanceSampler


def test_adaptive_importance_recovers_normal_normal_evidence():
    # Prior N(0, 3), observation y=4 with unit Normal noise.
    prior_scale = 3.0
    observed = 4.0

    def sample(size, rng):
        return rng.normal(0.0, prior_scale, (size, 1))

    def log_prior(x):
        return norm.logpdf(x[:, 0], 0.0, prior_scale)

    def log_likelihood(x):
        return norm.logpdf(observed, x[:, 0], 1.0)

    result = AdaptiveImportanceSampler(
        pilot_draws=256, draws=4096, defensive_fraction=0.1
    ).run(sample, log_prior, log_likelihood, rng=12)
    truth = norm.logpdf(observed, 0.0, np.sqrt(prior_scale**2 + 1.0))
    assert abs(result.log_evidence - truth) < 0.08
    assert result.diagnostic.effective_sample_size > 1000


def test_defensive_component_handles_unhelpful_pilot():
    def sample(size, rng):
        return rng.normal(size=(size, 1))

    sampler = AdaptiveImportanceSampler(
        pilot_draws=32, draws=256, defensive_fraction=1.0
    )
    result = sampler.run(
        sample,
        lambda x: norm.logpdf(x[:, 0]),
        lambda x: np.zeros(x.shape[0]),
        rng=1,
    )
    assert np.isclose(result.log_evidence, 0.0)
    assert np.isclose(result.diagnostic.effective_sample_size, 256)
