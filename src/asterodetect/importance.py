"""Defensive adaptive importance sampling for marginal likelihoods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ImportanceDiagnostic:
    """Diagnostics for one adaptive evidence estimate."""

    pilot_draws: int
    draws: int
    effective_sample_size: float
    log_evidence_standard_error: float
    defensive_fraction: float


@dataclass(frozen=True, slots=True)
class ImportanceResult:
    """One evidence estimate and its normalized log weights."""

    log_evidence: float
    diagnostic: ImportanceDiagnostic
    samples: Array
    log_weights: Array


class AdaptiveImportanceSampler:
    """Fit a Gaussian proposal to a likelihood-weighted prior pilot.

    The final proposal is a defensive mixture of the fitted Gaussian and the
    original prior.  The prior component guarantees support wherever the prior
    is non-zero and makes the estimator robust to a poor pilot fit.
    """

    def __init__(
        self,
        *,
        pilot_draws: int = 256,
        draws: int = 2048,
        defensive_fraction: float = 0.2,
        covariance_inflation: float = 1.5,
        covariance_regularization: float = 1e-6,
    ) -> None:
        for name, value in (("pilot_draws", pilot_draws), ("draws", draws)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if value < 2:
                raise ValueError(f"{name} must be at least two")
        if not 0 < defensive_fraction <= 1:
            raise ValueError("defensive_fraction must be in (0, 1]")
        if covariance_inflation <= 0 or covariance_regularization <= 0:
            raise ValueError("covariance controls must be positive")
        self.pilot_draws = int(pilot_draws)
        self.draws = int(draws)
        self.defensive_fraction = float(defensive_fraction)
        self.covariance_inflation = float(covariance_inflation)
        self.covariance_regularization = float(covariance_regularization)

    def run(
        self,
        prior_sample: Callable[[int, np.random.Generator], Array],
        prior_logpdf: Callable[[Array], Array],
        log_likelihood: Callable[[Array], Array],
        *,
        rng: np.random.Generator | int | None = None,
    ) -> ImportanceResult:
        """Estimate an evidence using a pilot-adapted defensive proposal."""

        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        pilot = _matrix(prior_sample(self.pilot_draws, generator))
        pilot_log_likelihood = _vector(log_likelihood(pilot), self.pilot_draws)
        finite = np.isfinite(pilot_log_likelihood)
        if not np.any(finite):
            raise ValueError("pilot produced no finite log likelihoods")
        fit_weights = np.zeros(self.pilot_draws)
        fit_weights[finite] = np.exp(
            pilot_log_likelihood[finite] - logsumexp(pilot_log_likelihood[finite])
        )
        mean = np.sum(pilot * fit_weights[:, None], axis=0)
        centred = pilot - mean
        covariance = (centred * fit_weights[:, None]).T @ centred
        covariance *= self.covariance_inflation
        scale = np.maximum(np.diag(covariance), 1.0)
        covariance += np.diag(self.covariance_regularization * scale)
        gaussian = multivariate_normal(mean=mean, cov=covariance, allow_singular=False)

        use_prior = generator.random(self.draws) < self.defensive_fraction
        samples = np.empty((self.draws, pilot.shape[1]))
        prior_count = int(np.sum(use_prior))
        if prior_count:
            samples[use_prior] = _matrix(prior_sample(prior_count, generator))
        if prior_count < self.draws:
            samples[~use_prior] = gaussian.rvs(
                size=self.draws - prior_count, random_state=generator
            ).reshape(-1, pilot.shape[1])

        log_prior = _vector(prior_logpdf(samples), self.draws)
        log_gaussian = np.asarray(gaussian.logpdf(samples), dtype=float).reshape(-1)
        if self.defensive_fraction == 1:
            log_proposal = log_prior
        else:
            log_proposal = np.logaddexp(
                np.log(self.defensive_fraction) + log_prior,
                np.log1p(-self.defensive_fraction) + log_gaussian,
            )
        log_weights = (
            _vector(log_likelihood(samples), self.draws)
            + log_prior
            - log_proposal
        )
        log_evidence = float(logsumexp(log_weights) - np.log(self.draws))
        normalized = np.exp(log_weights - np.max(log_weights))
        ess = float(normalized.sum() ** 2 / np.sum(normalized**2))
        standard_error = float(np.sqrt(max(0.0, 1.0 / ess - 1.0 / self.draws)))
        samples.setflags(write=False)
        log_weights.setflags(write=False)
        return ImportanceResult(
            log_evidence,
            ImportanceDiagnostic(
                self.pilot_draws,
                self.draws,
                ess,
                standard_error,
                self.defensive_fraction,
            ),
            samples,
            log_weights,
        )


def _matrix(values: Array) -> Array:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[0] < 1 or not np.all(np.isfinite(result)):
        raise ValueError("samples must be a finite two-dimensional array")
    return result


def _vector(values: Array, size: int) -> Array:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,):
        raise ValueError(f"function must return shape ({size},)")
    return result
