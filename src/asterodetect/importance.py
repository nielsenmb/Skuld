"""Defensive adaptive importance sampling for marginal likelihoods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import multivariate_t


Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ImportanceDiagnostic:
    """Diagnostics for one adaptive evidence estimate.

    Attributes
    ----------
    pilot_draws
        Number of prior draws used to fit the adaptive proposal.
    draws
        Number of draws used for the final evidence estimate.
    effective_sample_size
        Effective number of final importance samples.
    log_evidence_standard_error
        Delta-method standard error of the log evidence.
    defensive_fraction
        Fraction of final samples drawn directly from the prior.
    pilot_effective_sample_size
        Effective sample size of the tempered pilot weights.
    adaptation_temperature
        Likelihood power used to fit the proposal. A value below one indicates
        that tempering was needed to prevent the pilot fit collapsing.
    """

    pilot_draws: int
    draws: int
    effective_sample_size: float
    log_evidence_standard_error: float
    defensive_fraction: float
    pilot_effective_sample_size: float
    adaptation_temperature: float


@dataclass(frozen=True, slots=True)
class ImportanceResult:
    """One evidence estimate and its unnormalized log weights.

    Attributes
    ----------
    log_evidence
        Natural logarithm of the marginal likelihood.
    diagnostic
        Monte Carlo and proposal-fit diagnostics.
    samples
        Samples used in the final importance estimate.
    log_weights
        Unnormalized logarithmic importance weights.
    """

    log_evidence: float
    diagnostic: ImportanceDiagnostic
    samples: Array
    log_weights: Array


class AdaptiveImportanceSampler:
    """Fit a heavy-tailed proposal to a tempered posterior pilot.

    The pilot likelihood is automatically tempered until its effective sample
    size reaches ``pilot_ess_fraction`` of the pilot sample. This avoids fitting
    a nearly singular proposal to one lucky pilot point when the likelihood is
    much narrower than the prior. The final proposal is a defensive mixture of
    the fitted multivariate Student distribution and the
    original prior.  The prior component guarantees support wherever the prior
    is non-zero and makes the estimator robust to a poor pilot fit.

    Parameters
    ----------
    pilot_draws
        Number of prior draws used to fit the proposal.
    draws
        Number of samples in the final evidence estimate.
    defensive_fraction
        Fraction of final samples drawn from the original prior.
    covariance_inflation
        Multiplicative inflation applied to the fitted covariance.
    covariance_regularization
        Relative diagonal jitter added to the covariance.
    pilot_ess_fraction
        Minimum fraction of pilot draws represented by the tempered fitting
        weights.
    proposal_degrees_of_freedom
        Degrees of freedom of the multivariate Student proposal. Smaller
        values give heavier tails.
    """

    def __init__(
        self,
        *,
        pilot_draws: int = 256,
        draws: int = 2048,
        defensive_fraction: float = 0.2,
        covariance_inflation: float = 1.5,
        covariance_regularization: float = 1e-6,
        pilot_ess_fraction: float = 0.1,
        proposal_degrees_of_freedom: float = 5.0,
    ) -> None:
        """Initialize the adaptive evidence estimator."""
        for name, value in (("pilot_draws", pilot_draws), ("draws", draws)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if value < 2:
                raise ValueError(f"{name} must be at least two")
        if not 0 < defensive_fraction <= 1:
            raise ValueError("defensive_fraction must be in (0, 1]")
        if covariance_inflation <= 0 or covariance_regularization <= 0:
            raise ValueError("covariance controls must be positive")
        if not 0 < pilot_ess_fraction <= 1:
            raise ValueError("pilot_ess_fraction must be in (0, 1]")
        if proposal_degrees_of_freedom <= 2:
            raise ValueError("proposal_degrees_of_freedom must exceed two")
        self.pilot_draws = int(pilot_draws)
        self.draws = int(draws)
        self.defensive_fraction = float(defensive_fraction)
        self.covariance_inflation = float(covariance_inflation)
        self.covariance_regularization = float(covariance_regularization)
        self.pilot_ess_fraction = float(pilot_ess_fraction)
        self.proposal_degrees_of_freedom = float(proposal_degrees_of_freedom)

    def run(
        self,
        prior_sample: Callable[[int, np.random.Generator], Array],
        prior_logpdf: Callable[[Array], Array],
        log_likelihood: Callable[[Array], Array],
        *,
        rng: np.random.Generator | int | None = None,
    ) -> ImportanceResult:
        """Estimate an evidence using a pilot-adapted defensive proposal.

        Parameters
        ----------
        prior_sample
            Function drawing an ``(n, d)`` array from the normalized prior.
        prior_logpdf
            Function returning the prior log density for each input row.
        log_likelihood
            Function returning the log likelihood for each input row.
        rng
            Random generator or seed.

        Returns
        -------
        ImportanceResult
            Evidence estimate, final samples, weights, and diagnostics.

        Raises
        ------
        ValueError
            If the pilot contains no finite likelihood or a callback returns
            an invalid array.
        """

        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        pilot = _matrix(prior_sample(self.pilot_draws, generator))
        pilot_log_likelihood = _vector(log_likelihood(pilot), self.pilot_draws)
        finite = np.isfinite(pilot_log_likelihood)
        if not np.any(finite):
            raise ValueError("pilot produced no finite log likelihoods")
        target_pilot_ess = min(
            int(np.sum(finite)),
            max(2.0, self.pilot_ess_fraction * self.pilot_draws),
        )
        temperature, fit_weights = _tempered_weights(
            pilot_log_likelihood, target_pilot_ess
        )
        pilot_ess = _effective_sample_size(fit_weights)
        mean = np.sum(pilot * fit_weights[:, None], axis=0)
        centred = pilot - mean
        covariance = (centred * fit_weights[:, None]).T @ centred
        # Correct the biased weighted covariance when the pilot ESS is small.
        covariance /= max(1.0 - np.sum(fit_weights**2), np.finfo(float).eps)
        covariance *= self.covariance_inflation
        scale = np.maximum(np.diag(covariance), 1.0)
        covariance += np.diag(self.covariance_regularization * scale)
        proposal = multivariate_t(
            loc=mean,
            shape=covariance,
            df=self.proposal_degrees_of_freedom,
            allow_singular=False,
        )

        use_prior = generator.random(self.draws) < self.defensive_fraction
        samples = np.empty((self.draws, pilot.shape[1]))
        prior_count = int(np.sum(use_prior))
        if prior_count:
            samples[use_prior] = _matrix(prior_sample(prior_count, generator))
        if prior_count < self.draws:
            samples[~use_prior] = proposal.rvs(
                size=self.draws - prior_count, random_state=generator
            ).reshape(-1, pilot.shape[1])

        log_prior = _vector(prior_logpdf(samples), self.draws)
        log_adaptive = np.asarray(proposal.logpdf(samples), dtype=float).reshape(-1)
        if self.defensive_fraction == 1:
            log_proposal = log_prior
        else:
            log_proposal = np.logaddexp(
                np.log(self.defensive_fraction) + log_prior,
                np.log1p(-self.defensive_fraction) + log_adaptive,
            )
        log_weights = (
            _vector(log_likelihood(samples), self.draws)
            + log_prior
            - log_proposal
        )
        log_evidence = float(logsumexp(log_weights) - np.log(self.draws))
        normalized = np.exp(log_weights - np.max(log_weights))
        ess = _effective_sample_size(normalized)
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
                pilot_ess,
                temperature,
            ),
            samples,
            log_weights,
        )


def _matrix(values: Array) -> Array:
    """Validate and return a finite sample matrix.

    Parameters
    ----------
    values
        Candidate two-dimensional sample array.

    Returns
    -------
    numpy.ndarray
        Validated floating-point sample matrix.
    """
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[0] < 1 or not np.all(np.isfinite(result)):
        raise ValueError("samples must be a finite two-dimensional array")
    return result


def _vector(values: Array, size: int) -> Array:
    """Validate and return a one-dimensional callback result.

    Parameters
    ----------
    values
        Candidate callback result.
    size
        Required vector length.

    Returns
    -------
    numpy.ndarray
        Validated floating-point vector.
    """
    result = np.asarray(values, dtype=float)
    if result.shape != (size,):
        raise ValueError(f"function must return shape ({size},)")
    return result


def _effective_sample_size(weights: Array) -> float:
    """Return the effective sample size of non-negative weights.

    Parameters
    ----------
    weights
        Normalized or unnormalized non-negative weights.

    Returns
    -------
    float
        Effective sample size.
    """

    values = np.asarray(weights, dtype=float)
    return float(values.sum() ** 2 / np.sum(values**2))


def _tempered_weights(
    log_likelihood: Array,
    target_effective_sample_size: float,
) -> tuple[float, Array]:
    """Temper likelihood weights to retain a target pilot ESS.

    Parameters
    ----------
    log_likelihood
        Pilot log-likelihood values.
    target_effective_sample_size
        Minimum desired effective sample size for proposal fitting.

    Returns
    -------
    temperature
        Likelihood power between zero and one.
    weights
        Normalized tempered likelihood weights.
    """

    values = np.asarray(log_likelihood, dtype=float)
    finite = np.isfinite(values)

    def weights_at(temperature: float) -> Array:
        """Evaluate normalized pilot weights at one temperature.

        Parameters
        ----------
        temperature
            Likelihood power between zero and one.

        Returns
        -------
        numpy.ndarray
            Normalized weights, with zero weight for non-finite likelihoods.
        """

        weights = np.zeros(values.size)
        scaled = temperature * values[finite]
        weights[finite] = np.exp(scaled - logsumexp(scaled))
        return weights

    full_weights = weights_at(1.0)
    if _effective_sample_size(full_weights) >= target_effective_sample_size:
        return 1.0, full_weights

    lower = 0.0
    upper = 1.0
    for _ in range(50):
        midpoint = 0.5 * (lower + upper)
        if _effective_sample_size(weights_at(midpoint)) >= target_effective_sample_size:
            lower = midpoint
        else:
            upper = midpoint
    return lower, weights_at(lower)
