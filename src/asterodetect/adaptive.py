"""Adaptive nuisance integration while retaining empirical AsteroScale rows."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import norm

from .asteroscale import AsteroScaleSamples
from .data import PowerSpectrum
from .importance import AdaptiveImportanceSampler, ImportanceDiagnostic
from .marginal import (
    MarginalEvaluation,
    MonteCarloDiagnostic,
    PriorPredictiveMarginalizer,
)
from .nuisance import NuisancePrior
from .observation import ObservationModel
from .window import SpectralWindowOperator


@dataclass(frozen=True, slots=True)
class AdaptiveMarginalEvaluation:
    """Adaptive evidence result plus model-specific diagnostics.

    Attributes
    ----------
    evaluation
        Marginal evidences, model probabilities, and Monte Carlo diagnostics.
    diagnostics
        Detailed importance-sampling diagnostics keyed by model label.
    oscillation_samples
        Complete AsteroScale rows used in the final oscillation evaluation.
    oscillation_nuisance_draws
        Transformed nuisance draws used in the final oscillation evaluation.
    """

    evaluation: MarginalEvaluation
    diagnostics: Mapping[str, ImportanceDiagnostic]
    oscillation_samples: AsteroScaleSamples
    oscillation_nuisance_draws: Mapping[str, NDArray[np.float64]]


class AdaptiveNuisanceMarginalizer:
    """Adapt nuisance proposals without fitting a density to AsteroScale rows.

    Complete AsteroScale rows are uniformly resampled for every likelihood
    batch. Only continuous nuisance coordinates are adapted, so empirical
    stellar correlations are preserved exactly.

    Parameters
    ----------
    draws
        Number of final importance samples per spectral model.
    pilot_draws
        Number of prior draws used to fit each proposal.
    defensive_fraction
        Fraction of final samples drawn directly from the nuisance prior.
    pilot_ess_fraction
        Minimum fraction of pilot draws retained by likelihood tempering.
    proposal_degrees_of_freedom
        Degrees of freedom of the heavy-tailed Student proposal.
    stellar_draws_per_nuisance
        Number of intact AsteroScale rows averaged for each nuisance point.
        Values above one reduce pseudo-marginal likelihood noise.
    model_probabilities
        Prior probabilities for the three spectral models.
    """

    _coordinates = {
        "noise": (0, 4),
        "granulation": (0, 2, 3, 4),
        "oscillation": (0, 1, 2, 3, 4),
    }

    def __init__(
        self,
        *,
        draws: int = 2048,
        pilot_draws: int = 256,
        defensive_fraction: float = 0.2,
        pilot_ess_fraction: float = 0.1,
        proposal_degrees_of_freedom: float = 5.0,
        stellar_draws_per_nuisance: int = 8,
        model_probabilities: Mapping[str, float] | None = None,
    ) -> None:
        """Initialize model-specific adaptive nuisance integrators."""

        if (
            isinstance(stellar_draws_per_nuisance, bool)
            or not isinstance(stellar_draws_per_nuisance, (int, np.integer))
            or stellar_draws_per_nuisance < 1
        ):
            raise ValueError("stellar_draws_per_nuisance must be a positive integer")
        self.stellar_draws_per_nuisance = int(stellar_draws_per_nuisance)
        self.sampler = AdaptiveImportanceSampler(
            draws=draws,
            pilot_draws=pilot_draws,
            defensive_fraction=defensive_fraction,
            pilot_ess_fraction=pilot_ess_fraction,
            proposal_degrees_of_freedom=proposal_degrees_of_freedom,
        )
        self.marginalizer = PriorPredictiveMarginalizer(
            model_probabilities=model_probabilities
        )

    def evaluate(
        self,
        spectrum: PowerSpectrum,
        stellar_samples: AsteroScaleSamples,
        nuisance_prior: NuisancePrior,
        *,
        white_noise_centre: float,
        observation: ObservationModel | None = None,
        spectral_window: SpectralWindowOperator | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> AdaptiveMarginalEvaluation:
        """Estimate all three evidences using separate defensive proposals.

        Parameters
        ----------
        spectrum
            Fixed-bin power spectrum to evaluate.
        stellar_samples
            Empirical joint AsteroScale sample rows.
        nuisance_prior
            Target-specific nuisance prior and latent transform.
        white_noise_centre
            Positive centre of the white-noise prior.
        observation
            Cadence, dilution, visibility, and bolometric response model.
        spectral_window
            Optional target-specific spectral-window forward operator.
        rng
            Random generator or seed.

        Returns
        -------
        AdaptiveMarginalEvaluation
            Evidence calculation and proposal diagnostics.
        """

        generator = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )
        observation = observation or ObservationModel()
        evidences: dict[str, float] = {}
        importance_diagnostics: dict[str, ImportanceDiagnostic] = {}
        monte_carlo: dict[str, MonteCarloDiagnostic] = {}
        saved_samples: AsteroScaleSamples | None = None
        saved_nuisance: Mapping[str, NDArray[np.float64]] | None = None

        for label in self.marginalizer.labels:
            coordinates = self._coordinates[label]
            row_rng = np.random.default_rng(generator.integers(0, 2**63 - 1))
            last: dict[str, object] = {}

            def prior_sample(
                size: int, local_rng: np.random.Generator
            ) -> NDArray[np.float64]:
                """Draw active nuisance coordinates from their latent prior."""

                return local_rng.normal(size=(size, len(coordinates)))

            def prior_logpdf(values: NDArray[np.float64]) -> NDArray[np.float64]:
                """Evaluate the independent standard-Normal latent prior."""

                return np.sum(norm.logpdf(values), axis=1)

            def log_likelihood(values: NDArray[np.float64]) -> NDArray[np.float64]:
                """Evaluate one spectral model for latent nuisance rows."""

                size = len(values)
                latent = np.zeros((size, len(nuisance_prior.latent_names)))
                latent[:, coordinates] = values
                base_nuisance = nuisance_prior.transform_latent(
                    latent, white_noise_centre=white_noise_centre
                )
                repeats = (
                    1 if label == "noise" else self.stellar_draws_per_nuisance
                )
                nuisance = {
                    name: np.repeat(draws, repeats)
                    for name, draws in base_nuisance.items()
                }
                rows = AsteroScaleSamples(
                    stellar_samples.draw(size * repeats, rng=row_rng)
                )
                scaled = dict(rows.values)
                scaled["A_env"] = scaled["A_env"] * nuisance["envelope_scale"]
                scaled["A_gran"] = scaled["A_gran"] * nuisance["granulation_scale"]
                inference_samples = AsteroScaleSamples(scaled)
                likelihood = self.marginalizer.log_likelihoods(
                    spectrum,
                    inference_samples,
                    nuisance["white_noise"],
                    observation=observation,
                    granulation_variance_fraction_low=nuisance[
                        "granulation_variance_fraction_low"
                    ],
                    overdispersion=nuisance["overdispersion"],
                    spectral_window=spectral_window,
                    model_labels=(label,),
                )[label]
                grouped = likelihood.reshape(size, repeats)
                # Average likelihood, not log likelihood, over the empirical
                # AsteroScale row distribution.
                result = logsumexp(grouped, axis=1) - np.log(repeats)
                last["samples"] = AsteroScaleSamples(
                    {
                        name: draws.reshape(size, repeats)[:, 0]
                        for name, draws in inference_samples.values.items()
                    }
                )
                last["nuisance"] = base_nuisance
                return result

            result = self.sampler.run(
                prior_sample,
                prior_logpdf,
                log_likelihood,
                rng=np.random.default_rng(generator.integers(0, 2**63 - 1)),
            )
            evidences[label] = result.log_evidence
            importance_diagnostics[label] = result.diagnostic
            monte_carlo[label] = MonteCarloDiagnostic(
                result.diagnostic.draws,
                result.diagnostic.effective_sample_size,
                result.diagnostic.log_evidence_standard_error,
            )
            if label == "oscillation":
                saved_samples = cast(AsteroScaleSamples, last["samples"])
                saved_nuisance = cast(
                    Mapping[str, NDArray[np.float64]], last["nuisance"]
                )

        assert saved_samples is not None and saved_nuisance is not None
        return AdaptiveMarginalEvaluation(
            self.marginalizer.from_evidences(evidences, monte_carlo),
            MappingProxyType(importance_diagnostics),
            saved_samples,
            saved_nuisance,
        )
