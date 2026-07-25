"""Adaptive nuisance integration while retaining empirical AsteroScale rows."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, cast

import numpy as np
from numpy.typing import NDArray
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


@dataclass(frozen=True, slots=True)
class AdaptiveMarginalEvaluation:
    """Adaptive evidence result plus model-specific importance diagnostics."""

    evaluation: MarginalEvaluation
    diagnostics: Mapping[str, ImportanceDiagnostic]
    oscillation_samples: AsteroScaleSamples
    oscillation_nuisance_draws: Mapping[str, NDArray[np.float64]]


class AdaptiveNuisanceMarginalizer:
    """Adapt nuisance proposals without fitting a density to AsteroScale rows.

    Complete AsteroScale rows are uniformly resampled for every likelihood
    batch. Only continuous nuisance coordinates are adapted, so empirical
    stellar correlations are preserved exactly.
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
        model_probabilities: Mapping[str, float] | None = None,
    ) -> None:
        self.sampler = AdaptiveImportanceSampler(
            draws=draws,
            pilot_draws=pilot_draws,
            defensive_fraction=defensive_fraction,
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
        rng: np.random.Generator | int | None = None,
    ) -> AdaptiveMarginalEvaluation:
        """Estimate all three evidences using separate defensive proposals."""

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
                return local_rng.normal(size=(size, len(coordinates)))

            def prior_logpdf(values: NDArray[np.float64]) -> NDArray[np.float64]:
                return np.sum(norm.logpdf(values), axis=1)

            def log_likelihood(values: NDArray[np.float64]) -> NDArray[np.float64]:
                latent = np.zeros((len(values), len(nuisance_prior.latent_names)))
                latent[:, coordinates] = values
                nuisance = nuisance_prior.transform_latent(
                    latent, white_noise_centre=white_noise_centre
                )
                rows = AsteroScaleSamples(stellar_samples.draw(len(values), rng=row_rng))
                scaled = dict(rows.values)
                scaled["A_env"] = scaled["A_env"] * nuisance["envelope_scale"]
                scaled["A_gran"] = scaled["A_gran"] * nuisance["granulation_scale"]
                inference_samples = AsteroScaleSamples(scaled)
                last["samples"] = inference_samples
                last["nuisance"] = nuisance
                return self.marginalizer.log_likelihoods(
                    spectrum,
                    inference_samples,
                    nuisance["white_noise"],
                    observation=observation,
                    granulation_variance_fraction_low=nuisance[
                        "granulation_variance_fraction_low"
                    ],
                    overdispersion=nuisance["overdispersion"],
                )[label]

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
