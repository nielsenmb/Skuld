"""End-to-end orchestration for broad power-excess detection."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .asteroscale import AsteroScaleSamples
from .data import PowerSpectrum
from .marginal import MarginalEvaluation, PriorPredictiveMarginalizer
from .nuisance import NuisancePrior
from .observation import ObservationModel


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Complete result of one end-to-end detection run."""

    evaluation: MarginalEvaluation
    binned_spectrum: PowerSpectrum
    bin_width: float
    samples: AsteroScaleSamples
    nuisance_draws: Mapping[str, NDArray[np.float64]]

    @property
    def probabilities(self) -> Mapping[str, float]:
        return self.evaluation.responsibilities

    @property
    def classification(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__)


class Detector:
    """Run AsteroScale conditioning, fixed binning, and model marginalization."""

    def __init__(
        self,
        *,
        draws: int = 2048,
        observation: ObservationModel | None = None,
        nuisance_prior: NuisancePrior | None = None,
        model_probabilities: Mapping[str, float] | None = None,
        dnu_scale: float = 1.0,
        minimum_envelope_bins: int = 5,
    ) -> None:
        if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)):
            raise TypeError("draws must be an integer")
        if draws < 1:
            raise ValueError("draws must be positive")
        self.draws = int(draws)
        self.observation = observation or ObservationModel()
        self.nuisance_prior = nuisance_prior or NuisancePrior()
        self.model_probabilities = model_probabilities
        self.dnu_scale = float(dnu_scale)
        self.minimum_envelope_bins = minimum_envelope_bins

    def run(
        self,
        spectrum: PowerSpectrum,
        stellar_constraints: Mapping[str, Any] | AsteroScaleSamples,
        *,
        rng: np.random.Generator | int | None = None,
        white_noise_centre: float | None = None,
        solver: Any | None = None,
        **asteroscale_kwargs: Any,
    ) -> DetectionResult:
        """Return marginalized probabilities for one unbinned PSD."""

        if not isinstance(spectrum, PowerSpectrum):
            raise TypeError("spectrum must be a PowerSpectrum")
        generator = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )
        base = (
            stellar_constraints
            if isinstance(stellar_constraints, AsteroScaleSamples)
            else AsteroScaleSamples.infer(
                stellar_constraints, solver=solver, **asteroscale_kwargs
            )
        )
        bin_width = base.suggested_bin_width(
            dnu_scale=self.dnu_scale,
            minimum_envelope_bins=self.minimum_envelope_bins,
        )
        binned = spectrum.bin_by_width(bin_width)

        resampled = AsteroScaleSamples(base.draw(self.draws, rng=generator))
        nuisance = self.nuisance_prior.sample(
            spectrum,
            self.draws,
            rng=generator,
            white_noise_centre=white_noise_centre,
        )
        values = dict(resampled.values)
        values["A_env"] = values["A_env"] * nuisance["envelope_scale"]
        values["A_gran"] = values["A_gran"] * nuisance["granulation_scale"]
        inference_samples = AsteroScaleSamples(values)

        evaluation = PriorPredictiveMarginalizer(
            model_probabilities=self.model_probabilities
        ).evaluate(
            binned,
            inference_samples,
            nuisance["white_noise"],
            observation=self.observation,
            granulation_variance_fraction_low=nuisance[
                "granulation_variance_fraction_low"
            ],
            overdispersion=nuisance["overdispersion"],
        )
        return DetectionResult(
            evaluation=evaluation,
            binned_spectrum=binned,
            bin_width=bin_width,
            samples=inference_samples,
            nuisance_draws=MappingProxyType(dict(nuisance)),
        )
