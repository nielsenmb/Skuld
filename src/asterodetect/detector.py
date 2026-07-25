"""End-to-end orchestration for broad power-excess detection."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .asteroscale import AsteroScaleSamples
from .adaptive import AdaptiveNuisanceMarginalizer
from .data import PowerSpectrum
from .marginal import MarginalEvaluation, PriorPredictiveMarginalizer
from .nuisance import NuisancePrior
from .observation import ObservationModel


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Complete result of one end-to-end detection run.

    Attributes
    ----------
    evaluation
        Marginal evidences, model probabilities, and convergence diagnostics.
    binned_spectrum
        Fixed-bin spectrum used for likelihood evaluation.
    bin_width
        Physical bin width in microhertz.
    samples
        AsteroScale sample rows used for the oscillation model.
    nuisance_draws
        Nuisance values aligned with ``samples``.
    estimator
        Evidence estimator name.
    """

    evaluation: MarginalEvaluation
    binned_spectrum: PowerSpectrum
    bin_width: float
    samples: AsteroScaleSamples
    nuisance_draws: Mapping[str, NDArray[np.float64]]
    estimator: str = "prior"

    @property
    def probabilities(self) -> Mapping[str, float]:
        """Return posterior probabilities keyed by model label.

        Returns
        -------
        mapping
            Read-only noise, granulation, and oscillation probabilities.
        """

        return self.evaluation.responsibilities

    @property
    def classification(self) -> str:
        """Return the maximum-posterior model label.

        Returns
        -------
        str
            One of ``"noise"``, ``"granulation"``, or ``"oscillation"``.
        """

        return max(self.probabilities, key=self.probabilities.__getitem__)


class Detector:
    """Run AsteroScale conditioning, fixed binning, and model marginalization.

    Parameters
    ----------
    draws
        Number of final Monte Carlo samples.
    observation
        Cadence, dilution, visibility, and bolometric response model.
    nuisance_prior
        Target-specific nuisance prior.
    model_probabilities
        Prior probabilities for the three spectral models.
    dnu_scale
        Bin width as a multiple of the predicted large separation.
    minimum_envelope_bins
        Minimum number of bins retained across the predicted envelope FWHM.
    estimator
        ``"prior"`` for plain prior averaging or ``"adaptive"`` for
        defensive importance sampling.
    pilot_draws
        Number of prior draws used to fit each adaptive proposal.
    defensive_fraction
        Fraction of adaptive samples drawn directly from the prior.
    pilot_ess_fraction
        Minimum pilot ESS fraction enforced through likelihood tempering.
    proposal_degrees_of_freedom
        Degrees of freedom of the adaptive Student proposal.
    stellar_draws_per_nuisance
        Intact AsteroScale rows averaged at each adaptive nuisance point.
    """

    def __init__(
        self,
        *,
        draws: int = 2048,
        observation: ObservationModel | None = None,
        nuisance_prior: NuisancePrior | None = None,
        model_probabilities: Mapping[str, float] | None = None,
        dnu_scale: float = 1.0,
        minimum_envelope_bins: int = 5,
        estimator: str = "prior",
        pilot_draws: int = 256,
        defensive_fraction: float = 0.2,
        pilot_ess_fraction: float = 0.1,
        proposal_degrees_of_freedom: float = 5.0,
        stellar_draws_per_nuisance: int = 8,
    ) -> None:
        """Initialize an end-to-end detector."""

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
        if estimator not in {"prior", "adaptive"}:
            raise ValueError("estimator must be 'prior' or 'adaptive'")
        self.estimator = estimator
        self.pilot_draws = pilot_draws
        self.defensive_fraction = defensive_fraction
        self.pilot_ess_fraction = pilot_ess_fraction
        self.proposal_degrees_of_freedom = proposal_degrees_of_freedom
        self.stellar_draws_per_nuisance = stellar_draws_per_nuisance

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
        """Return marginalized probabilities for one unbinned PSD.

        Parameters
        ----------
        spectrum
            Unbinned observed power spectrum.
        stellar_constraints
            Existing AsteroScale samples or measurements passed to AsteroScale.
        rng
            Random generator or seed.
        white_noise_centre
            Optional positive centre for the white-noise prior.
        solver
            Optional preconfigured AsteroScale solver.
        **asteroscale_kwargs
            Additional keywords passed to :meth:`AsteroScaleSamples.infer`.

        Returns
        -------
        DetectionResult
            Model probabilities, diagnostics, binning, and inference draws.
        """

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

        if self.estimator == "adaptive":
            centre = (
                self.nuisance_prior.estimate_white_noise(spectrum)
                if white_noise_centre is None
                else float(white_noise_centre)
            )
            adaptive = AdaptiveNuisanceMarginalizer(
                draws=self.draws,
                pilot_draws=self.pilot_draws,
                defensive_fraction=self.defensive_fraction,
                pilot_ess_fraction=self.pilot_ess_fraction,
                proposal_degrees_of_freedom=self.proposal_degrees_of_freedom,
                stellar_draws_per_nuisance=self.stellar_draws_per_nuisance,
                model_probabilities=self.model_probabilities,
            ).evaluate(
                binned,
                base,
                self.nuisance_prior,
                white_noise_centre=centre,
                observation=self.observation,
                rng=generator,
            )
            return DetectionResult(
                evaluation=adaptive.evaluation,
                binned_spectrum=binned,
                bin_width=bin_width,
                samples=adaptive.oscillation_samples,
                nuisance_draws=adaptive.oscillation_nuisance_draws,
                estimator="adaptive",
            )

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
            estimator="prior",
        )
