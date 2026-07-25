"""Prior-predictive marginalization over complete spectral models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf, gammaln, logsumexp

from .asteroscale import AsteroScaleSamples
from .data import PowerSpectrum
from .observation import (
    ObservationModel,
    cadence_amplitude_response,
    granulation_component_amplitudes,
)


@dataclass(frozen=True, slots=True)
class MonteCarloDiagnostic:
    """Convergence information for one prior-predictive integral."""

    draws: int
    effective_sample_size: float
    log_evidence_standard_error: float


@dataclass(frozen=True, slots=True)
class MarginalEvaluation:
    """Marginal likelihoods and posterior complete-model probabilities."""

    log_evidences: Mapping[str, float]
    responsibilities: Mapping[str, float]
    diagnostics: Mapping[str, MonteCarloDiagnostic]


def _log_evidence(log_likelihoods: NDArray[np.float64]) -> tuple[float, MonteCarloDiagnostic]:
    n = log_likelihoods.size
    log_z = float(logsumexp(log_likelihoods) - np.log(n))
    weights = np.exp(log_likelihoods - np.max(log_likelihoods))
    ess = float(weights.sum() ** 2 / np.sum(weights**2))
    # Delta-method uncertainty of log(mean(likelihood)).
    standard_error = float(np.sqrt(max(0.0, 1.0 / ess - 1.0 / n)))
    return log_z, MonteCarloDiagnostic(n, ess, standard_error)


def _gamma_log_likelihood_batch(power, expected, shape):
    shape = np.asarray(shape, dtype=float)
    if shape.ndim == 1:
        shape = shape[None, :]
    power = np.asarray(power, dtype=float)[None, :]
    return np.sum(
        shape * np.log(shape)
        - gammaln(shape)
        + (shape - 1.0) * np.log(power)
        - shape * np.log(expected)
        - shape * power / expected,
        axis=1,
    )


def _granulation_means(
    spectrum: PowerSpectrum,
    amplitudes: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    integration_time_seconds: float | None,
) -> NDArray[np.float64]:
    """Bin-average two normalized super-Lorentzians for every prior row."""

    nodes, weights = np.polynomial.legendre.leggauss(12)
    lower, upper = spectrum.bin_lower, spectrum.bin_upper
    widths = upper - lower
    sample_frequency = (
        0.5 * widths[:, None] * nodes[None, :]
        + 0.5 * (upper + lower)[:, None]
    )
    # Shapes: draw, component, bin, quadrature point.
    ratio = sample_frequency[None, None, :, :] / frequencies[:, :, None, None]
    normalization = 2.0 * np.sqrt(2.0) / np.pi
    height = normalization * amplitudes**2 / frequencies
    profiles = height[:, :, None, None] / (1.0 + ratio**4)
    if integration_time_seconds is not None:
        response = cadence_amplitude_response(
            sample_frequency, integration_time_seconds
        )
        profiles *= response[None, None, :, :] ** 2
    return 0.5 * np.sum(profiles * weights[None, None, None, :], axis=(1, 3))


def _envelope_means(spectrum: PowerSpectrum, parameters) -> NDArray[np.float64]:
    lower = spectrum.bin_lower[None, :]
    upper = spectrum.bin_upper[None, :]
    widths = upper - lower
    sigma = parameters["sigma"][:, None]
    numax = parameters["numax"][:, None]
    probability = 0.5 * (
        erf((upper - numax) / (np.sqrt(2.0) * sigma))
        - erf((lower - numax) / (np.sqrt(2.0) * sigma))
    )
    return parameters["integrated_power"][:, None] * probability / widths


class PriorPredictiveMarginalizer:
    """Integrate N, G, and O whole-spectrum models over joint prior draws.

    ``white_noise`` may be one positive value or one value per AsteroScale
    row.  The same rows are used for G and O, preserving all AsteroScale
    correlations.  The PSD must already have its fixed binning.
    """

    labels = ("noise", "granulation", "oscillation")

    def __init__(
        self,
        *,
        model_probabilities: Mapping[str, float] | None = None,
        overdispersion: float = 1.0,
    ) -> None:
        probabilities = model_probabilities or {label: 1 / 3 for label in self.labels}
        if set(probabilities) != set(self.labels):
            raise ValueError(f"model_probabilities must have labels {self.labels}")
        values = np.asarray([probabilities[label] for label in self.labels], dtype=float)
        if np.any(values <= 0) or not np.all(np.isfinite(values)) or not np.isclose(values.sum(), 1):
            raise ValueError("model probabilities must be positive and sum to one")
        if not np.isfinite(overdispersion) or overdispersion < 1:
            raise ValueError("overdispersion must be finite and at least one")
        self.probabilities = MappingProxyType(dict(probabilities))
        self.overdispersion = float(overdispersion)

    def evaluate(
        self,
        spectrum: PowerSpectrum,
        samples: AsteroScaleSamples,
        white_noise: ArrayLike,
        *,
        observation: ObservationModel | None = None,
        granulation_variance_fraction_low: ArrayLike | None = None,
        overdispersion: ArrayLike | None = None,
    ) -> MarginalEvaluation:
        """Return marginal model probabilities and Monte Carlo diagnostics."""

        if not isinstance(spectrum, PowerSpectrum):
            raise TypeError("spectrum must be a PowerSpectrum")
        if not isinstance(samples, AsteroScaleSamples):
            raise TypeError("samples must be AsteroScaleSamples")
        observation = observation or ObservationModel()
        white = np.asarray(white_noise, dtype=float)
        if white.ndim == 0:
            white = np.full(len(samples), float(white))
        if white.shape != (len(samples),) or np.any(white <= 0) or not np.all(np.isfinite(white)):
            raise ValueError("white_noise must be positive and scalar or one value per sample")

        granulation = samples.granulation_parameters(observation)
        if granulation_variance_fraction_low is not None:
            low, high = granulation_component_amplitudes(
                samples.values["A_gran"],
                variance_fraction_low=granulation_variance_fraction_low,
                bolometric_correction=observation.bolometric_correction,
                dilution=observation.dilution,
            )
            granulation = dict(granulation)
            granulation["amplitudes"] = np.column_stack((low, high))
        granulation_mean = _granulation_means(
            spectrum,
            granulation["amplitudes"],
            granulation["frequencies"],
            observation.integration_time_seconds,
        )
        envelope_mean = _envelope_means(
            spectrum, samples.envelope_parameters(observation)
        )
        noise_mean = np.broadcast_to(white[:, None], granulation_mean.shape)
        expected = {
            "noise": noise_mean,
            "granulation": noise_mean + granulation_mean,
            "oscillation": noise_mean + granulation_mean + envelope_mean,
        }
        dispersion = self.overdispersion if overdispersion is None else np.asarray(
            overdispersion, dtype=float
        )
        if np.ndim(dispersion) == 0:
            if not np.isfinite(dispersion) or dispersion < 1:
                raise ValueError("overdispersion must be finite and at least one")
            shape = spectrum.bins_averaged / dispersion
        else:
            if (
                dispersion.shape != (len(samples),)
                or not np.all(np.isfinite(dispersion))
                or np.any(dispersion < 1)
            ):
                raise ValueError(
                    "overdispersion must be scalar or one value per sample, all >= 1"
                )
            shape = spectrum.bins_averaged[None, :] / dispersion[:, None]
        evidences: dict[str, float] = {}
        diagnostics: dict[str, MonteCarloDiagnostic] = {}
        for label in self.labels:
            likelihoods = _gamma_log_likelihood_batch(
                spectrum.power, expected[label], shape
            )
            evidences[label], diagnostics[label] = _log_evidence(likelihoods)
        joint = np.asarray(
            [evidences[label] + np.log(self.probabilities[label]) for label in self.labels]
        )
        total = logsumexp(joint)
        responsibilities = {
            label: float(value) for label, value in zip(self.labels, np.exp(joint - total), strict=True)
        }
        return MarginalEvaluation(
            MappingProxyType(evidences),
            MappingProxyType(responsibilities),
            MappingProxyType(diagnostics),
        )
