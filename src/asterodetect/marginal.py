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
from .window import SpectralWindowOperator


@dataclass(frozen=True, slots=True)
class MonteCarloDiagnostic:
    """Convergence information for one prior-predictive integral.

    Attributes
    ----------
    draws
        Number of Monte Carlo likelihood evaluations.
    effective_sample_size
        Effective number of likelihood-weighted draws.
    log_evidence_standard_error
        Delta-method uncertainty of the log evidence.
    """

    draws: int
    effective_sample_size: float
    log_evidence_standard_error: float


@dataclass(frozen=True, slots=True)
class MarginalEvaluation:
    """Marginal likelihoods and posterior complete-model probabilities.

    Attributes
    ----------
    log_evidences
        Natural-log marginal likelihood keyed by model.
    responsibilities
        Posterior model probabilities.
    diagnostics
        Model-specific Monte Carlo diagnostics.
    """

    log_evidences: Mapping[str, float]
    responsibilities: Mapping[str, float]
    diagnostics: Mapping[str, MonteCarloDiagnostic]

    def reweight(
        self,
        model_probabilities: Mapping[str, float],
    ) -> "MarginalEvaluation":
        """Recombine existing evidences under alternative model priors.

        This diagnostic does not repeat any likelihood calculations. Models
        omitted from ``model_probabilities`` receive posterior probability
        zero, while retained models are normalized using the supplied priors.

        Parameters
        ----------
        model_probabilities
            Prior probabilities for at least two models present in
            :attr:`log_evidences`. Values must be positive and sum to one.

        Returns
        -------
        MarginalEvaluation
            The same evidences and diagnostics with recomputed model
            probabilities.
        """

        supplied = dict(model_probabilities)
        if len(supplied) < 2:
            raise ValueError("model_probabilities must retain at least two models")
        unknown = set(supplied) - set(self.log_evidences)
        if unknown:
            raise ValueError(
                "model_probabilities contains models without evidences: "
                f"{sorted(unknown)}"
            )
        priors = np.asarray(list(supplied.values()), dtype=float)
        if (
            np.any(priors <= 0)
            or not np.all(np.isfinite(priors))
            or not np.isclose(priors.sum(), 1.0)
        ):
            raise ValueError("model probabilities must be positive and sum to one")

        labels = tuple(self.log_evidences)
        joint = np.full(len(labels), -np.inf)
        for index, label in enumerate(labels):
            if label in supplied:
                joint[index] = (
                    self.log_evidences[label] + np.log(supplied[label])
                )
        total = logsumexp(joint)
        responsibilities = {
            label: float(probability)
            for label, probability in zip(
                labels,
                np.exp(joint - total),
                strict=True,
            )
        }
        return MarginalEvaluation(
            self.log_evidences,
            MappingProxyType(responsibilities),
            self.diagnostics,
        )


def _log_evidence(log_likelihoods: NDArray[np.float64]) -> tuple[float, MonteCarloDiagnostic]:
    """Estimate log evidence and Monte Carlo diagnostics from prior draws."""

    n = log_likelihoods.size
    log_z = float(logsumexp(log_likelihoods) - np.log(n))
    weights = np.exp(log_likelihoods - np.max(log_likelihoods))
    ess = float(weights.sum() ** 2 / np.sum(weights**2))
    # Delta-method uncertainty of log(mean(likelihood)).
    standard_error = float(np.sqrt(max(0.0, 1.0 / ess - 1.0 / n)))
    return log_z, MonteCarloDiagnostic(n, ess, standard_error)


def _gamma_log_likelihood_batch(power, expected, shape):
    """Evaluate Gamma log likelihoods for a batch of expected spectra."""

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
    """Bin-average two normalized super-Lorentzians for every prior row.

    Parameters
    ----------
    spectrum
        Fixed-bin observed spectrum.
    amplitudes
        Component RMS amplitudes with shape ``(draw, component)``.
    frequencies
        Characteristic frequencies aligned with ``amplitudes``.
    integration_time_seconds
        Optional exposure time for cadence apodization.

    Returns
    -------
    numpy.ndarray
        Granulation mean with shape ``(draw, bin)``.
    """

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
    """Average Gaussian envelopes over the observed frequency bins."""

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


def _windowed_component_means(
    spectrum: PowerSpectrum,
    operator: SpectralWindowOperator,
    *,
    white_noise: NDArray[np.float64],
    granulation_amplitudes: NDArray[np.float64],
    granulation_frequencies: NDArray[np.float64],
    envelope_parameters: Mapping[str, NDArray[np.float64]],
    integration_time_seconds: float | None,
    include_granulation: bool = True,
    include_envelope: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return window-convolved noise, granulation, and envelope bin means."""

    frequency = operator.frequency
    response_squared = (
        cadence_amplitude_response(frequency, integration_time_seconds) ** 2
        if integration_time_seconds is not None
        else np.ones_like(frequency)
    )
    noise_template = operator.convolve_and_bin(
        np.ones_like(frequency),
        spectrum,
    )
    noise_mean = white_noise[:, None] * noise_template[None, :]
    granulation_mean = np.zeros((white_noise.size, spectrum.power.size))
    envelope_mean = np.zeros_like(granulation_mean)
    normalization = 2.0 * np.sqrt(2.0) / np.pi
    for start in range(0, white_noise.size, operator.row_batch_size):
        stop = min(start + operator.row_batch_size, white_noise.size)
        if include_granulation:
            amplitudes = granulation_amplitudes[start:stop]
            characteristic = granulation_frequencies[start:stop]
            ratio = frequency[None, None, :] / characteristic[:, :, None]
            height = normalization * amplitudes**2 / characteristic
            granulation_power = np.sum(
                height[:, :, None] / (1.0 + ratio**4),
                axis=1,
            )
            granulation_power *= response_squared[None, :]
            granulation_mean[start:stop] = operator.convolve_and_bin(
                granulation_power,
                spectrum,
            )

        if include_envelope:
            integrated_power = envelope_parameters["integrated_power"][
                start:stop, None
            ]
            numax = envelope_parameters["numax"][start:stop, None]
            sigma = envelope_parameters["sigma"][start:stop, None]
            envelope_power = (
                integrated_power
                / (np.sqrt(2.0 * np.pi) * sigma)
                * np.exp(
                    -0.5 * ((frequency[None, :] - numax) / sigma) ** 2
                )
            )
            envelope_mean[start:stop] = operator.convolve_and_bin(
                envelope_power,
                spectrum,
            )
    return noise_mean, granulation_mean, envelope_mean


class PriorPredictiveMarginalizer:
    """Integrate N, G, and O whole-spectrum models over joint prior draws.

    ``white_noise`` may be one positive value or one value per AsteroScale
    row.  The same rows are used for G and O, preserving all AsteroScale
    correlations.  The PSD must already have its fixed binning.

    Parameters
    ----------
    model_probabilities
        Prior probabilities for the three complete models.
    overdispersion
        Default factor reducing the nominal Gamma shape.
    """

    labels = ("noise", "granulation", "oscillation")

    def __init__(
        self,
        *,
        model_probabilities: Mapping[str, float] | None = None,
        overdispersion: float = 1.0,
    ) -> None:
        """Validate and store model priors and default overdispersion."""

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
        spectral_window: SpectralWindowOperator | None = None,
    ) -> MarginalEvaluation:
        """Return marginal model probabilities and Monte Carlo diagnostics.

        Parameters
        ----------
        spectrum
            Fixed-bin observed power spectrum.
        samples
            Aligned AsteroScale sample rows.
        white_noise
            Scalar or sample-aligned white-noise levels.
        observation
            Observation response model.
        granulation_variance_fraction_low
            Optional low-frequency Harvey variance fraction per sample.
        overdispersion
            Optional scalar or sample-aligned overdispersion.
        spectral_window
            Optional target-specific operator applied to every complete
            spectral model before likelihood evaluation.

        Returns
        -------
        MarginalEvaluation
            Evidences, posterior model probabilities, and diagnostics.
        """

        if not isinstance(spectrum, PowerSpectrum):
            raise TypeError("spectrum must be a PowerSpectrum")
        if not isinstance(samples, AsteroScaleSamples):
            raise TypeError("samples must be AsteroScaleSamples")
        likelihoods = self.log_likelihoods(
            spectrum,
            samples,
            white_noise,
            observation=observation,
            granulation_variance_fraction_low=granulation_variance_fraction_low,
            overdispersion=overdispersion,
            spectral_window=spectral_window,
        )
        evidences: dict[str, float] = {}
        diagnostics: dict[str, MonteCarloDiagnostic] = {}
        for label in self.labels:
            evidences[label], diagnostics[label] = _log_evidence(likelihoods[label])
        return self.from_evidences(evidences, diagnostics)

    def log_likelihoods(
        self,
        spectrum: PowerSpectrum,
        samples: AsteroScaleSamples,
        white_noise: ArrayLike,
        *,
        observation: ObservationModel | None = None,
        granulation_variance_fraction_low: ArrayLike | None = None,
        overdispersion: ArrayLike | None = None,
        spectral_window: SpectralWindowOperator | None = None,
        model_labels: tuple[str, ...] | None = None,
    ) -> Mapping[str, NDArray[np.float64]]:
        """Evaluate every complete model for aligned stellar and nuisance rows.

        Parameters
        ----------
        spectrum
            Fixed-bin observed power spectrum.
        samples
            Aligned AsteroScale sample rows.
        white_noise
            Scalar or sample-aligned white-noise levels.
        observation
            Observation response model.
        granulation_variance_fraction_low
            Optional low-frequency Harvey variance fraction per sample.
        overdispersion
            Optional scalar or sample-aligned overdispersion.
        spectral_window
            Optional target-specific operator applied to every complete
            spectral model before likelihood evaluation.
        model_labels
            Optional subset of complete models to evaluate.

        Returns
        -------
        mapping
            Sample log likelihoods keyed by complete-model label.
        """

        if not isinstance(spectrum, PowerSpectrum):
            raise TypeError("spectrum must be a PowerSpectrum")
        if not isinstance(samples, AsteroScaleSamples):
            raise TypeError("samples must be AsteroScaleSamples")
        requested_labels = (
            self.labels if model_labels is None else tuple(model_labels)
        )
        if (
            not requested_labels
            or len(set(requested_labels)) != len(requested_labels)
            or not set(requested_labels).issubset(self.labels)
        ):
            raise ValueError(
                f"model_labels must be a non-empty subset of {self.labels}"
            )
        include_granulation = bool(
            {"granulation", "oscillation"} & set(requested_labels)
        )
        include_envelope = "oscillation" in requested_labels
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
        envelope_parameters = samples.envelope_parameters(observation)
        if spectral_window is None:
            granulation_mean = _granulation_means(
                spectrum,
                granulation["amplitudes"],
                granulation["frequencies"],
                observation.integration_time_seconds,
            )
            envelope_mean = _envelope_means(
                spectrum, envelope_parameters
            )
            noise_mean = np.broadcast_to(
                white[:, None], granulation_mean.shape
            )
        else:
            if not isinstance(spectral_window, SpectralWindowOperator):
                raise TypeError(
                    "spectral_window must be a SpectralWindowOperator"
                )
            noise_mean, granulation_mean, envelope_mean = (
                _windowed_component_means(
                    spectrum,
                    spectral_window,
                    white_noise=white,
                    granulation_amplitudes=granulation["amplitudes"],
                    granulation_frequencies=granulation["frequencies"],
                    envelope_parameters=envelope_parameters,
                    integration_time_seconds=observation.integration_time_seconds,
                    include_granulation=include_granulation,
                    include_envelope=include_envelope,
                )
            )
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
        likelihoods: dict[str, NDArray[np.float64]] = {}
        for label in requested_labels:
            likelihoods[label] = _gamma_log_likelihood_batch(
                spectrum.power, expected[label], shape
            )
        return MappingProxyType(likelihoods)

    def from_evidences(
        self,
        evidences: Mapping[str, float],
        diagnostics: Mapping[str, MonteCarloDiagnostic],
    ) -> MarginalEvaluation:
        """Combine model evidences with configured model probabilities.

        Parameters
        ----------
        evidences
            Natural-log evidences keyed by model label.
        diagnostics
            Monte Carlo diagnostics keyed by model label.

        Returns
        -------
        MarginalEvaluation
            Normalized posterior model probabilities and supplied inputs.
        """

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
