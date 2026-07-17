"""Bayesian power-excess detection for solar-like oscillations."""

from .components import GaussianEnvelope, HarveyComponent
from .asteroscale import AsteroScaleSamples
from .data import PowerSpectrum
from .likelihoods import gamma_log_likelihood
from .mixture import MixtureEvaluation, WholeSpectrumMixture
from .models import SpectralModel
from .observation import (
    DEFAULT_TOTAL_MODE_VISIBILITY,
    ObservationModel,
    cadence_amplitude_response,
    envelope_integrated_power,
)
from .priors import JointPrior
from .simulation import simulate_periodogram

__all__ = [
    "GaussianEnvelope",
    "HarveyComponent",
    "AsteroScaleSamples",
    "JointPrior",
    "MixtureEvaluation",
    "ObservationModel",
    "PowerSpectrum",
    "SpectralModel",
    "WholeSpectrumMixture",
    "DEFAULT_TOTAL_MODE_VISIBILITY",
    "cadence_amplitude_response",
    "envelope_integrated_power",
    "gamma_log_likelihood",
    "simulate_periodogram",
]

__version__ = "0.1.0"
