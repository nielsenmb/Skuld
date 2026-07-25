"""Bayesian power-excess detection for solar-like oscillations."""

from .components import GaussianEnvelope, HarveyComponent
from .calibration import CalibrationResult, InjectionCase, Recovery, evaluate_injections
from .asteroscale import AsteroScaleSamples
from .data import PowerSpectrum
from .detector import DetectionResult, Detector
from .likelihoods import gamma_log_likelihood
from .mixture import MixtureEvaluation, WholeSpectrumMixture
from .marginal import MarginalEvaluation, MonteCarloDiagnostic, PriorPredictiveMarginalizer
from .models import SpectralModel
from .observation import (
    DEFAULT_TOTAL_MODE_VISIBILITY,
    ObservationModel,
    cadence_amplitude_response,
    envelope_integrated_power,
    granulation_component_amplitudes,
)
from .priors import JointPrior
from .nuisance import NuisancePrior
from .simulation import simulate_periodogram

__all__ = [
    "GaussianEnvelope",
    "HarveyComponent",
    "CalibrationResult",
    "InjectionCase",
    "Recovery",
    "AsteroScaleSamples",
    "DetectionResult",
    "Detector",
    "JointPrior",
    "NuisancePrior",
    "MixtureEvaluation",
    "MarginalEvaluation",
    "MonteCarloDiagnostic",
    "PriorPredictiveMarginalizer",
    "ObservationModel",
    "PowerSpectrum",
    "SpectralModel",
    "WholeSpectrumMixture",
    "DEFAULT_TOTAL_MODE_VISIBILITY",
    "cadence_amplitude_response",
    "envelope_integrated_power",
    "granulation_component_amplitudes",
    "gamma_log_likelihood",
    "simulate_periodogram",
    "evaluate_injections",
]

__version__ = "0.1.0"
