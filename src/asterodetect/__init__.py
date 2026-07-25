"""Bayesian power-excess detection for solar-like oscillations."""

from .components import GaussianEnvelope, HarveyComponent
from .calibration import (
    CalibrationResult,
    DetectionMetrics,
    InjectionCase,
    ProbabilityBin,
    Recovery,
    build_detection_study,
    build_injection_grid,
    evaluate_injections,
    probability_reliability,
    summarize_recoveries,
    threshold_curve,
)
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
from .simulation import (
    DEFAULT_MODE_VISIBILITIES,
    AstrophysicalInjectionFactory,
    lorentzian_mode_comb,
    regular_frequency_grid,
    simulate_periodogram,
)

__all__ = [
    "GaussianEnvelope",
    "HarveyComponent",
    "CalibrationResult",
    "DetectionMetrics",
    "InjectionCase",
    "ProbabilityBin",
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
    "DEFAULT_MODE_VISIBILITIES",
    "AstrophysicalInjectionFactory",
    "cadence_amplitude_response",
    "envelope_integrated_power",
    "granulation_component_amplitudes",
    "gamma_log_likelihood",
    "simulate_periodogram",
    "lorentzian_mode_comb",
    "regular_frequency_grid",
    "build_detection_study",
    "build_injection_grid",
    "evaluate_injections",
    "probability_reliability",
    "summarize_recoveries",
    "threshold_curve",
]

__version__ = "0.1.0"
