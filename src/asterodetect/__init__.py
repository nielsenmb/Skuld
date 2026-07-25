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
from .adaptive import AdaptiveMarginalEvaluation, AdaptiveNuisanceMarginalizer
from .data import PowerSpectrum
from .detector import DetectionResult, Detector
from .likelihoods import gamma_log_likelihood
from .mixture import MixtureEvaluation, WholeSpectrumMixture
from .marginal import MarginalEvaluation, MonteCarloDiagnostic, PriorPredictiveMarginalizer
from .importance import (
    AdaptiveImportanceSampler,
    ImportanceDiagnostic,
    ImportanceResult,
)
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
from .sensitivity import (
    EstimatorComparison,
    SensitivityRun,
    SensitivityStudy,
    SensitivitySummary,
    run_sensitivity_study,
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
    "AdaptiveMarginalEvaluation",
    "AdaptiveNuisanceMarginalizer",
    "DetectionResult",
    "Detector",
    "EstimatorComparison",
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
    "SensitivityRun",
    "SensitivityStudy",
    "SensitivitySummary",
    "run_sensitivity_study",
    "build_detection_study",
    "build_injection_grid",
    "evaluate_injections",
    "probability_reliability",
    "summarize_recoveries",
    "threshold_curve",
    "AdaptiveImportanceSampler",
    "ImportanceDiagnostic",
    "ImportanceResult",
]

__version__ = "0.1.0"
