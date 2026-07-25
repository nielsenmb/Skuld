"""Bayesian power-excess detection for solar-like oscillations."""

from .components import GaussianEnvelope, HarveyComponent
from .calibration import (
    CalibrationSplit,
    CalibrationResult,
    DetectionMetrics,
    InjectionCase,
    InjectionSplit,
    ProbabilityBin,
    Recovery,
    build_detection_study,
    build_injection_grid,
    build_regime_detection_study,
    evaluate_injections,
    probability_reliability,
    reweight_calibration_models,
    select_detection_threshold,
    split_calibration,
    split_injections,
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
    ObservingWindow,
    ObservingWindowDiagnostic,
    TESS_WINDOW_PROFILES,
    continuous_observing_window,
    lorentzian_mode_comb,
    observing_window_diagnostics,
    regular_frequency_grid,
    simulate_periodogram,
    simulate_windowed_periodogram,
    tess_observing_window,
    tess_like_observing_window,
)
from .sensitivity import (
    EstimatorComparison,
    SensitivityRun,
    SensitivityStudy,
    SensitivitySummary,
    run_sensitivity_study,
)
from .window import SpectralWindowOperator

__all__ = [
    "GaussianEnvelope",
    "HarveyComponent",
    "CalibrationResult",
    "CalibrationSplit",
    "DetectionMetrics",
    "InjectionCase",
    "InjectionSplit",
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
    "SpectralWindowOperator",
    "WholeSpectrumMixture",
    "DEFAULT_TOTAL_MODE_VISIBILITY",
    "DEFAULT_MODE_VISIBILITIES",
    "AstrophysicalInjectionFactory",
    "ObservingWindow",
    "ObservingWindowDiagnostic",
    "TESS_WINDOW_PROFILES",
    "cadence_amplitude_response",
    "continuous_observing_window",
    "envelope_integrated_power",
    "granulation_component_amplitudes",
    "gamma_log_likelihood",
    "simulate_periodogram",
    "simulate_windowed_periodogram",
    "tess_observing_window",
    "tess_like_observing_window",
    "observing_window_diagnostics",
    "lorentzian_mode_comb",
    "regular_frequency_grid",
    "SensitivityRun",
    "SensitivityStudy",
    "SensitivitySummary",
    "run_sensitivity_study",
    "build_detection_study",
    "build_injection_grid",
    "build_regime_detection_study",
    "evaluate_injections",
    "probability_reliability",
    "reweight_calibration_models",
    "select_detection_threshold",
    "split_calibration",
    "split_injections",
    "summarize_recoveries",
    "threshold_curve",
    "AdaptiveImportanceSampler",
    "ImportanceDiagnostic",
    "ImportanceResult",
]

__version__ = "0.1.0"
