"""Bayesian power-excess detection for solar-like oscillations."""

from .components import GaussianEnvelope, HarveyComponent
from .data import PowerSpectrum
from .likelihoods import gamma_log_likelihood
from .mixture import MixtureEvaluation, WholeSpectrumMixture
from .models import SpectralModel
from .priors import JointPrior
from .simulation import simulate_periodogram

__all__ = [
    "GaussianEnvelope",
    "HarveyComponent",
    "JointPrior",
    "MixtureEvaluation",
    "PowerSpectrum",
    "SpectralModel",
    "WholeSpectrumMixture",
    "gamma_log_likelihood",
    "simulate_periodogram",
]

__version__ = "0.1.0"

