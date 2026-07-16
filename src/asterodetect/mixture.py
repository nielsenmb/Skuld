"""Mixtures over complete power-spectrum models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from .data import PowerSpectrum
from .likelihoods import model_log_likelihood
from .models import SpectralModel


@dataclass(frozen=True, slots=True)
class MixtureEvaluation:
    """Likelihood and responsibilities at fixed model parameters."""

    log_likelihood: float
    component_log_likelihoods: Mapping[str, float]
    responsibilities: Mapping[str, float]


class WholeSpectrumMixture:
    """A categorical mixture over complete spectral models.

    The supplied probabilities are prior class probabilities at this level;
    they are not fractions of PSD power and are not fitted amplitudes.
    """

    def __init__(
        self,
        models: Mapping[str, SpectralModel],
        probabilities: Mapping[str, float],
    ) -> None:
        if not models:
            raise ValueError("at least one spectral model is required")
        if set(models) != set(probabilities):
            raise ValueError("models and probabilities must have identical labels")
        if not all(isinstance(model, SpectralModel) for model in models.values()):
            raise TypeError("all models must be SpectralModel instances")

        values = np.asarray(list(probabilities.values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("mixture probabilities must be finite and positive")
        if not np.isclose(values.sum(), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("mixture probabilities must sum to one")

        self._models = MappingProxyType(dict(models))
        self._probabilities = MappingProxyType(
            {label: float(probabilities[label]) for label in models}
        )

    @property
    def models(self) -> Mapping[str, SpectralModel]:
        return self._models

    @property
    def probabilities(self) -> Mapping[str, float]:
        return self._probabilities

    def evaluate(self, spectrum: PowerSpectrum) -> MixtureEvaluation:
        """Evaluate the mixture and posterior responsibilities."""

        labels = tuple(self._models)
        component_logs = np.asarray(
            [
                model_log_likelihood(spectrum, self._models[label])
                for label in labels
            ]
        )
        joint_logs = component_logs + np.log(
            [self._probabilities[label] for label in labels]
        )
        total = float(logsumexp(joint_logs))
        responsibilities = np.exp(joint_logs - total)
        return MixtureEvaluation(
            log_likelihood=total,
            component_log_likelihoods=MappingProxyType(
                dict(zip(labels, map(float, component_logs), strict=True))
            ),
            responsibilities=MappingProxyType(
                dict(zip(labels, map(float, responsibilities), strict=True))
            ),
        )

