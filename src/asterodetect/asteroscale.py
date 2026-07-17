"""Boundary between AsteroScale inference and Skuld's spectral model."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


ASTERO_SCALE_PARAMETERS = (
    "numax",
    "dnu",
    "FWHM_env",
    "A_env",
    "A_gran",
    "b_gran_low",
    "b_gran_high",
)


@dataclass(frozen=True, slots=True)
class AsteroScaleSamples:
    """Joint AsteroScale posterior or prior-predictive samples.

    Values remain in AsteroScale's native conventions: frequencies are in
    microhertz, ``A_env`` is maximum radial-mode RMS amplitude in ppm, and
    ``A_gran`` is granulation RMS amplitude in ppm.  In particular,
    ``A_env`` is deliberately not treated as integrated Gaussian-envelope
    power; that physical conversion belongs in the later observation model.
    """

    values: Mapping[str, NDArray[np.float64]]

    def __init__(self, values: Mapping[str, Any]) -> None:
        missing = set(ASTERO_SCALE_PARAMETERS) - set(values)
        if missing:
            raise ValueError(
                "missing AsteroScale quantities: " + ", ".join(sorted(missing))
            )

        arrays: dict[str, NDArray[np.float64]] = {}
        size: int | None = None
        for name in ASTERO_SCALE_PARAMETERS:
            array = np.atleast_1d(np.asarray(values[name], dtype=float))
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite one-dimensional array")
            if size is None:
                size = array.size
            elif array.size != size:
                raise ValueError("all AsteroScale quantities must have equal length")
            array.setflags(write=False)
            arrays[name] = array
        if not size:
            raise ValueError("AsteroScale returned no samples")
        object.__setattr__(self, "values", MappingProxyType(arrays))

    @classmethod
    def infer(
        cls,
        given: Mapping[str, Any],
        *,
        solver: Any | None = None,
        bandpass: str = "TESS",
        input_mode: str = "likelihood",
        **solve_kwargs: Any,
    ) -> "AsteroScaleSamples":
        """Run AsteroScale and retain the joint prediction needed by Skuld.

        ``input_mode='likelihood'`` is the default because detection priors
        should condition a population model on measurements.  Pass
        ``'propagate'`` explicitly for calculator-style uncertainty
        propagation.
        """

        if solver is None:
            from asteroscale import Solver

            solver = Solver(bandpass=bandpass, input_mode=input_mode)
        result = solver.solve(
            dict(given),
            want=list(ASTERO_SCALE_PARAMETERS),
            bandpass=bandpass,
            input_mode=input_mode,
            **solve_kwargs,
        )
        return cls(result)

    def __len__(self) -> int:
        return next(iter(self.values.values())).size

    def draw(
        self,
        size: int,
        *,
        rng: np.random.Generator | int | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Resample complete rows, preserving all joint correlations."""

        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise TypeError("size must be an integer")
        if size < 1:
            raise ValueError("size must be positive")
        generator = (
            rng
            if isinstance(rng, np.random.Generator)
            else np.random.default_rng(rng)
        )
        indices = generator.integers(0, len(self), size=size)
        return {name: values[indices] for name, values in self.values.items()}

    def suggested_bin_width(
        self,
        *,
        dnu_scale: float = 1.0,
        minimum_envelope_bins: int = 5,
    ) -> float:
        """Choose a fixed PSD bin width from the independent prior.

        The default averages approximately one radial order while retaining
        at least five bins across the predicted envelope FWHM.
        """

        dnu_scale = float(dnu_scale)
        if not np.isfinite(dnu_scale) or dnu_scale <= 0:
            raise ValueError("dnu_scale must be finite and positive")
        if (
            isinstance(minimum_envelope_bins, bool)
            or not isinstance(minimum_envelope_bins, (int, np.integer))
            or minimum_envelope_bins < 1
        ):
            raise ValueError("minimum_envelope_bins must be a positive integer")
        dnu_width = dnu_scale * float(np.median(self.values["dnu"]))
        envelope_width = float(np.median(self.values["FWHM_env"]))
        return min(dnu_width, envelope_width / minimum_envelope_bins)

    def bin_spectrum(
        self,
        spectrum: "PowerSpectrum",
        *,
        dnu_scale: float = 1.0,
        minimum_envelope_bins: int = 5,
        origin: float | None = None,
    ) -> "PowerSpectrum":
        """Bin a spectrum once using the AsteroScale prediction."""

        from .data import PowerSpectrum

        if not isinstance(spectrum, PowerSpectrum):
            raise TypeError("spectrum must be a PowerSpectrum")
        width = self.suggested_bin_width(
            dnu_scale=dnu_scale,
            minimum_envelope_bins=minimum_envelope_bins,
        )
        return spectrum.bin_by_width(width, origin=origin)

    def envelope_parameters(
        self,
        observation: "ObservationModel | None" = None,
    ) -> Mapping[str, NDArray[np.float64]]:
        """Return correlated Gaussian-envelope parameters for inference.

        The returned arrays retain AsteroScale's sample-row ordering.  The
        envelope power includes the observation model's visibility, cadence,
        and dilution terms; ``sigma`` is in microhertz and power is in ppm
        squared.
        """

        from .observation import ObservationModel

        if observation is None:
            observation = ObservationModel()
        if not isinstance(observation, ObservationModel):
            raise TypeError("observation must be an ObservationModel")
        numax = self.values["numax"]
        fwhm = self.values["FWHM_env"]
        parameters = {
            "integrated_power": observation.envelope_power(
                self.values["A_env"],
                self.values["dnu"],
                fwhm,
                numax=numax,
            ),
            "numax": numax,
            "sigma": fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))),
        }
        for values in parameters.values():
            values.setflags(write=False)
        return MappingProxyType(parameters)
