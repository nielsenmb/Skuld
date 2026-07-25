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

    Parameters
    ----------
    values
        Mapping containing every quantity in ``ASTERO_SCALE_PARAMETERS``.
        Arrays must be finite, one-dimensional, and equally long.
    """

    values: Mapping[str, NDArray[np.float64]]

    def __init__(self, values: Mapping[str, Any]) -> None:
        """Validate and store aligned AsteroScale sample rows."""

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

        Parameters
        ----------
        given
            Stellar or seismic measurements accepted by AsteroScale.
        solver
            Optional preconfigured AsteroScale solver.
        bandpass
            Observational bandpass used for amplitude predictions.
        input_mode
            AsteroScale conditioning mode.
        **solve_kwargs
            Additional arguments passed to ``solver.solve``.

        Returns
        -------
        AsteroScaleSamples
            Aligned joint samples required by Skuld.
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
        """Return the number of aligned sample rows.

        Returns
        -------
        int
            Number of joint AsteroScale samples.
        """

        return next(iter(self.values.values())).size

    def draw(
        self,
        size: int,
        *,
        rng: np.random.Generator | int | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Resample complete rows, preserving all joint correlations.

        Parameters
        ----------
        size
            Number of rows to draw with replacement.
        rng
            Random generator or seed.

        Returns
        -------
        dict
            Arrays keyed by AsteroScale parameter name.
        """

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

        Parameters
        ----------
        dnu_scale
            Candidate bin width in units of the median large separation.
        minimum_envelope_bins
            Minimum number of bins across the median envelope FWHM.

        Returns
        -------
        float
            Suggested physical bin width in microhertz.
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
        """Bin a spectrum once using the AsteroScale prediction.

        Parameters
        ----------
        spectrum
            Unbinned power spectrum.
        dnu_scale
            Candidate bin width in units of the median large separation.
        minimum_envelope_bins
            Minimum number of bins across the envelope FWHM.
        origin
            Optional origin of the fixed physical-frequency bins.

        Returns
        -------
        PowerSpectrum
            Fixed-bin power spectrum.
        """

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

        Parameters
        ----------
        observation
            Observation response model.

        Returns
        -------
        mapping
            Aligned ``integrated_power``, ``numax``, and ``sigma`` arrays.
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

    def granulation_parameters(
        self,
        observation: "ObservationModel | None" = None,
    ) -> Mapping[str, NDArray[np.float64]]:
        """Return aligned two-component Kallinger background parameters.

        Parameters
        ----------
        observation
            Observation response model.

        Returns
        -------
        mapping
            Component amplitudes, characteristic frequencies, and exponents.
        """

        from .observation import ObservationModel

        if observation is None:
            observation = ObservationModel()
        if not isinstance(observation, ObservationModel):
            raise TypeError("observation must be an ObservationModel")
        low, high = observation.granulation_amplitudes(self.values["A_gran"])
        parameters = {
            "amplitudes": np.column_stack((low, high)),
            "frequencies": np.column_stack(
                (self.values["b_gran_low"], self.values["b_gran_high"])
            ),
            "exponents": np.full((len(self), 2), 4.0),
        }
        for values in parameters.values():
            values.setflags(write=False)
        return MappingProxyType(parameters)
