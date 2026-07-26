"""Reproducible preparation and evaluation of real TESS light curves."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import PowerSpectrum
from .detector import Detector
from .simulation import ObservingWindow, observing_window_diagnostics

REFERENCE_LABELS = (
    "confirmed_detection",
    "reported_non_detection",
    "challenge",
)
"""Allowed literature labels for real-data validation targets."""

_SEISMIC_CONSTRAINTS = {
    "numax",
    "dnu",
    "FWHM_env",
    "A_env",
    "A_gran",
    "b_gran_low",
    "b_gran_high",
}


def _optional_float(value: Any) -> float | None:
    """Return a finite float or ``None`` for an empty value."""

    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("optional numeric values must be finite")
    return result


def _parse_sectors(value: Any) -> tuple[int, ...]:
    """Parse a semicolon-separated sector list."""

    if value is None or str(value).strip() == "":
        return ()
    sectors = tuple(int(item) for item in str(value).split(";"))
    if any(sector < 1 for sector in sectors) or len(set(sectors)) != len(sectors):
        raise ValueError("sectors must be unique positive integers")
    return sectors


def _normalise_constraints(
    constraints: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Validate independent AsteroScale constraints and freeze the mapping."""

    values = {} if constraints is None else dict(constraints)
    leaked = _SEISMIC_CONSTRAINTS.intersection(values)
    if leaked:
        raise ValueError(
            "reference seismic quantities cannot be detector constraints: "
            + ", ".join(sorted(leaked))
        )
    normalised: dict[str, Any] = {}
    for name, value in values.items():
        if isinstance(value, list):
            value = tuple(value)
        normalised[str(name)] = value
    return MappingProxyType(normalised)


@dataclass(frozen=True, slots=True)
class TessValidationTarget:
    """One real TESS target and its literature reference information.

    Published seismic values are retained only for validation after detection.
    They are deliberately forbidden from ``stellar_constraints`` to prevent
    the answer leaking into the AsteroScale prior.

    Attributes
    ----------
    name
        Human-readable target name.
    tic_id
        TESS Input Catalog identifier.
    reference_label
        Literature status: ``"confirmed_detection"``,
        ``"reported_non_detection"``, or ``"challenge"``.
    regime
        Stellar-regime label used when grouping validation results.
    reference
        Citation or catalogue identifier supporting the reference label.
    stellar_constraints
        Independent measurements passed to AsteroScale. Seismic quantities
        such as ``numax`` and ``dnu`` are rejected.
    reference_numax, reference_numax_error
        Optional published frequency of maximum oscillation power and its
        uncertainty, in microhertz. These are validation labels only.
    reference_dnu, reference_dnu_error
        Optional published large separation and its uncertainty, in
        microhertz. These are validation labels only.
    preferred_cadence_seconds
        Preferred TESS exposure time, or ``None`` to select the shortest
        available product.
    sectors
        Optional TESS sectors to include. An empty tuple accepts all sectors.
    data_author
        Requested light-curve producer, or ``"auto"`` for the first
        available producer in the validation script's priority order.
    notes
        Free-text information about extraction or target selection.
    """

    name: str
    tic_id: int
    reference_label: str
    regime: str
    reference: str
    stellar_constraints: Mapping[str, Any]
    reference_numax: float | None = None
    reference_numax_error: float | None = None
    reference_dnu: float | None = None
    reference_dnu_error: float | None = None
    preferred_cadence_seconds: float | None = None
    sectors: tuple[int, ...] = ()
    data_author: str = "SPOC"
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate and freeze target metadata."""

        name = str(self.name).strip()
        if not name:
            raise ValueError("target name must not be empty")
        if isinstance(self.tic_id, bool) or int(self.tic_id) < 1:
            raise ValueError("tic_id must be a positive integer")
        if self.reference_label not in REFERENCE_LABELS:
            raise ValueError(f"reference_label must be one of {REFERENCE_LABELS}")
        regime = str(self.regime).strip()
        if not regime:
            raise ValueError("regime must not be empty")
        reference = str(self.reference).strip()
        if not reference:
            raise ValueError("reference must not be empty")
        cadence = _optional_float(self.preferred_cadence_seconds)
        if cadence is not None and cadence <= 0:
            raise ValueError("preferred cadence must be positive")
        sectors = tuple(int(sector) for sector in self.sectors)
        if any(sector < 1 for sector in sectors) or len(set(sectors)) != len(
            sectors
        ):
            raise ValueError("sectors must be unique positive integers")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tic_id", int(self.tic_id))
        object.__setattr__(self, "regime", regime)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(
            self,
            "stellar_constraints",
            _normalise_constraints(self.stellar_constraints),
        )
        object.__setattr__(self, "preferred_cadence_seconds", cadence)
        object.__setattr__(self, "sectors", sectors)
        object.__setattr__(self, "data_author", str(self.data_author).strip())
        for field_name in (
            "reference_numax",
            "reference_numax_error",
            "reference_dnu",
            "reference_dnu_error",
        ):
            value = _optional_float(getattr(self, field_name))
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)


def load_tess_target_manifest(
    path: str | Path,
) -> tuple[TessValidationTarget, ...]:
    """Load and validate a CSV target manifest.

    Parameters
    ----------
    path
        Manifest containing the fields accepted by
        :class:`TessValidationTarget`. Sector numbers are separated by
        semicolons and ``stellar_constraints`` is encoded as JSON.

    Returns
    -------
    tuple of TessValidationTarget
        Targets in manifest order.

    Raises
    ------
    ValueError
        If the manifest is empty, repeats a TIC identifier, or contains
        invalid target metadata.
    """

    targets: list[TessValidationTarget] = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            constraints_text = row.get("stellar_constraints", "").strip()
            constraints = (
                {} if not constraints_text else json.loads(constraints_text)
            )
            targets.append(
                TessValidationTarget(
                    name=row["name"],
                    tic_id=int(row["tic_id"]),
                    reference_label=row["reference_label"],
                    regime=row["regime"],
                    reference=row["reference"],
                    stellar_constraints=constraints,
                    reference_numax=row.get("reference_numax"),
                    reference_numax_error=row.get("reference_numax_error"),
                    reference_dnu=row.get("reference_dnu"),
                    reference_dnu_error=row.get("reference_dnu_error"),
                    preferred_cadence_seconds=row.get(
                        "preferred_cadence_seconds"
                    ),
                    sectors=_parse_sectors(row.get("sectors")),
                    data_author=row.get("data_author", "SPOC"),
                    notes=row.get("notes", ""),
                )
            )
    if not targets:
        raise ValueError("target manifest is empty")
    tic_ids = [target.tic_id for target in targets]
    if len(set(tic_ids)) != len(tic_ids):
        raise ValueError("target manifest contains duplicate TIC IDs")
    return tuple(targets)


def _infer_cadence_seconds(time_days: NDArray[np.float64]) -> float:
    """Infer the modal short spacing of an irregular time series."""

    differences = np.diff(time_days)
    differences = differences[differences > 0]
    if differences.size == 0:
        raise ValueError("time must contain at least two distinct samples")
    lower = differences[differences <= np.quantile(differences, 0.75)]
    cadence = float(np.median(lower) * 86400.0)
    if not np.isfinite(cadence) or cadence <= 0:
        raise ValueError("could not infer a positive cadence")
    return cadence


def _collapse_long_gaps(
    time_days: NDArray[np.float64],
    *,
    cadence_days: float,
    threshold_days: float | None,
) -> NDArray[np.float64]:
    """Remove only the excess duration of very long gaps."""

    if threshold_days is None:
        return time_days.copy()
    threshold = float(threshold_days)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("long_gap_days must be positive or None")
    differences = np.diff(time_days)
    removed = np.maximum(differences - cadence_days, 0.0)
    removed[differences <= threshold] = 0.0
    offsets = np.concatenate(([0.0], np.cumsum(removed)))
    return time_days - offsets


@dataclass(frozen=True, slots=True)
class PreparedTessLightCurve:
    """A detrended TESS light curve placed on a regular zero-filled grid.

    Attributes
    ----------
    flux_ppm
        Mean-subtracted flux in parts per million. Missing cadences are zero.
    observed
        Boolean mask distinguishing observed samples from zero-filled gaps.
    cadence_seconds
        Regular sampling interval in seconds.
    start_time_days
        Time coordinate of the first scheduled cadence.
    source
        Description of the input light-curve products.
    dilution
        Fraction of aperture flux attributed to the target, in ``(0, 1]``.
    """

    flux_ppm: NDArray[np.float64]
    observed: NDArray[np.bool_]
    cadence_seconds: float
    start_time_days: float
    source: str = ""
    dilution: float = 1.0

    def __post_init__(self) -> None:
        """Validate and freeze the prepared arrays."""

        flux = np.asarray(self.flux_ppm, dtype=float)
        observed = np.asarray(self.observed)
        if flux.ndim != 1 or observed.ndim != 1 or flux.shape != observed.shape:
            raise ValueError("flux_ppm and observed must be matching 1D arrays")
        if flux.size < 4 or np.count_nonzero(observed) < 4:
            raise ValueError("prepared light curve needs at least four samples")
        if observed.dtype.kind != "b":
            raise TypeError("observed must contain boolean values")
        if not np.all(np.isfinite(flux)):
            raise ValueError("flux_ppm must contain only finite values")
        if np.any(flux[~observed] != 0):
            raise ValueError("unobserved cadences must have zero flux")
        cadence = float(self.cadence_seconds)
        if not np.isfinite(cadence) or cadence <= 0:
            raise ValueError("cadence_seconds must be finite and positive")
        dilution = float(self.dilution)
        if not np.isfinite(dilution) or not 0 < dilution <= 1:
            raise ValueError("dilution must be finite and in (0, 1]")
        flux = flux.copy()
        observed = observed.copy()
        flux.setflags(write=False)
        observed.setflags(write=False)
        object.__setattr__(self, "flux_ppm", flux)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "cadence_seconds", cadence)
        object.__setattr__(self, "start_time_days", float(self.start_time_days))
        object.__setattr__(self, "dilution", dilution)

    @classmethod
    def from_irregular(
        cls,
        time_days: ArrayLike,
        flux: ArrayLike,
        *,
        quality: ArrayLike | None = None,
        cadence_seconds: float | None = None,
        flux_unit: str = "relative",
        sigma_clip: float | None = 5.0,
        long_gap_days: float | None = 50.0,
        source: str = "",
        dilution: float = 1.0,
    ) -> PreparedTessLightCurve:
        """Regularize an already detrended light curve.

        ``flux_unit='relative'`` expects values near unity, as returned by a
        normalized Lightkurve object. ``'ppm'`` accepts an existing ppm series.
        Gaps longer than ``long_gap_days`` are shortened to one cadence, matching
        the treatment used for re-observations separated by years in Hatt et al.
        (2023).

        Parameters
        ----------
        time_days
            Irregular observation times in days.
        flux
            Detrended relative or ppm flux values.
        quality
            Optional quality flags; only zero-valued cadences are retained.
        cadence_seconds
            Known regular cadence. If omitted, it is inferred from the
            shortest common time differences.
        flux_unit
            ``"relative"`` for flux near unity or ``"ppm"`` for ppm values.
        sigma_clip
            Symmetric robust clipping threshold in scaled median absolute
            deviations, or ``None`` to disable clipping.
        long_gap_days
            Gaps longer than this are shortened to one cadence, or ``None`` to
            preserve every scheduled cadence.
        source
            Description stored with the prepared product.
        dilution
            Fraction of aperture flux attributed to the target.

        Returns
        -------
        PreparedTessLightCurve
            Immutable regular light curve and exact observing mask.
        """

        time = np.asarray(time_days, dtype=float)
        values = np.asarray(flux, dtype=float)
        if time.ndim != 1 or values.ndim != 1 or time.shape != values.shape:
            raise ValueError("time_days and flux must be matching 1D arrays")
        valid = np.isfinite(time) & np.isfinite(values)
        if quality is not None:
            quality_array = np.asarray(quality)
            if quality_array.shape != time.shape:
                raise ValueError("quality must match time_days")
            valid &= quality_array == 0
        time = time[valid]
        values = values[valid]
        if time.size < 4:
            raise ValueError("fewer than four valid cadences remain")
        order = np.argsort(time)
        time = time[order]
        values = values[order]
        cadence = (
            _infer_cadence_seconds(time)
            if cadence_seconds is None
            else float(cadence_seconds)
        )
        if not np.isfinite(cadence) or cadence <= 0:
            raise ValueError("cadence_seconds must be finite and positive")

        if flux_unit == "relative":
            centre = float(np.median(values))
            if not np.isfinite(centre) or centre == 0:
                raise ValueError("relative flux must have a finite non-zero median")
            values = (values / centre - 1.0) * 1.0e6
        elif flux_unit == "ppm":
            values = values - np.median(values)
        else:
            raise ValueError("flux_unit must be 'relative' or 'ppm'")

        if sigma_clip is not None:
            clip = float(sigma_clip)
            if not np.isfinite(clip) or clip <= 0:
                raise ValueError("sigma_clip must be positive or None")
            centre = float(np.median(values))
            mad = float(np.median(np.abs(values - centre)))
            if mad > 0:
                keep = np.abs(values - centre) <= clip * 1.4826 * mad
                time = time[keep]
                values = values[keep]
        if time.size < 4:
            raise ValueError("sigma clipping retained fewer than four cadences")

        cadence_days = cadence / 86400.0
        collapsed = _collapse_long_gaps(
            time,
            cadence_days=cadence_days,
            threshold_days=long_gap_days,
        )
        start = float(collapsed[0])
        indices = np.rint((collapsed - start) / cadence_days).astype(int)
        residual = np.abs(collapsed - (start + indices * cadence_days))
        aligned = residual <= 0.2 * cadence_days
        indices = indices[aligned]
        values = values[aligned]
        if indices.size < 4:
            raise ValueError("fewer than four cadences align with the regular grid")

        size = int(indices[-1]) + 1
        sums = np.bincount(indices, weights=values, minlength=size)
        counts = np.bincount(indices, minlength=size)
        observed = counts > 0
        regular = np.zeros(size, dtype=float)
        regular[observed] = sums[observed] / counts[observed]
        regular[observed] -= np.mean(regular[observed])
        return cls(
            regular,
            observed,
            cadence,
            start,
            source=source,
            dilution=dilution,
        )

    @property
    def observing_window(self) -> ObservingWindow:
        """Return the exact regular mask used for spectral convolution."""

        return ObservingWindow(
            self.observed,
            self.cadence_seconds,
            label="real-tess",
        )

    @property
    def duration_days(self) -> float:
        """Return the scheduled baseline after long-gap collapsing."""

        return self.flux_ppm.size * self.cadence_seconds / 86400.0

    def to_power_spectrum(self) -> PowerSpectrum:
        """Return a one-sided ppm²/µHz FFT power-density spectrum."""

        transform = np.fft.rfft(self.flux_ppm)
        frequency = np.fft.rfftfreq(
            self.flux_ppm.size,
            self.cadence_seconds,
        )
        scale = (
            2.0
            * self.cadence_seconds
            / (self.flux_ppm.size * np.mean(self.observed))
            * 1.0e-6
        )
        power = scale * np.abs(transform) ** 2
        if self.flux_ppm.size % 2 == 0:
            power[-1] *= 0.5
        return PowerSpectrum(frequency[1:] * 1.0e6, power[1:])

    def save(self, path: str | Path) -> None:
        """Save a compact, reusable preparation product.

        Parameters
        ----------
        path
            Destination ``.npz`` file.
        """

        np.savez_compressed(
            Path(path),
            flux_ppm=self.flux_ppm,
            observed=self.observed,
            cadence_seconds=self.cadence_seconds,
            start_time_days=self.start_time_days,
            source=self.source,
            dilution=self.dilution,
        )

    @classmethod
    def load(cls, path: str | Path) -> PreparedTessLightCurve:
        """Load a product written by :meth:`save`.

        Parameters
        ----------
        path
            Existing ``.npz`` preparation product.

        Returns
        -------
        PreparedTessLightCurve
            Validated light curve reconstructed from disk.
        """

        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                data["flux_ppm"],
                data["observed"],
                float(data["cadence_seconds"]),
                float(data["start_time_days"]),
                source=str(data["source"]),
                dilution=(
                    float(data["dilution"])
                    if "dilution" in data
                    else 1.0
                ),
            )


@dataclass(frozen=True, slots=True)
class TessValidationRecovery:
    """Detector output paired with one literature-labelled TESS target.

    Attributes
    ----------
    target
        Target metadata and literature labels.
    probabilities
        Posterior probability for each complete spectral model.
    classification
        Maximum-posterior model label.
    detected
        Whether the oscillation probability meets the chosen threshold.
    bin_width
        Detector bin width in microhertz.
    duty_cycle
        Fraction of scheduled cadences present in the prepared light curve.
    duration_days
        Scheduled baseline after collapsing very long gaps.
    gap_count
        Number of contiguous missing-data intervals.
    maximum_gap_hours
        Duration of the longest retained gap.
    """

    target: TessValidationTarget
    probabilities: Mapping[str, float]
    classification: str
    detected: bool
    bin_width: float
    duty_cycle: float
    duration_days: float
    gap_count: int
    maximum_gap_hours: float


def evaluate_tess_target(
    target: TessValidationTarget,
    light_curve: PreparedTessLightCurve,
    detector: Detector,
    *,
    threshold: float = 0.45,
    rng: np.random.Generator | int | None = None,
    window_fft_workers: int = 1,
    window_row_batch_size: int = 8,
    **asteroscale_kwargs: Any,
) -> TessValidationRecovery:
    """Evaluate one prepared target with its exact observing window.

    Parameters
    ----------
    target
        Target metadata containing independent AsteroScale constraints.
    light_curve
        Prepared regular light curve and observing mask.
    detector
        Configured detector. Its observation model should contain the
        target's cadence and dilution.
    threshold
        Oscillation-probability threshold used for the binary detection flag.
    rng
        Random generator or seed passed to the detector.
    window_fft_workers
        Worker count used by the spectral-window FFTs.
    window_row_batch_size
        Number of full-resolution model rows transformed together.
    **asteroscale_kwargs
        Additional keywords passed through to :meth:`Detector.run` and then
        to AsteroScale inference.

    Returns
    -------
    TessValidationRecovery
        Detection probabilities, binary decision, and window diagnostics.
    """

    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    if not target.stellar_constraints:
        raise ValueError(
            f"TIC {target.tic_id} has no independent stellar constraints"
        )
    result = detector.run(
        light_curve.to_power_spectrum(),
        target.stellar_constraints,
        observing_window=light_curve.observing_window,
        window_fft_workers=window_fft_workers,
        window_row_batch_size=window_row_batch_size,
        rng=rng,
        **asteroscale_kwargs,
    )
    diagnostics = observing_window_diagnostics(light_curve.observing_window)
    probabilities = MappingProxyType(dict(result.probabilities))
    return TessValidationRecovery(
        target=target,
        probabilities=probabilities,
        classification=result.classification,
        detected=probabilities["oscillation"] >= threshold,
        bin_width=result.bin_width,
        duty_cycle=diagnostics.duty_cycle,
        duration_days=light_curve.duration_days,
        gap_count=diagnostics.gap_count,
        maximum_gap_hours=diagnostics.maximum_gap_hours,
    )


def summarize_tess_recoveries(
    recoveries: Sequence[TessValidationRecovery],
) -> dict[str, Any]:
    """Return JSON-safe real-data metrics without inventing negative truth.

    Parameters
    ----------
    recoveries
        Completed real-target evaluations.

    Returns
    -------
    dict
        Confirmed-detection TPR, reported-non-detection flag rate, and
        confirmed-detection TPR grouped by stellar regime.

    Notes
    -----
    The reported-non-detection flag rate is deliberately not labelled as a
    false-positive rate because absence of a published detection is not known
    negative truth.
    """

    items = tuple(recoveries)
    if not items:
        raise ValueError("at least one recovery is required")
    confirmed = tuple(
        item
        for item in items
        if item.target.reference_label == "confirmed_detection"
    )
    non_detections = tuple(
        item
        for item in items
        if item.target.reference_label == "reported_non_detection"
    )

    def fraction_detected(group: Sequence[TessValidationRecovery]) -> float:
        """Return the detected fraction or NaN for an empty subgroup."""

        return (
            float(np.mean([item.detected for item in group]))
            if group
            else float("nan")
        )

    return {
        "targets": len(items),
        "confirmed_detections": len(confirmed),
        "confirmed_detection_tpr": fraction_detected(confirmed),
        "reported_non_detections": len(non_detections),
        "reported_non_detection_flag_rate": fraction_detected(non_detections),
        "note": (
            "The reported-non-detection flag rate is not a false-positive "
            "rate because absence of a published detection is not negative truth."
        ),
        "by_regime": {
            regime: {
                "targets": len(group),
                "confirmed_detections": sum(
                    item.target.reference_label == "confirmed_detection"
                    for item in group
                ),
                "confirmed_detection_tpr": fraction_detected(
                    [
                        item
                        for item in group
                        if item.target.reference_label == "confirmed_detection"
                    ]
                ),
            }
            for regime in sorted({item.target.regime for item in items})
            for group in (
                tuple(item for item in items if item.target.regime == regime),
            )
        },
    }


def recovery_to_dict(recovery: TessValidationRecovery) -> dict[str, Any]:
    """Convert one recovery to a JSON-safe record.

    Parameters
    ----------
    recovery
        Completed target evaluation.

    Returns
    -------
    dict
        Target labels, probabilities, decision, and window diagnostics.
    """

    return {
        "name": recovery.target.name,
        "tic_id": recovery.target.tic_id,
        "reference_label": recovery.target.reference_label,
        "regime": recovery.target.regime,
        "reference": recovery.target.reference,
        "reference_numax": recovery.target.reference_numax,
        "reference_numax_error": recovery.target.reference_numax_error,
        "reference_dnu": recovery.target.reference_dnu,
        "reference_dnu_error": recovery.target.reference_dnu_error,
        "probabilities": dict(recovery.probabilities),
        "classification": recovery.classification,
        "detected": recovery.detected,
        "bin_width": recovery.bin_width,
        "duty_cycle": recovery.duty_cycle,
        "duration_days": recovery.duration_days,
        "gap_count": recovery.gap_count,
        "maximum_gap_hours": recovery.maximum_gap_hours,
    }
