"""Prepare and run a resumable real-TESS validation campaign.

Light-curve downloads are intentionally optional. Install the validation
dependencies with ``uv sync --extra tess-validation`` before using ``prepare``.
The core ``run`` and ``summarize`` stages operate only on cached local files.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from mimir import (
    download_lightcurves,
    lightcurve_to_timeseries,
    search_lightcurves,
)

from asterodetect import (
    Detector,
    ObservationModel,
    PreparedTessLightCurve,
    TessValidationTarget,
    evaluate_tess_target,
    load_tess_target_manifest,
    recovery_to_dict,
    summarize_tess_recoveries,
)


def _selected_targets(
    manifest: Path,
    *,
    limit: int | None,
    tic_ids: Iterable[int] = (),
) -> tuple[TessValidationTarget, ...]:
    """Select a stable manifest subset."""

    targets = load_tess_target_manifest(manifest)
    requested = set(tic_ids)
    if requested:
        targets = tuple(target for target in targets if target.tic_id in requested)
        missing = requested - {target.tic_id for target in targets}
        if missing:
            raise ValueError(f"TIC IDs not present in manifest: {sorted(missing)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        targets = targets[:limit]
    return targets


def _value(row: Any, *names: str) -> float | None:
    """Read the first finite numeric field from an Astropy table row."""

    for name in names:
        if name not in row.colnames:
            continue
        candidate = row[name]
        if np.ma.is_masked(candidate):
            continue
        try:
            result = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(result):
            return result
    return None


def _tic_constraints(tic_id: int) -> dict[str, Any]:
    """Query independent TIC stellar properties for AsteroScale."""

    try:
        from astroquery.mast import Catalogs
    except ImportError as error:
        raise RuntimeError(
            "prepare requires the 'tess-validation' optional dependencies"
        ) from error
    table = Catalogs.query_criteria(catalog="Tic", ID=tic_id)
    if len(table) != 1:
        raise RuntimeError(f"TIC query returned {len(table)} rows for {tic_id}")
    row = table[0]
    constraints: dict[str, Any] = {}
    mappings = {
        "Teff": (("Teff", "teff"), ("e_Teff", "e_teff")),
        "R": (("rad", "Rad", "radius"), ("e_rad", "e_Rad", "e_radius")),
        "FeH": (("MH", "FeH", "feh"), ("e_MH", "e_FeH", "e_feh")),
    }
    for output, (value_names, error_names) in mappings.items():
        centre = _value(row, *value_names)
        uncertainty = _value(row, *error_names)
        if centre is None:
            continue
        constraints[output] = (
            [centre, uncertainty]
            if uncertainty is not None and uncertainty > 0
            else centre
        )
    if "Teff" not in constraints or "R" not in constraints:
        raise RuntimeError(
            f"TIC {tic_id} lacks the independent Teff/R constraints required "
            "for this validation factory"
        )
    return constraints


def _search_and_download(target: TessValidationTarget, download_dir: Path):
    """Download the preferred available TESS light-curve products."""

    search = search_lightcurves(f"TIC {target.tic_id}", mission="TESS")
    if target.data_author and target.data_author.lower() != "auto":
        authors = np.asarray(search.table["author"]).astype(str)
        search = search[authors == target.data_author]
    if target.sectors:
        sectors = np.asarray(search.table["sequence_number"], dtype=int)
        search = search[np.isin(sectors, target.sectors)]
    if len(search) == 0:
        raise RuntimeError(f"no matching light curves found for TIC {target.tic_id}")

    exposure_column = search.table["exptime"]
    exposure_quantity = getattr(exposure_column, "quantity", exposure_column)
    exposure = np.asarray(
        (
            exposure_quantity.to_value("s")
            if hasattr(exposure_quantity, "to_value")
            else exposure_quantity
        ),
        dtype=float,
    )
    preferred = target.preferred_cadence_seconds
    selected_exposure = (
        np.min(exposure)
        if preferred is None
        else exposure[np.argmin(np.abs(exposure - preferred))]
    )
    search = search[np.isclose(exposure, selected_exposure)]
    if target.data_author.lower() == "auto":
        authors = np.asarray(search.table["author"]).astype(str)
        priorities = (
            "SPOC",
            "TESS-SPOC",
            "TASOC",
            "QLP",
            "GSFC-ELEANOR-LITE",
            "TGLC",
        )
        available = set(authors)
        selected_author = next(
            (author for author in priorities if author in available),
            min(available),
        )
        search = search[authors == selected_author]
    collection = download_lightcurves(search, download_dir=download_dir)
    return collection, float(selected_exposure)


def prepare_target(
    target: TessValidationTarget,
    cache_dir: Path,
    *,
    download_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Download and cache one regularized light curve plus its constraints."""

    output = cache_dir / f"tic-{target.tic_id}.npz"
    constraints_path = cache_dir / f"tic-{target.tic_id}.constraints.json"
    if output.exists() and constraints_path.exists() and not overwrite:
        return output

    collection, cadence = _search_and_download(target, download_dir)
    times = []
    fluxes = []
    sources = []
    dilution_values = []
    dilution_weights = []
    for light_curve in collection:
        quality = np.asarray(
            getattr(light_curve, "quality", np.zeros(len(light_curve))),
            dtype=int,
        )
        series = lightcurve_to_timeseries(
            light_curve[quality == 0],
            ppm=True,
        )
        if series.n_samples < 4:
            continue
        times.append(series.time)
        fluxes.append(series.flux)
        sources.append(
            f"{light_curve.meta.get('AUTHOR', 'unknown')}:"
            f"sector-{light_curve.meta.get('SECTOR', 'unknown')}"
        )
        crowding = light_curve.meta.get("CROWDSAP")
        if (
            crowding is not None
            and np.isfinite(float(crowding))
            and 0 < float(crowding) <= 1
        ):
            dilution_values.append(float(crowding))
            dilution_weights.append(series.n_samples)
    if not times:
        raise RuntimeError(f"all products were empty for TIC {target.tic_id}")
    prepared = PreparedTessLightCurve.from_irregular(
        np.concatenate(times),
        np.concatenate(fluxes),
        cadence_seconds=cadence,
        flux_unit="ppm",
        sigma_clip=5.0,
        long_gap_days=50.0,
        source=";".join(sources),
        dilution=(
            float(np.average(dilution_values, weights=dilution_weights))
            if dilution_values
            else 1.0
        ),
    )
    constraints = _tic_constraints(target.tic_id)
    constraints.update(target.stellar_constraints)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared.save(output)
    constraints_path.write_text(
        json.dumps(constraints, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _target_with_cached_constraints(
    target: TessValidationTarget,
    cache_dir: Path,
) -> TessValidationTarget:
    """Attach the independent constraints saved during preparation."""

    path = cache_dir / f"tic-{target.tic_id}.constraints.json"
    constraints = json.loads(path.read_text(encoding="utf-8"))
    return replace(target, stellar_constraints=constraints)


def run_target(
    target: TessValidationTarget,
    *,
    cache_dir: Path,
    output_dir: Path,
    detector: Detector,
    threshold: float,
    seed: int,
    fft_workers: int,
    window_row_batch_size: int,
    overwrite: bool = False,
) -> Path:
    """Run one target and atomically save its result."""

    output = output_dir / f"tic-{target.tic_id}.json"
    if output.exists() and not overwrite:
        return output
    target = _target_with_cached_constraints(target, cache_dir)
    prepared = PreparedTessLightCurve.load(
        cache_dir / f"tic-{target.tic_id}.npz"
    )
    observation = replace(
        detector.observation,
        integration_time_seconds=prepared.cadence_seconds,
        dilution=prepared.dilution,
    )
    target_detector = Detector(
        draws=detector.draws,
        observation=observation,
        nuisance_prior=detector.nuisance_prior,
        model_probabilities=detector.model_probabilities,
        dnu_scale=detector.dnu_scale,
        minimum_envelope_bins=detector.minimum_envelope_bins,
        estimator=detector.estimator,
        pilot_draws=detector.pilot_draws,
        defensive_fraction=detector.defensive_fraction,
        pilot_ess_fraction=detector.pilot_ess_fraction,
        proposal_degrees_of_freedom=detector.proposal_degrees_of_freedom,
        stellar_draws_per_nuisance=detector.stellar_draws_per_nuisance,
    )
    recovery = evaluate_tess_target(
        target,
        prepared,
        target_detector,
        threshold=threshold,
        rng=seed,
        window_fft_workers=fft_workers,
        window_row_batch_size=window_row_batch_size,
    )
    record = recovery_to_dict(recovery)
    record["threshold"] = threshold
    record["draws"] = detector.draws
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def _recovery_from_record(
    target: TessValidationTarget,
    record: dict[str, Any],
):
    """Rebuild the compact recovery fields needed by the summarizer."""

    from asterodetect import TessValidationRecovery

    return TessValidationRecovery(
        target=target,
        probabilities=record["probabilities"],
        classification=record["classification"],
        detected=record["detected"],
        bin_width=record["bin_width"],
        duty_cycle=record["duty_cycle"],
        duration_days=record["duration_days"],
        gap_count=record["gap_count"],
        maximum_gap_hours=record["maximum_gap_hours"],
    )


def summarize_results(
    targets: Sequence[TessValidationTarget],
    output_dir: Path,
) -> dict[str, Any]:
    """Summarize all completed targets and list missing ones."""

    completed = []
    missing = []
    for target in targets:
        path = output_dir / f"tic-{target.tic_id}.json"
        if not path.exists():
            missing.append(target.tic_id)
            continue
        completed.append(
            _recovery_from_record(
                target,
                json.loads(path.read_text(encoding="utf-8")),
            )
        )
    summary = (
        summarize_tess_recoveries(completed)
        if completed
        else {"targets": 0, "note": "No completed targets."}
    )
    summary["missing_tic_ids"] = missing
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("tess_targets.csv"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tic", type=int, action="append", default=[])
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--cache-dir", type=Path, default=Path("tess-cache"))
    prepare.add_argument(
        "--download-dir",
        type=Path,
        default=Path("tess-downloads"),
    )
    prepare.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--cache-dir", type=Path, default=Path("tess-cache"))
    run.add_argument("--output-dir", type=Path, default=Path("tess-results"))
    run.add_argument("--draws", type=int, default=256)
    run.add_argument("--threshold", type=float, default=0.45)
    run.add_argument("--seed", type=int, default=90210)
    run.add_argument("--fft-workers", type=int, default=-1)
    run.add_argument("--window-row-batch-size", type=int, default=8)
    run.add_argument("--overwrite", action="store_true")

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tess-results"),
    )
    summarize.add_argument("--output", type=Path)
    return parser


def main() -> None:
    """Run the selected validation-factory stage."""

    args = build_parser().parse_args()
    targets = _selected_targets(
        args.manifest,
        limit=args.limit,
        tic_ids=args.tic,
    )
    if args.command == "prepare":
        for target in targets:
            print(f"Preparing TIC {target.tic_id} ({target.name})")
            prepare_target(
                target,
                args.cache_dir,
                download_dir=args.download_dir,
                overwrite=args.overwrite,
            )
        return
    if args.command == "run":
        detector = Detector(
            draws=args.draws,
            estimator="adaptive",
            observation=ObservationModel(),
        )
        seeds = np.random.SeedSequence(args.seed).spawn(len(targets))
        for target, seed in zip(targets, seeds, strict=True):
            print(f"Running TIC {target.tic_id} ({target.name})")
            run_target(
                target,
                cache_dir=args.cache_dir,
                output_dir=args.output_dir,
                detector=detector,
                threshold=args.threshold,
                seed=int(seed.generate_state(1)[0]),
                fft_workers=args.fft_workers,
                window_row_batch_size=args.window_row_batch_size,
                overwrite=args.overwrite,
            )
        return

    summary = summarize_results(targets, args.output_dir)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
