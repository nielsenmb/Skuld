from pathlib import Path

import numpy as np
import pytest

from asterodetect import (
    PreparedTessLightCurve,
    TessValidationRecovery,
    TessValidationTarget,
    load_tess_target_manifest,
    summarize_tess_recoveries,
)


def _target(
    tic_id=1,
    *,
    label="confirmed_detection",
    regime="main_sequence",
):
    return TessValidationTarget(
        name=f"TIC {tic_id}",
        tic_id=tic_id,
        reference_label=label,
        regime=regime,
        reference="Test+2026",
        stellar_constraints={"Teff": [5772.0, 80.0], "R": [1.0, 0.05]},
        reference_numax=3090.0,
        reference_dnu=135.1,
        preferred_cadence_seconds=120.0,
    )


def _recovery(target, probability):
    return TessValidationRecovery(
        target=target,
        probabilities={
            "noise": (1 - probability) / 2,
            "granulation": (1 - probability) / 2,
            "oscillation": probability,
        },
        classification="oscillation" if probability > 0.5 else "granulation",
        detected=probability >= 0.45,
        bin_width=10.0,
        duty_cycle=0.95,
        duration_days=27.4,
        gap_count=2,
        maximum_gap_hours=1.0,
    )


def test_target_rejects_seismic_information_as_detector_constraint():
    with pytest.raises(ValueError, match="cannot be detector constraints"):
        TessValidationTarget(
            name="leaky",
            tic_id=1,
            reference_label="confirmed_detection",
            regime="dwarf",
            reference="Test+2026",
            stellar_constraints={"Teff": 5772.0, "numax": 3090.0},
        )


def test_irregular_light_curve_preserves_gaps_and_physical_psd_normalization():
    rng = np.random.default_rng(4)
    cadence = 120.0
    samples = 1024
    time = np.arange(samples) * cadence / 86400.0
    flux_ppm = rng.normal(0.0, 20.0, samples)
    observed = np.ones(samples, dtype=bool)
    observed[100:120] = False
    prepared = PreparedTessLightCurve.from_irregular(
        time[observed],
        flux_ppm[observed],
        cadence_seconds=cadence,
        flux_unit="ppm",
        sigma_clip=None,
    )
    assert prepared.observed.size == samples
    assert np.all(~prepared.observed[100:120])
    spectrum = prepared.to_power_spectrum()
    spacing = spectrum.frequency[1] - spectrum.frequency[0]
    observed_variance = np.mean(prepared.flux_ppm[prepared.observed] ** 2)
    assert np.isclose(
        np.sum(spectrum.power) * spacing,
        observed_variance,
        rtol=2e-3,
    )


def test_long_intercycle_gap_is_collapsed_without_filling_short_gaps():
    cadence = 120.0
    first = np.arange(100) * cadence / 86400.0
    second = 400.0 + np.arange(100) * cadence / 86400.0
    prepared = PreparedTessLightCurve.from_irregular(
        np.concatenate((first, second)),
        np.ones(200),
        cadence_seconds=cadence,
        sigma_clip=None,
        long_gap_days=50.0,
    )
    assert prepared.observed.size == 200
    assert np.all(prepared.observed)


def test_prepared_light_curve_round_trips(tmp_path):
    prepared = PreparedTessLightCurve(
        [1.0, 0.0, -1.0, 0.5, -0.5],
        np.asarray([True, False, True, True, True]),
        120.0,
        42.0,
        source="test",
        dilution=0.8,
    )
    path = tmp_path / "target.npz"
    prepared.save(path)
    loaded = PreparedTessLightCurve.load(path)
    np.testing.assert_array_equal(loaded.flux_ppm, prepared.flux_ppm)
    np.testing.assert_array_equal(loaded.observed, prepared.observed)
    assert loaded.source == "test"
    assert loaded.dilution == 0.8


def test_real_summary_does_not_call_non_detections_false_positives():
    confirmed = _target(1)
    non_detection = _target(2, label="reported_non_detection")
    summary = summarize_tess_recoveries(
        [_recovery(confirmed, 0.8), _recovery(non_detection, 0.7)]
    )
    assert summary["confirmed_detection_tpr"] == 1.0
    assert summary["reported_non_detection_flag_rate"] == 1.0
    assert "false-positive" in summary["note"]


def test_shipped_manifest_is_balanced_and_parseable():
    manifest = Path(__file__).parents[1] / "examples" / "tess_targets.csv"
    targets = load_tess_target_manifest(manifest)
    assert len(targets) == 24
    assert sum(target.regime == "red_giant" for target in targets) == 12
    assert sum(target.regime != "red_giant" for target in targets) == 12
    assert all(target.reference_label == "confirmed_detection" for target in targets)
