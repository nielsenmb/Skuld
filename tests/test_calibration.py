import numpy as np

from asterodetect import (
    AsteroScaleSamples, Detector, InjectionCase, NuisancePrior,
    PowerSpectrum, Recovery, build_detection_study, build_injection_grid, evaluate_injections,
    probability_reliability, summarize_recoveries, threshold_curve,
)
from asterodetect.detector import DetectionResult
from asterodetect.marginal import MarginalEvaluation
from asterodetect.asteroscale import ASTERO_SCALE_PARAMETERS


def _samples():
    values = {name: np.ones(8) for name in ASTERO_SCALE_PARAMETERS}
    values.update(
        numax=np.full(8, 100.0), dnu=np.full(8, 10.0),
        FWHM_env=np.full(8, 50.0), A_env=np.full(8, 3.0),
        A_gran=np.full(8, 8.0), b_gran_low=np.full(8, 30.0),
        b_gran_high=np.full(8, 90.0),
    )
    return AsteroScaleSamples(values)


def test_injection_framework_records_confusion_and_brier_score():
    frequency = np.arange(0.5, 200.0, 0.5)
    spectrum = PowerSpectrum(frequency, np.full_like(frequency, 0.1))
    cases = [
        InjectionCase("n1", "noise", spectrum, _samples()),
        InjectionCase("n2", "noise", spectrum, _samples()),
    ]
    detector = Detector(
        draws=16,
        nuisance_prior=NuisancePrior(
            white_noise_log_scatter=0, envelope_log_scatter=0,
            granulation_log_scatter=0, overdispersion_log_scatter=0,
        ),
    )
    result = evaluate_injections(detector, cases, seed=4)
    assert result.confusion_matrix.shape == (3, 3)
    assert result.confusion_matrix.sum() == 2
    assert 0 <= result.accuracy <= 1
    assert 0 <= result.multiclass_brier_score <= 2


def test_build_grid_is_reproducible_and_preserves_coordinates():
    frequency = np.arange(0.5, 200.0, 0.5)

    def factory(name, parameters, rng):
        power = rng.exponential(parameters["white_noise"], frequency.size)
        return InjectionCase(
            name,
            "noise",
            PowerSpectrum(frequency, power),
            _samples(),
            metadata={"duration_days": 27.4},
        )

    axes = {"white_noise": [0.1, 1.0], "dnu_scale": [0.5, 1.0]}
    first = build_injection_grid(axes, factory, repeats=2, seed=42)
    second = build_injection_grid(axes, factory, repeats=2, seed=42)
    assert len(first) == 8
    assert np.array_equal(first[3].spectrum.power, second[3].spectrum.power)
    assert first[0].metadata["duration_days"] == 27.4
    assert first[-1].metadata["white_noise"] == 1.0
    assert first[-1].metadata["dnu_scale"] == 1.0
    assert first[-1].metadata["repeat"] == 1


def test_calibration_can_be_grouped_and_subset_by_metadata():
    frequency = np.arange(0.5, 200.0, 0.5)
    spectrum = PowerSpectrum(frequency, np.full_like(frequency, 0.1))
    cases = [
        InjectionCase(
            f"n{index}",
            "noise",
            spectrum,
            _samples(),
            metadata={"cadence": cadence, "amplitude_scale": 0.0},
        )
        for index, cadence in enumerate((20, 20, 120))
    ]
    detector = Detector(
        draws=16,
        nuisance_prior=NuisancePrior(
            white_noise_log_scatter=0, envelope_log_scatter=0,
            granulation_log_scatter=0, overdispersion_log_scatter=0,
        ),
    )
    result = evaluate_injections(detector, cases, seed=4)
    grouped = result.group_by("cadence")
    assert len(grouped[20].recoveries) == 2
    assert len(grouped[120].recoveries) == 1
    assert len(result.subset(cadence=20, amplitude_scale=0).recoveries) == 2


def _recovery(name, truth, oscillation_probability):
    remaining = 1.0 - oscillation_probability
    evaluation = MarginalEvaluation(
        log_evidences={"noise": 0.0, "granulation": 0.0, "oscillation": 0.0},
        responsibilities={
            "noise": remaining / 2,
            "granulation": remaining / 2,
            "oscillation": oscillation_probability,
        },
        diagnostics={},
    )
    frequency = np.arange(1.0, 5.0)
    spectrum = PowerSpectrum(frequency, np.ones_like(frequency))
    result = DetectionResult(
        evaluation=evaluation,
        binned_spectrum=spectrum,
        bin_width=1.0,
        samples=_samples(),
        nuisance_draws={},
    )
    return Recovery(InjectionCase(name, truth, spectrum, _samples()), result)


def test_binary_detection_metrics_distinguish_completeness_and_false_positives():
    calibration = summarize_recoveries(
        [
            _recovery("o1", "oscillation", 0.9),
            _recovery("o2", "oscillation", 0.4),
            _recovery("g1", "granulation", 0.8),
            _recovery("n1", "noise", 0.1),
        ]
    )
    metrics = calibration.detection_metrics(threshold=0.5, probability_bins=2)
    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 1
    assert metrics.completeness == 0.5
    assert metrics.false_positive_rate == 0.5
    assert metrics.precision == 0.5
    assert len(metrics.reliability) == 2


def test_probability_reliability_includes_probability_one_in_last_bin():
    bins = probability_reliability([0.0, 0.2, 0.8, 1.0], [False, False, True, True], bins=2)
    assert [item.count for item in bins] == [2, 2]
    assert bins[-1].observed_frequency == 1


def test_threshold_curve_returns_requested_operating_points():
    calibration = summarize_recoveries(
        [_recovery("o", "oscillation", 0.7), _recovery("n", "noise", 0.3)]
    )
    curve = threshold_curve(calibration, [0.25, 0.5, 0.75])
    assert [item.threshold for item in curve] == [0.25, 0.5, 0.75]
    assert curve[1].completeness == 1
    assert curve[1].false_positive_rate == 0


def test_detection_study_does_not_duplicate_negative_classes_by_amplitude():
    frequency = np.arange(0.5, 10.0, 0.5)

    def factory(name, parameters, rng):
        truth = parameters["truth"]
        return InjectionCase(
            name,
            truth,
            PowerSpectrum(frequency, rng.exponential(1.0, frequency.size)),
            _samples(),
            metadata=dict(parameters),
        )

    cases = build_detection_study(
        {"white_noise": [0.1, 1.0]},
        factory,
        oscillation_amplitudes=[0.1, 1.0],
        repeats=2,
        seed=1,
    )
    assert len(cases) == 2 * 4 * 2
    assert sum(case.truth == "noise" for case in cases) == 4
    assert sum(case.truth == "granulation" for case in cases) == 4
    assert sum(case.truth == "oscillation" for case in cases) == 8
    assert {
        case.metadata["amplitude_scale"]
        for case in cases
        if case.truth == "oscillation"
    } == {0.1, 1.0}
