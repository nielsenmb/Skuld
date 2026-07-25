import numpy as np

from asterodetect import (
    AsteroScaleSamples, Detector, InjectionCase, NuisancePrior,
    PowerSpectrum, build_injection_grid, evaluate_injections,
)
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
