import numpy as np

from asterodetect import (
    AsteroScaleSamples,
    AstrophysicalInjectionFactory,
    build_injection_grid,
    lorentzian_mode_comb,
    regular_frequency_grid,
)
from asterodetect.asteroscale import ASTERO_SCALE_PARAMETERS


def _samples():
    values = {name: np.ones(12) for name in ASTERO_SCALE_PARAMETERS}
    values.update(
        numax=np.full(12, 100.0),
        dnu=np.full(12, 10.0),
        FWHM_env=np.full(12, 50.0),
        A_env=np.full(12, 3.0),
        A_gran=np.full(12, 8.0),
        b_gran_low=np.full(12, 30.0),
        b_gran_high=np.full(12, 90.0),
    )
    return AsteroScaleSamples(values)


def test_regular_frequency_grid_has_fourier_resolution_and_nyquist():
    frequency = regular_frequency_grid(2.0, 120.0)
    assert np.isclose(frequency[0], 1e6 / (2 * 86400))
    assert frequency[-1] <= 1e6 / (2 * 120)


def test_lorentzian_comb_is_positive_and_peaked_near_numax():
    frequency = np.arange(1.0, 201.0, 0.05)
    comb = lorentzian_mode_comb(
        frequency,
        numax=100.0,
        dnu=10.0,
        fwhm_envelope=50.0,
        radial_mode_rms_amplitude=3.0,
        linewidth=0.5,
    )
    assert np.all(comb >= 0)
    central = comb[(frequency > 75) & (frequency < 125)].max()
    wings = comb[(frequency < 30) | (frequency > 170)].max()
    assert central > wings


def test_factory_generates_all_three_classes_reproducibly():
    factory = AstrophysicalInjectionFactory(
        _samples(), duration_days=2.0, cadence_seconds=600.0
    )
    axes = {
        "truth": ["noise", "granulation", "oscillation"],
        "amplitude_scale": [0.3],
    }
    first = build_injection_grid(axes, factory, seed=7)
    second = build_injection_grid(axes, factory, seed=7)
    assert [case.truth for case in first] == [
        "noise",
        "granulation",
        "oscillation",
    ]
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.spectrum.power, right.spectrum.power)
        assert left.metadata["duration_days"] == 2.0
    assert not np.array_equal(first[0].spectrum.power, first[2].spectrum.power)


def test_zero_amplitude_oscillation_case_contains_granulation_only():
    factory = AstrophysicalInjectionFactory(
        _samples(), duration_days=2.0, cadence_seconds=600.0
    )
    case = factory(
        "suppressed",
        {"truth": "oscillation", "amplitude_scale": 0.0},
        np.random.default_rng(2),
    )
    assert case.truth == "oscillation"
    assert case.metadata["amplitude_scale"] == 0.0
