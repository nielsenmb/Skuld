import numpy as np

from asterodetect import (
    AsteroScaleSamples, Detector, InjectionCase, NuisancePrior,
    PowerSpectrum, evaluate_injections,
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
