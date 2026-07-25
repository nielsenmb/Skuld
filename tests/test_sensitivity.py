import numpy as np
import pytest

from asterodetect import (
    AsteroScaleSamples,
    InjectionCase,
    NuisancePrior,
    PowerSpectrum,
    run_sensitivity_study,
)
from asterodetect.asteroscale import ASTERO_SCALE_PARAMETERS


def _samples(draws=16):
    values = {name: np.ones(draws) for name in ASTERO_SCALE_PARAMETERS}
    values.update(
        numax=np.full(draws, 100.0),
        dnu=np.full(draws, 10.0),
        FWHM_env=np.full(draws, 50.0),
        A_env=np.full(draws, 2.0),
        A_gran=np.full(draws, 8.0),
        b_gran_low=np.full(draws, 30.0),
        b_gran_high=np.full(draws, 90.0),
    )
    return AsteroScaleSamples(values)


def _case():
    frequency = np.arange(0.5, 200.0, 0.5)
    power = np.random.default_rng(8).exponential(0.1, frequency.size)
    return InjectionCase(
        "fixed-noise", "noise", PowerSpectrum(frequency, power), _samples()
    )


def test_sensitivity_study_is_paired_and_summarizes_diagnostics():
    study = run_sensitivity_study(
        [_case()],
        draw_counts=[8, 16],
        dnu_scales=[0.5, 1.0],
        repeats=2,
        seed=9,
        nuisance_prior=NuisancePrior(
            white_noise_log_scatter=0,
            envelope_log_scatter=0,
            granulation_log_scatter=0,
            overdispersion_log_scatter=0,
        ),
    )
    assert len(study.runs) == 8
    assert {(run.draws, run.dnu_scale) for run in study.runs} == {
        (8, 0.5),
        (8, 1.0),
        (16, 0.5),
        (16, 1.0),
    }
    assert len({run.case_name for run in study.runs}) == 1
    assert len(study.summaries()) == 4
    for summary in study.summaries():
        assert summary.evaluations == 2
        assert summary.oscillation_probability_std >= 0
        assert 0 <= summary.minimum_median_ess_fraction <= 1
        assert summary.maximum_median_log_evidence_standard_error >= 0


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("draw_counts", [], ValueError),
        ("draw_counts", [0], ValueError),
        ("draw_counts", [True], TypeError),
        ("dnu_scales", [], ValueError),
        ("dnu_scales", [0], ValueError),
    ],
)
def test_sensitivity_study_validates_axes(keyword, value, error):
    arguments = {
        "draw_counts": [8],
        "dnu_scales": [1.0],
        keyword: value,
    }
    with pytest.raises(error):
        run_sensitivity_study([_case()], **arguments)


def test_sensitivity_study_compares_estimators_on_the_same_case():
    study = run_sensitivity_study(
        [_case()],
        draw_counts=[16],
        dnu_scales=[1.0],
        estimators=["prior", "adaptive"],
        pilot_draws=16,
        repeats=1,
        seed=3,
    )
    assert {run.estimator for run in study.runs} == {"prior", "adaptive"}
    assert {summary.estimator for summary in study.summaries()} == {
        "prior",
        "adaptive",
    }
    comparison = study.estimator_comparisons()
    assert len(comparison) == 1
    assert comparison[0].evaluations_per_estimator == 1
    assert comparison[0].mean_absolute_oscillation_probability_difference >= 0
    assert comparison[0].minimum_median_ess_fraction_ratio >= 0
    assert comparison[0].maximum_median_log_evidence_standard_error_ratio >= 0


def test_estimator_comparisons_require_both_estimators():
    study = run_sensitivity_study(
        [_case()],
        draw_counts=[8],
        dnu_scales=[1.0],
        estimators=["prior"],
        repeats=1,
        seed=3,
    )
    assert study.estimator_comparisons() == ()
