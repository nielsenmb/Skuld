import importlib.util
import sys
from pathlib import Path

import numpy as np


EXAMPLES = Path(__file__).parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location(
    "focused_window_study",
    EXAMPLES / "focused_window_study.py",
)
FOCUSED = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FOCUSED)


def test_focused_campaign_pairs_all_window_profiles():
    cases = FOCUSED.build_focused_window_campaign(repeats=1, seed=9)
    assert len(cases) == 3 * 2 * 2 * 5
    pairs = {}
    for case in cases:
        pairs.setdefault(case.metadata["pair_id"], set()).add(
            case.metadata["window_profile"]
        )
        assert case.truth == "oscillation"
        assert case.metadata["amplitude_scale"] == 1.0
    assert all(
        profiles == set(FOCUSED.TESS_WINDOW_PROFILES)
        for profiles in pairs.values()
    )


def test_momentum_interval_campaign_pairs_rgb_profiles():
    cases = FOCUSED.build_momentum_interval_campaign(
        repeats=1,
        seed=4,
        intervals_days=(2.5, 4.0),
    )
    assert len(cases) == 2 * 2 * 3
    assert {case.metadata["stellar_regime"] for case in cases} == {
        "low_luminosity_rgb"
    }
    assert {
        case.metadata["gap_interval_days"]
        for case in cases
    } == {None, 2.5, 4.0}


def test_momentum_interval_campaign_can_match_gap_fraction():
    cases = FOCUSED.build_momentum_interval_campaign(
        repeats=1,
        seed=4,
        intervals_days=(2.5, 5.0),
        match_duty_cycle=True,
    )
    comparison = [
        case
        for case in cases
        if case.metadata["window_profile"] == "momentum-dumps"
    ]
    by_duration = {}
    for case in comparison:
        by_duration.setdefault(case.metadata["duration_days"], []).append(case)
    for duration_cases in by_duration.values():
        duty_cycles = [case.metadata["duty_cycle"] for case in duration_cases]
        assert np.ptp(duty_cycles) < 5e-4
    assert {
        case.metadata["gap_duration_minutes"] for case in comparison
    } == {30.0, 60.0}


def test_wilson_interval_contains_observed_fraction():
    lower, upper = FOCUSED._wilson_interval(8, 10)
    assert lower < 0.8 < upper
    assert np.isnan(FOCUSED._wilson_interval(0, 0)[0])


def test_focused_campaign_validates_profiles_and_repeats():
    for profiles in (("tess-like",), ("continuous", "mystery")):
        try:
            FOCUSED.build_focused_window_campaign(
                repeats=1,
                seed=1,
                window_profiles=profiles,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid profile set was accepted")
    try:
        FOCUSED.build_focused_window_campaign(repeats=0, seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("zero repeats were accepted")
