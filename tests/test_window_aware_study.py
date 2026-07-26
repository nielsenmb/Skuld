import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from asterodetect import Detector


EXAMPLES = Path(__file__).parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location(
    "window_aware_study",
    EXAMPLES / "window_aware_study.py",
)
STUDY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(STUDY)


def test_window_aware_campaign_pairs_classes_and_windows():
    cases = STUDY.build_window_aware_campaign(repeats=1, seed=7)
    assert len(cases) == 2 * 2 * 3 * 2
    assert {case.truth for case in cases} == {
        "noise",
        "granulation",
        "oscillation",
    }
    pairs = {}
    for case in cases:
        pairs.setdefault(case.metadata["pair_id"], set()).add(
            case.metadata["window_profile"]
        )
        assert case.metadata["momentum_dump_interval_days"] == 2.5
    assert all(
        profiles == {"continuous", "momentum-dumps"}
        for profiles in pairs.values()
    )


def test_reconstructed_window_matches_saved_metadata():
    case = next(
        case
        for case in STUDY.build_window_aware_campaign(repeats=1, seed=8)
        if case.metadata["window_profile"] == "momentum-dumps"
    )
    window = STUDY._window_for_case(case)
    assert window.label == "momentum-dumps"
    assert np.isclose(window.duty_cycle, case.metadata["duty_cycle"])


def test_window_aware_evaluation_reuses_each_gapped_spectrum(monkeypatch):
    cases = STUDY.build_window_aware_campaign(repeats=1, seed=9)[:2]
    original = Detector.run

    def fast_run(self, *args, **kwargs):
        """Use a small prior estimate in this structural study test."""

        self.draws = 2
        self.estimator = "prior"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Detector, "run", fast_run)
    result = STUDY.evaluate_window_aware_campaign(
        Detector(draws=2),
        cases,
        seed=10,
    )
    profiles = [
        recovery.case.metadata["evaluation_profile"]
        for recovery in result.recoveries
    ]
    assert profiles == list(STUDY.EVALUATION_PROFILES)
    assert not result.recoveries[1].result.spectral_window_applied
    assert result.recoveries[2].result.spectral_window_applied


def test_window_aware_campaign_validates_repeats():
    for repeats in (0, True):
        try:
            STUDY.build_window_aware_campaign(repeats=repeats, seed=1)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid repeats were accepted")
    for truths in ((), ("unknown",), ("noise", "noise")):
        try:
            STUDY.build_window_aware_campaign(
                repeats=1,
                seed=1,
                truths=truths,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid truths were accepted")


def test_saved_window_aware_reports_are_consistent():
    data = Path(__file__).parents[1] / "notebooks" / "data"
    study = json.loads((data / "window_aware_study.json").read_text())
    replay = json.loads(
        (data / "window_aware_rgb_replay.json").read_text()
    )
    assert study["draws"] == replay["draws"] == 256
    assert (
        study["profiles"]["gapped-window-aware"]["false_positives"]
        == study["profiles"]["gapped-current"]["false_positives"]
    )
    assert (
        replay["profiles"]["gapped-window-aware"]["detections"]
        == replay["profiles"]["continuous-current"]["detections"]
        == 31
    )
