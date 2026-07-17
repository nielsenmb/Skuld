import numpy as np
import pytest

from asterodetect import AsteroScaleSamples, ObservationModel
from asterodetect.asteroscale import ASTERO_SCALE_PARAMETERS


def test_asteroscale_samples_preserve_joint_rows_when_resampled():
    values = {
        name: np.arange(10, dtype=float) + offset
        for offset, name in enumerate(ASTERO_SCALE_PARAMETERS)
    }
    samples = AsteroScaleSamples(values)
    draws = samples.draw(100, rng=42)
    reference_offset = draws[ASTERO_SCALE_PARAMETERS[0]]
    for offset, name in enumerate(ASTERO_SCALE_PARAMETERS):
        np.testing.assert_array_equal(draws[name] - offset, reference_offset)


def test_asteroscale_infer_uses_bayesian_mode_by_default():
    class FakeSolver:
        def __init__(self):
            self.call = None

        def solve(self, given, want, **kwargs):
            self.call = (given, want, kwargs)
            return {name: np.arange(3, dtype=float) + 1 for name in want}

    solver = FakeSolver()
    result = AsteroScaleSamples.infer({"Teff": (5777, 50)}, solver=solver)
    assert len(result) == 3
    assert solver.call[2]["input_mode"] == "likelihood"
    assert solver.call[2]["bandpass"] == "TESS"


def test_suggested_bin_width_uses_dnu_and_envelope_resolution():
    values = {
        name: np.full(10, 100.0)
        for name in ASTERO_SCALE_PARAMETERS
    }
    values["dnu"] = np.full(10, 20.0)
    values["FWHM_env"] = np.full(10, 75.0)
    samples = AsteroScaleSamples(values)
    assert samples.suggested_bin_width() == 15.0


def test_envelope_parameters_preserve_rows_and_apply_observation_model():
    values = {
        name: np.arange(1.0, 4.0)
        for name in ASTERO_SCALE_PARAMETERS
    }
    values["numax"] = np.array([1000.0, 2000.0, 3000.0])
    values["dnu"] = np.full(3, 100.0)
    values["FWHM_env"] = np.full(3, 400.0)
    values["A_env"] = np.array([1.0, 2.0, 3.0])
    samples = AsteroScaleSamples(values)
    intrinsic = samples.envelope_parameters()
    diluted = samples.envelope_parameters(ObservationModel(dilution=0.5))

    np.testing.assert_array_equal(intrinsic["numax"], values["numax"])
    assert diluted["integrated_power"][1] == pytest.approx(
        intrinsic["integrated_power"][1] / 4
    )
    assert not diluted["integrated_power"].flags.writeable
