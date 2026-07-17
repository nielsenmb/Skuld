import numpy as np
import pytest

from asterodetect import (
    DEFAULT_TOTAL_MODE_VISIBILITY,
    ObservationModel,
    cadence_amplitude_response,
    envelope_integrated_power,
)


def test_solar_tess_amplitude_recovers_reference_envelope_height():
    dnu = 135.1
    fwhm = 598.0
    power = envelope_integrated_power(2.1, dnu, fwhm)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    height = power / (np.sqrt(2 * np.pi) * sigma)
    expected = DEFAULT_TOTAL_MODE_VISIBILITY * 2.1**2 / dnu
    assert height == pytest.approx(expected)
    assert height == pytest.approx(0.1, rel=0.01)


def test_cadence_response_matches_nyquist_equation():
    integration_time = 120.0
    nyquist = 1.0e6 / (2 * integration_time)
    assert cadence_amplitude_response(nyquist, integration_time) == pytest.approx(
        2 / np.pi
    )


def test_observation_terms_act_on_power_quadratically():
    intrinsic = ObservationModel().envelope_power(2.0, 100.0, 400.0)
    diluted = ObservationModel(dilution=0.5).envelope_power(
        2.0, 100.0, 400.0
    )
    assert diluted == pytest.approx(intrinsic / 4)

    cadence = 120.0
    nyquist = 1.0e6 / (2 * cadence)
    attenuated = ObservationModel(integration_time_seconds=cadence).envelope_power(
        2.0, 100.0, 400.0, numax=nyquist
    )
    assert attenuated == pytest.approx(intrinsic * (2 / np.pi) ** 2)


def test_numax_is_required_when_cadence_attenuation_is_enabled():
    with pytest.raises(ValueError, match="numax is required"):
        ObservationModel(integration_time_seconds=120).envelope_power(
            2.0, 100.0, 400.0
        )


def test_envelope_conversion_broadcasts_joint_samples():
    result = envelope_integrated_power(
        [1.0, 2.0],
        [100.0, 100.0],
        400.0,
    )
    assert result.shape == (2,)
    assert result[1] == pytest.approx(4 * result[0])
