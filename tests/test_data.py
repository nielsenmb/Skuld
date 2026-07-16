import numpy as np
import pytest

from asterodetect import PowerSpectrum


def test_power_spectrum_validates_and_broadcasts_shape():
    spectrum = PowerSpectrum([1, 2, 3], [4, 5, 6], bins_averaged=2)
    np.testing.assert_array_equal(spectrum.bins_averaged, [2, 2, 2])


@pytest.mark.parametrize(
    "frequency,power",
    [([1, 1], [1, 2]), ([2, 1], [1, 2]), ([1, 2], [1, 0]), ([1], [np.nan])],
)
def test_power_spectrum_rejects_invalid_values(frequency, power):
    with pytest.raises(ValueError):
        PowerSpectrum(frequency, power)


def test_regular_binning_updates_gamma_shape():
    spectrum = PowerSpectrum(np.arange(8.0), np.arange(8.0) + 1, 1)
    binned = spectrum.bin_regular(2)
    np.testing.assert_allclose(binned.frequency, [0.5, 2.5, 4.5, 6.5])
    np.testing.assert_allclose(binned.power, [1.5, 3.5, 5.5, 7.5])
    np.testing.assert_allclose(binned.bins_averaged, 2)

