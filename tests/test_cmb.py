import numpy as np
import pytest

from minkasi.needlet.cmb import cmb_fourier_weight, lensed_cmb_cl

PIX = np.deg2rad(3 / 3600)


def test_weight_is_one_without_cmb():
    """Zero scale, or no spectrum, leaves the white noise model untouched."""
    cl = np.full(5000, 1e-2)
    assert np.all(cmb_fourier_weight((32, 32), PIX, cl, 0.01, 1.0, PIX**2, 0.0) == 1)
    assert np.all(
        cmb_fourier_weight((32, 32), PIX, np.zeros(10), 0.01, 1.0, PIX**2) == 1
    )


def test_weight_follows_the_ratio_of_cmb_to_noise():
    """N / (N + C B^2): small where the CMB dominates and tending to 1 where the beam removes it."""
    shape, beam, area, ivar = (64, 64), np.deg2rad(2.1 / 60), (10 * PIX) ** 2, 4.0
    cl = np.full(300000, 1e-3)  # flat C_l, uK^2 sr
    w = cmb_fourier_weight(shape, PIX, cl, beam, ivar, area)
    assert w.shape == (64, 33)
    ell = (
        2
        * np.pi
        * np.hypot(*np.meshgrid(np.fft.fftfreq(64), np.fft.rfftfreq(64), indexing="ij"))
        / PIX
    )
    expected = 1 / (
        1 + ivar * 1e-3 * np.exp(-((ell * beam / np.sqrt(8 * np.log(2))) ** 2)) / area
    )
    np.testing.assert_allclose(w, expected)
    assert w[0, 1] < w[0, 32]  # more CMB, less weight, at low l
    assert w.max() <= 1


def test_lensed_cmb_cl_has_the_first_acoustic_peak(tmp_path):
    """D_l = l (l + 1) C_l / 2 pi is about 5700 uK^2 at l = 220, and the cache is reused."""
    pytest.importorskip("camb")
    cache = str(tmp_path / "cl.npy")
    cl = lensed_cmb_cl(lmax=2500, cache=cache)
    assert cl[0] == cl[1] == 0
    assert 5000 < cl[220] * 220 * 221 / (2 * np.pi) < 6500
    np.testing.assert_array_equal(lensed_cmb_cl(lmax=2500, cache=cache), cl)
