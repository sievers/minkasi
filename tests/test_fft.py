import minkasi.tools.fft as old_fft
import minkasi.tools.py_fft as new_fft

import numpy as np
import jax
import jax.numpy as jnp


def test_rfft():
    dat = np.random.rand(190, 14394)
    old_datft = old_fft.rfftn(dat)
    new_datft = new_fft.rfftn(dat)

    assert np.all(np.isclose(old_datft, new_datft))

    old_dat = old_fft.irfftn(old_datft)
    new_dat = new_fft.irfftn(new_datft)

    assert np.all(np.isclose(new_dat, old_dat))
    assert np.all(np.isclose(new_dat, dat))

    dat = np.random.rand(190, 14395)
    old_datft = old_fft.rfftn(dat)
    new_datft = new_fft.rfftn(dat)

    assert np.all(np.isclose(old_datft, new_datft))

    old_dat = old_fft.irfftn(old_datft, iseven=False)
    new_dat = new_fft.irfftn(new_datft, iseven=False)

    assert np.all(np.isclose(new_dat, old_dat))
    assert np.all(np.isclose(new_dat, dat))


def test_3dfft():
    dat = np.random.rand(100, 100, 100)

    old_datft = old_fft.fft_r2c_3d(dat=dat)
    new_datft = new_fft.fft_r2c_3d(dat=dat)

    assert np.all(np.isclose(old_datft, new_datft))

    old_dat = old_fft.fft_c2r_3d(old_datft)
    new_dat = new_fft.fft_c2r_3d(new_datft)

    assert np.all(np.isclose(new_dat, old_dat))
    assert np.all(np.isclose(new_dat, dat))

    dat = np.random.rand(100, 100, 99)

    old_datft = old_fft.fft_r2c_3d(dat=dat)
    new_datft = new_fft.fft_r2c_3d(dat=dat)

    assert np.all(np.isclose(old_datft, new_datft))

    old_dat = old_fft.fft_c2r_3d(old_datft, iseven=False)
    new_dat = new_fft.fft_c2r_3d(new_datft, iseven=False)

    assert np.all(np.isclose(new_dat, old_dat))
    assert np.all(np.isclose(new_dat, dat))


def test_mult_r2c():
    dat = np.random.rand(100, 5000)

    dat_shape = dat.shape
    dat_shape = np.asarray(dat_shape, dtype="int")
    ft_shape = dat_shape.copy()
    ft_shape[-1] = ft_shape[-1] // 2 + 1

    old_datft = old_fft.fft_r2c(dat=dat)
    new_datft = new_fft.fft_r2c(dat=dat)

    # The fftw implementation of fft_r2c returns n_dat x m_dat
    # instead of the expected n_dat x (m_dat // 2 + 1). This
    # check only compares the valid part of the ffts
    assert np.all(
        np.isclose(
            old_datft[: ft_shape[0], : ft_shape[1]],
            new_datft[: ft_shape[0], : ft_shape[1]],
        )
    )

    old_dat = old_fft.fft_c2r(datft=old_datft)
    new_dat = new_fft.fft_c2r(datft=new_datft)

    assert np.all(np.isclose(old_dat, new_dat))
    assert np.all(np.isclose(dat, new_dat))


def test_fft_r2r_1d():
    dat = np.random.rand(10000)

    old_datft = old_fft.fft_r2r_1d(dat)
    new_datft = new_fft.fft_r2r_1d(dat)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r_1d(dat, kind=2)
    new_datft = new_fft.fft_r2r_1d(dat, kind=2)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r_1d(dat, kind=3)
    new_datft = new_fft.fft_r2r_1d(dat, kind=3)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r_1d(dat, kind=4)
    new_datft = new_fft.fft_r2r_1d(dat, kind=4)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r_1d(dat, kind=11)
    new_datft = new_fft.fft_r2r_1d(dat, kind=11)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r_1d(dat, kind=12)
    new_datft = new_fft.fft_r2r_1d(dat, kind=12)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r_1d(dat, kind=13)
    new_datft = new_fft.fft_r2r_1d(dat, kind=13)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r_1d(dat, kind=14)
    new_datft = new_fft.fft_r2r_1d(dat, kind=14)

    assert np.all(np.isclose(old_datft, new_datft))


def test_fft_r2r():
    dat = np.random.rand(100, 10000)

    old_datft = old_fft.fft_r2r(dat)
    new_datft = new_fft.fft_r2r(dat)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r(dat, kind=2)
    new_datft = new_fft.fft_r2r(dat, kind=2)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r(dat, kind=3)
    new_datft = new_fft.fft_r2r(dat, kind=3)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r(dat, kind=4)
    new_datft = new_fft.fft_r2r(dat, kind=4)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r(dat, kind=11)
    new_datft = new_fft.fft_r2r(dat, kind=11)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r(dat, kind=12)
    new_datft = new_fft.fft_r2r(dat, kind=12)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r(dat, kind=13)
    new_datft = new_fft.fft_r2r(dat, kind=13)

    assert np.all(np.isclose(old_datft, new_datft))

    old_datft = old_fft.fft_r2r(dat, kind=14)
    new_datft = new_fft.fft_r2r(dat, kind=14)

    assert np.all(np.isclose(old_datft, new_datft))


def test_unchanged_funcs():
    n = 123
    old_int = old_fft.find_good_fft_lens(n)
    new_int = new_fft.find_good_fft_lens(n)
    assert old_int == new_int
