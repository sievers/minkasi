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
    dat = np.random.rand(100,100,100)

    old_datft = old_fft.fft_r2c_3d(dat=dat)
    new_datft = new_fft.fft_r2c_3d(dat=dat)

    assert np.all(np.isclose(old_datft, new_datft))

    old_dat = old_fft.fft_c2r_3d(old_datft)
    new_dat = new_fft.fft_c2r_3d(new_datft)

    assert np.all(np.isclose(new_dat, old_dat))
    assert np.all(np.isclose(new_dat, dat))

    dat = np.random.rand(100,100,99)

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

    #The fftw implementation of fft_r2c returns n_dat x m_dat
    #instead of the expected n_dat x (m_dat // 2 + 1). This
    #check only compares the valid part of the ffts
    assert np.all(np.isclose(old_datft[:ft_shape[0], :ft_shape[1]], new_datft[:ft_shape[0], :ft_shape[1]]))





