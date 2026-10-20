import minkasi.tools.py_fft as new_fft

import numpy as np
import jax
import jax.numpy as jnp


def test_rfft():
    dat = np.random.rand(190, 14394)
    new_datft = new_fft.rfftn(dat)

    new_dat = new_fft.irfftn(new_datft)
    assert np.all(np.isclose(new_dat, dat))

    dat = np.random.rand(190, 14395)
    new_datft = new_fft.rfftn(dat)

    new_dat = new_fft.irfftn(new_datft, iseven=False)

    assert np.all(np.isclose(new_dat, dat))


def test_3dfft():
    dat = np.random.rand(100, 100, 100)
    new_datft = new_fft.fft_r2c_3d(dat=dat)
    new_dat = new_fft.fft_c2r_3d(new_datft)

    assert np.all(np.isclose(new_dat, dat))

    dat = np.random.rand(100, 100, 99)

    new_datft = new_fft.fft_r2c_3d(dat=dat)
    new_dat = new_fft.fft_c2r_3d(new_datft, iseven=False)

    assert np.all(np.isclose(new_dat, dat))


def test_mult_r2c():
    dat = np.random.rand(100, 5000)

    new_datft = new_fft.fft_r2c(dat=dat)
    new_dat = new_fft.fft_c2r(datft=new_datft)
    assert np.all(np.isclose(dat, new_dat))


def test_mult_r2c_32bit():
    dat = np.random.rand(100, 5000)
    dat = dat.astype(np.float32)

    new_datft = new_fft.fft_r2c(dat=dat)
    new_dat = new_fft.fft_c2r(datft=new_datft)

    # Note looser rtol since 32bits means we will lose precision
    # np.isclose defaults are very close to machine precision for inputs
    # with order of magnitude 1 or greater.
    assert np.all(np.isclose(dat, new_dat, rtol=3e-1))


def test_fft_r2r():
    dat = np.random.rand(100, 10000)
    new_datft = new_fft.fft_r2r(dat)

    # We can only do a consistency test here since old_ffw.fft_r2r doesn't work for these kinds
    block_datft = new_fft.fft_r2r(dat, kind=2)
    row_datft = np.zeros(block_datft.shape)

    for i in range(len(row_datft)):
        row_datft[i] = new_fft.fft_r2r_1d(dat[i], kind=2)

    assert np.all(np.isclose(block_datft, row_datft))

    block_datft = new_fft.fft_r2r(dat, kind=3)
    row_datft = np.zeros(block_datft.shape)

    for i in range(len(row_datft)):
        row_datft[i] = new_fft.fft_r2r_1d(dat[i], kind=3)

    assert np.all(np.isclose(block_datft, row_datft))

    block_datft = new_fft.fft_r2r(dat, kind=4)
    row_datft = np.zeros(block_datft.shape)

    for i in range(len(row_datft)):
        row_datft[i] = new_fft.fft_r2r_1d(dat[i], kind=4)

    assert np.all(np.isclose(block_datft, row_datft))

    block_datft = new_fft.fft_r2r(dat, kind=11)
    row_datft = np.zeros(block_datft.shape)

    for i in range(len(row_datft)):
        row_datft[i] = new_fft.fft_r2r_1d(dat[i], kind=11)

    assert np.all(np.isclose(block_datft, row_datft))

    block_datft = new_fft.fft_r2r(dat, kind=12)
    row_datft = np.zeros(block_datft.shape)

    for i in range(len(row_datft)):
        row_datft[i] = new_fft.fft_r2r_1d(dat[i], kind=12)

    assert np.all(np.isclose(block_datft, row_datft))

    block_datft = new_fft.fft_r2r(dat, kind=13)
    row_datft = np.zeros(block_datft.shape)

    for i in range(len(row_datft)):
        row_datft[i] = new_fft.fft_r2r_1d(dat[i], kind=13)

    assert np.all(np.isclose(block_datft, row_datft))

    block_datft = new_fft.fft_r2r(dat, kind=14)
    row_datft = np.zeros(block_datft.shape)

    for i in range(len(row_datft)):
        row_datft[i] = new_fft.fft_r2r_1d(dat[i], kind=14)

    assert np.all(np.isclose(block_datft, row_datft))
