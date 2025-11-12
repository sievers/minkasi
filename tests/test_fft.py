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