"""
Functions for beam smoothing.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from . import fft
except ImportError:
    from . import py_fft as fft


def smooth_spectra(
    spec: NDArray[np.floating], fwhm: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Smooth spectra with a gaussian.

    Parameters
    ----------
    spec : NDArray[np.floating]
        The specta to smooth.
        Should be a 2d array where each row is a spectra.
    fwhm : float
        The full width half max of the gaussian.

    Returns
    -------
    xtrans : NDArray[np.floating]
        The smoothed spectra.
    to_conv_ft : NDArray[np.floating]
        The fourier transform of the gaussian used to smooth.
    """
    n = spec.shape[1]
    x = np.arange(n)
    sig = fwhm / np.sqrt(8 * np.log(2))
    to_conv = np.exp(-0.5 * (x / sig) ** 2)
    tot = to_conv[0] + to_conv[-1] + 2 * to_conv[1:-1].sum()  # r2r normalization
    to_conv = to_conv / tot
    to_conv_ft = fft.fft_r2r(to_conv)
    xtrans = fft.fft_r2r(spec)
    xtrans *= to_conv_ft
    return xtrans, to_conv_ft


def smooth_many_vecs(
    vecs: NDArray[np.floating], fwhm: float = 20
) -> NDArray[np.floating]:
    """
    Smooth vectors with a gaussian.

    Parameters
    ----------
    spec : NDArray[np.floating]
        The data to smooth.
        Should be a 2d array that will be smoothed along each row.
    fwhm : float, default: 20
        The full width half max of the gaussian.

    Returns
    -------
    smoothed : NDArray[np.floating]
        The smoothed data.
    """
    n = vecs.shape[1]
    x = np.arange(n)
    sig = fwhm / np.sqrt(8 * np.log(2))
    to_conv = np.exp(-0.5 * (x / sig) ** 2)
    tot = to_conv[0] + to_conv[-1] + 2 * to_conv[1:-1].sum()  # r2r normalization
    to_conv = to_conv / tot
    to_conv_ft = fft.fft_r2r(to_conv)
    xtrans = fft.fft_r2r(vecs)
    xtrans *= to_conv_ft
    back = fft.fft_r2r(xtrans)
    return back / (2 * (n - 1))


def smooth_vec(vec: NDArray[np.floating], fwhm: float = 20) -> NDArray[np.floating]:
    """
    Smooth a vector with a gaussian.

    Parameters
    ----------
    spec : NDArray[np.floating]
        The data to smooth.
    fwhm : float, default: 20
        The full width half max of the gaussian.

    Returns
    -------
    smoothed : NDArray[np.floating]
        The smoothed data.
    """
    vecs = vec[None, :]
    smoothed = smooth_many_vecs(vecs, fwhm)
    return smoothed[0]
