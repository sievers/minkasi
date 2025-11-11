import os

from typing import Optional, Sequence, Union, overload

import numpy as np
from numpy.typing import NDArray
import pyfftw


def set_threaded(n: int = -1):
    """
    Set number of threads for FFTW.

    Parameters
    ----------
    n : int, default: -1
        Number of threads to use.
        If negetive then all availible threads are used.
    """

    if n < 0:
        n = os.environ.get("OMP_NUM_THREADS")
    print("Setting FFTW to have %d threads.".format(n))
    pyfftw.config.NUM_THREADS = n


def set_effort(effort: str = "FFTW_ESTIMATE"):
    """
    Set the planning effort for pyfftw. See:
    https://www.fftw.org/fftw3_doc/Planner-Flags.html

    Parameters
    ----------
    effort : str, default: "FFTW_ESTIMATE"
        Planning effort to use for pyfftw
    """
    pyfftw.config.PLANNER_EFFORT = effort


def rfftn(
    dat: NDArray[np.float64], effort: str = "FFTW_ESTIMATE"
) -> NDArray[np.complex128]:
    """
    Take the FFT of a real valued input in Nd.

    Parameters
    ----------
    dat : NDArray[np.float64]
        Array to FFT.
    effort : str, default: "FFTW_ESTIMATE"
        Planning effort to use for pyfftw

    Returns
    -------
    datft : NDArray[np.complex128]
        The FFTed array.
        Given a dat with shape (..., n), datft will have shape (..., n//2 + 1)
    """

    dat_shape = np.asarray(dat.shape, dtype="int32")
    ft_shape = dat_shape.copy()
    ft_shape[-1] = ft_shape[-1] // 2 + 1
    datft = np.zeros(ft_shape, dtype="complex")

    fft_object = pyfftw.builders.fftn(
        dat, s=ft_shape, norm="backward", planner_effort=effort
    )
    datft = fft_object()

    return datft
