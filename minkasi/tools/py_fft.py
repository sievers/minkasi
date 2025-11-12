import numpy as np
from numpy.typing import NDArray

import jax
import jax.numpy as jnp

from typing import Optional, Sequence, Union, overload

jax.config.update("jax_enable_x64", True)


def rfftn(
    dat: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """
    Take the FFT of a real valued input in Nd.

    Parameters
    ----------
    dat : NDArray[np.float64]
        Array to FFT.

    Returns
    -------
    datft : NDArray[np.complex128]
        The FFTed array.
        Given a dat with shape (..., n), datft will have shape (..., n//2 + 1)
    """
    datft = jnp.fft.rfftn(dat)
    return datft

def irfftn(
    datft: NDArray[np.complex128], iseven: bool = True, preserve_input: bool = True
) -> NDArray[np.float64]:
    """
    Inverse FFT with real valued output in Nd.

    Parameters
    ----------
    datft : NDArray[np.complex128]
        The FFTed array.
    iseven : bool, default: True
        Set to True if the output should be even.
        Assuming datft has shape (..., n) this means the output will have shape (..., 2*(n-1)).
        If False the output will have shape (..., 2*n - 1).
    preserve_input : bool, default: True
        Dummy input for compatibility with FFTW functions; jax always preserves input.

    Returns
    -------
    dat : NDArray[np.float64]
        The inverse FFT. See iseven for shape details.
    """
    ft_shape = np.asarray(datft.shape, dtype="int32")
    dat_shape = ft_shape.copy()
    if iseven:
        dat_shape[-1] = 2 * (dat_shape[-1] - 1)
    else:
        dat_shape[-1] = 2 * dat_shape[-1] - 1
 
    dat = jnp.fft.irfftn(datft, s= dat_shape)
    return dat

def fft_r2c_3d(dat: NDArray[np.float64]) -> NDArray[np.complex128]:
    """
    Take the FFT of a real valued input in 3d.

    Parameters
    ----------
    dat : NDArray[np.float64]
        3d array to FFT.

    Returns
    -------
    datft : NDArray[np.complex128]
        The FFTed array.
        Given a dat with shape (..., n), datft will have shape (..., n//2 + 1)
    """
    dat_shape = dat.shape
    assert len(dat_shape) == 3
    datft = rfftn(dat=dat)
    return datft

def fft_c2r_3d(
    datft: NDArray[np.complex128], iseven: bool = True, preserve_input: bool = True
) -> NDArray[np.float64]:
    """
    Inverse FFT with real valued output in 3d.

    Parameters
    ----------
    datft : NDArray[np.complex128]
        The 3d FFTed array.
    iseven : bool, default: True
        Set to True if the output should be even.
        Assuming datft has shape (..., n) this means the output will have shape (..., 2*(n-1)).
        If False the output will have shape (..., 2*n - 1).
    preserve_input : bool, default: True
        Dummy input for compatibility with FFTW functions; jax always preserves input.

    Returns
    -------
    dat : NDArray[np.float64]
        The inverse FFT. See iseven for shape details.
    """
    ft_shape = datft.shape
    assert len(ft_shape) == 3
    dat = irfftn(datft=datft, iseven=iseven)
    return dat

@overload
def fft_r2c(dat: NDArray[np.float64]) -> NDArray[np.complex128]:
    ...


@overload
def fft_r2c(dat: NDArray[np.float32]) -> NDArray[np.complex64]:
    ...


def fft_r2c(
    dat: Union[NDArray[np.float64], NDArray[np.float32]],
) -> Union[NDArray[np.complex128], NDArray[np.complex64]]:
    """
    Take many 1d FFTs of a real valued input.

    Parameters
    ----------
    dat : NDArray[np.float64] | NDArray[np.float32]
        Array to FFT.
        Should be (ndat, ntrans) where each row has a 1d FFT applied.

    Returns
    -------
    datft : NDArray[np.complex128] | NDArray[np.complex64]
        The FFTed array.
    """
    if dat.dtype == np.dtype("float64"):
        datft = np.empty(dat.shape, dtype="complex128")
    elif dat.dtype == np.dtype("float32"):
        datft = np.empty(dat.shape, dtype="complex64")
    else:
        raise TypeError("Error, input must be float32 or float64.")

    for i in range(dat.shape[0]):
        cur_dat = dat[i]
        #For whatever reason the legacy fftw code returns an array of 
        #n_dat x m_dat, instead of the usual n_dat x (m_dat // 2 + 1)
        #We keep the legacy shape, so we have to do a slightly ugly asignment. 
        cur_datft = jnp.fft.rfft(cur_dat)
        datft[i,:len(cur_datft)] = cur_datft 

    return datft

@overload
def fft_c2r(datft: NDArray[np.complex128]) -> NDArray[np.float64]:
    ...


@overload
def fft_c2r(datft: NDArray[np.complex64]) -> NDArray[np.float32]:
    ...


def fft_c2r(
    datft: Union[NDArray[np.complex128], NDArray[np.complex64]],
) -> Union[NDArray[np.float64], NDArray[np.float32]]:
    """
    Take many 1d inverse FFTs with a real valued output.

    Parameters
    ----------
    datft : NDArray[np.complex128] | NDArray[np.complex64]
        The FFTed array.
        Should be (ndat, ntrans) where each row has a 1d FFT applied.

    Returns
    -------
    dat : NDArray[np.float64] | NDArray[np.float32]
        The inverse FFT.
    """
    if datft.dtype == np.dtype("complex128"):
        dat = np.empty(datft.shape, dtype="float64")
    elif datft.dtype == np.dtype("complex64"):
        dat = np.empty(datft.shape, dtype="float32")
    else:
        raise TypeError("Error, input must be float32 or float64.")

    ndat = datft.shape[1]

    for i in range(datft.shape[0]):
        cur_datft = datft[i]
        #For whatever reason the legacy fftw code returns an array of 
        #n_dat x m_dat, instead of the usual n_dat x (m_dat // 2 + 1)
        #We keep the legacy shape, so we have to do a slightly ugly asignment. 
        cur_datft = jnp.fft.irfft(cur_datft)
        dat[i] = cur_datft 

        dat = dat / ndat
    
    return dat



