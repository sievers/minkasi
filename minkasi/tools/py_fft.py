import numpy as np
from numpy.typing import NDArray

import jax
import jax.numpy as jnp

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