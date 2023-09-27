import numpy as np
from numpy.typing import NDArray

from . import fft

try:
    have_numba = True
    import numba as nb
except ImportError:
    have_numba = False
    from . import no_numba as nb


def downsample_array_r2r(
    arr: NDArray[np.floating], fac: int, axis: int = -1
) -> NDArray[np.floating]:
    """
    Downsample array using fourier transform.

    Parameters
    ----------
    arr : NDArray[np.floating]
        Array to downsample.
    fac : int
        Factor to downsample by.
    axis : int, default: -1
        The axis to downsample along.

    Returns
    -------
    downsampled_arr : NDArray[np.floating]
        The downsampled array.
    """
    n = arr.shape[axis]
    nn = int(n / fac)
    arr_ft = fft.fft_r2r(arr)
    arr_ft = np.take(arr_ft, indices=range(0, nn), axis=axis)
    downsampled_arr = fft.fft_r2r(arr_ft) / (2 * (n - 1))
    return downsampled_arr


def downsample_vec_r2r(vec: NDArray[np.floating], fac: int) -> NDArray[np.floating]:
    """
    Just a wrapped around downsample_array_r2r to support legacy code.
    """
    return downsample_array_r2r(vec, fac)


def decimate(
    vec: NDArray[np.floating], nrep: int = 1, axis: int = -1
) -> NDArray[np.floating]:
    """
    Decimate an array by a factor of 2^nrep.

    Parameters
    ----------
    vec : NDArray[np.floating]
        The array to decimate.
    nrep : int, default: 1
        The number of times to decimate the array
    axis : int, default: -1
        The axis to decimate along.

    Returns
    -------
    decimated : NDArray[np.floating]
        The decimated array.
    """
    even = [slice(None)] * len(vec.shape)
    odd = [slice(None)] * len(vec.shape)
    for _ in range(nrep):
        end = vec.shape[axis]
        if end % 2:
            end -= 1
        even[axis] = slice(0, end, 2)
        odd[axis] = slice(1, end, 2)
        vec = 0.5 * (vec[tuple(even)] + vec[tuple(odd)])
    return vec


@nb.njit(parallel=True)
def axpy_in_place(y: NDArray[np.floating], x: NDArray[np.floating], a: float = 1.0):
    """
    Apply an A*x + y operation in place.

    Parameters
    ----------
    y : NDArray[np.floating]
        The y values in the axpy eq, should be 2d.
        This is modified in place.
    x : NDArray[np.floating]
        The x values in the axpy eq, should have the same shape as y.
    a : float, default: 1.0
        The A value in the axpy eq.
    """
    # add b into a
    n = x.shape[0]
    m = x.shape[1]
    assert n == y.shape[0]
    assert m == y.shape[1]
    # Numba has a bug, as of at least 0.53.1 (an 0.52.0) where
    # both parts of a conditional can get executed, so don't
    # try to be fancy.  Lower-down code can be used in the future.
    for i in nb.prange(n):
        y[i] += x[i] * a

    # isone=(a==1.0)
    # if isone:
    #    for  i in nb.prange(n):
    #        for j in np.arange(m):
    #            y[i,j]=y[i,j]+x[i,j]
    # else:
    #    for i in nb.prange(n):
    #        for j in np.arange(m):
    #            y[i,j]=y[i,j]+x[i,j]*a


@nb.njit(parallel=True)
def scale_matrix_by_vector(mat: NDArray, vec: NDArray, axis=1):
    """
    Multiply a matrix by a vector along an axis.

    Parameters
    ----------
    mat : NDArray
        The matrix to scale.
    vec : NDArray
        The vector to scale by.
        Should be of shape (n,) where n is the length of mat along axis.
    axis : int, default: 1
        The axis to scale along.
    """
    n = mat.shape[axis]
    assert len(vec) == n
    mat_moved = np.moveaxis(mat, axis, 0)
    for i in nb.prange(n):
        mat_moved[i] *= vec[i]
