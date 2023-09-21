import os
from typing import overload
import numpy as np
from numpy.typing import NDArray
import ctypes

try:
    mylib = ctypes.cdll.LoadLibrary("libmkfftw.so")
except OSError:
    mylib = ctypes.cdll.LoadLibrary(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmkfftw.so")
    )

many_fft_r2c_1d_c = mylib.many_fft_r2c_1d
many_fft_r2c_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


many_fftf_r2c_1d_c = mylib.many_fftf_r2c_1d
many_fftf_r2c_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


many_fft_c2r_1d_c = mylib.many_fft_c2r_1d
many_fft_c2r_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

many_fftf_c2r_1d_c = mylib.many_fftf_c2r_1d
many_fftf_c2r_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

fft_r2r_1d_c = mylib.fft_r2r_1d
fft_r2r_1d_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

many_fft_r2r_1d_c = mylib.many_fft_r2r_1d
many_fft_r2r_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

many_fftf_r2r_1d_c = mylib.many_fftf_r2r_1d
many_fftf_r2r_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

fft_r2c_n_c = mylib.fft_r2c_n
fft_r2c_n_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

fft_c2r_n_c = mylib.fft_c2r_n
fft_c2r_n_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


fft_r2c_3d_c = mylib.fft_r2c_3d
fft_r2c_3d_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

fft_c2r_3d_c = mylib.fft_c2r_3d
fft_c2r_3d_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


set_threaded_c = mylib.set_threaded
set_threaded_c.argtypes = [ctypes.c_int]

read_wisdom_c = mylib.read_wisdom
read_wisdom_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

write_wisdom_c = mylib.write_wisdom
write_wisdom_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def set_threaded(n: int = -1):
    """
    Set number of threads for FFTW.

    Parameters
    ----------
    n : int, default: -1
        Number of threads to use.
        If negetive then all availible threads are used.
    """
    set_threaded_c(n)


def rfftn(dat: NDArray[np.float64]) -> NDArray[np.complex128]:
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
    dat_shape = np.asarray(dat.shape, dtype="int32")
    ft_shape = dat_shape.copy()
    ft_shape[-1] = ft_shape[-1] // 2 + 1
    datft = np.zeros(ft_shape, dtype="complex")
    fft_r2c_n_c(
        dat.ctypes.data, datft.ctypes.data, len(dat_shape), dat_shape.ctypes.data
    )

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
        The c2r transforms destroy input, set to True to make a copy and preserve datft.

    Returns
    -------
    dat : NDArray[np.float64]
        The inverse FFT. See iseven for shape details.
    """
    if preserve_input:
        datft = datft.copy()
    ft_shape = np.asarray(datft.shape, dtype="int32")
    dat_shape = ft_shape.copy()
    if iseven:
        dat_shape[-1] = 2 * (dat_shape[-1] - 1)
    else:
        dat_shape[-1] = 2 * dat_shape[-1] - 1
    dat = np.empty(dat_shape, dtype="float64")
    fft_c2r_n_c(
        datft.ctypes.data, dat.ctypes.data, len(dat_shape), dat_shape.ctypes.data
    )
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
    dat_shape = np.asarray(dat_shape, dtype="int")
    ft_shape = dat_shape.copy()
    ft_shape[-1] = ft_shape[-1] // 2 + 1
    datft = np.zeros(ft_shape, dtype="complex")
    fft_r2c_3d_c(dat.ctypes.data, datft.ctypes.data, dat_shape.ctypes.data)
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
        The c2r transforms destroy input, set to True to make a copy and preserve datft.

    Returns
    -------
    dat : NDArray[np.float64]
        The inverse FFT. See iseven for shape details.
    """
    if preserve_input:
        datft = datft.copy()
    ft_shape = datft.shape
    assert len(ft_shape) == 3
    ft_shape = np.asarray(ft_shape, dtype="int")
    dat_shape = ft_shape.copy()
    if iseven:
        dat_shape[-1] = 2 * (dat_shape[-1] - 1)
    else:
        dat_shape[-1] = 2 * dat_shape[-1] - 1
    dat = np.empty(dat_shape, dtype="float64")
    fft_c2r_3d_c(datft.ctypes.data, dat.ctypes.data, dat_shape.ctypes.data)
    return dat


@overload
def fft_r2c(dat: NDArray[np.float64]) -> NDArray[np.complex128]:
    ...


@overload
def fft_r2c(dat: NDArray[np.float32]) -> NDArray[np.complex64]:
    ...


def fft_r2c(
    dat: NDArray[np.float64] | NDArray[np.float32],
) -> NDArray[np.complex128] | NDArray[np.complex64]:
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
    ndat = dat.shape[1]
    ntrans = dat.shape[0]

    if dat.dtype == np.dtype("float64"):
        # datft=np.zeros(dat.shape,dtype=complex)
        datft = np.empty(dat.shape, dtype=complex)
        many_fft_r2c_1d_c(dat.ctypes.data, datft.ctypes.data, ntrans, ndat, ndat, ndat)
    else:
        assert dat.dtype == np.dtype("float32")
        datft = np.empty(dat.shape, dtype="complex64")
        many_fftf_r2c_1d_c(dat.ctypes.data, datft.ctypes.data, ntrans, ndat, ndat, ndat)
    return datft


@overload
def fft_c2r(datft: NDArray[np.complex128]) -> NDArray[np.float64]:
    ...


@overload
def fft_c2r(datft: NDArray[np.complex64]) -> NDArray[np.float32]:
    ...


def fft_c2r(
    datft: NDArray[np.complex128] | NDArray[np.complex64],
) -> NDArray[np.float64] | NDArray[np.float32]:
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
    ndat = datft.shape[1]
    ntrans = datft.shape[0]
    if datft.dtype == np.dtype("complex128"):
        dat = np.zeros(datft.shape)
        many_fft_c2r_1d_c(datft.ctypes.data, dat.ctypes.data, ntrans, ndat, ndat, ndat)
        dat = dat / ndat
    else:
        assert datft.dtype == np.dtype("complex64")
        dat = np.zeros(datft.shape, dtype="float32")
        many_fftf_c2r_1d_c(datft.ctypes.data, dat.ctypes.data, ntrans, ndat, ndat, ndat)
        dat = dat / np.float32(ndat)
    return dat


def fft_r2r_1d(dat: NDArray[np.float64], kind: int = 1) -> NDArray[np.float64]:
    """
    1d real to real FFW.

    Parameters
    ----------
    dat : NDArray[np.float64]
        The data to FFT.
    kind : int, default: 1
        The kind of r2r transform.
        Accepted values and their respective transforms are:
        * 1 = FFTW_REDFT00
        * 2 = FFTW_REDFT10
        * 3 = FFTW_REDFT01
        * 4 = FFTW_REDFT11
        * 11 = FFTW_RODFT00
        * 12 = FFTW_RODFT10
        * 13 = FFTW_RODFT01
        * 14 = FFTW_RODFT11
        If a different value is used transform will default to FFTW_REDFT00.

    Returns
    -------
    datft : NDArray[np.float64]
        The FFTed data.
    """
    nn = dat.size
    trans = np.zeros(nn)
    fft_r2r_1d_c(dat.ctypes.data, trans.ctypes.data, nn, kind)
    return trans


@overload
def fft_r2r(
    dat: NDArray[np.float64], datft: NDArray[np.float64] | None = None, kind: int = 1
) -> NDArray[np.float64]:
    ...


@overload
def fft_r2r(
    dat: NDArray[np.float32], datft: NDArray[np.float32] | None = None, kind: int = 1
) -> NDArray[np.float32]:
    ...


def fft_r2r(
    dat: NDArray[np.float64] | NDArray[np.float32],
    datft: NDArray[np.float64] | NDArray[np.float32] | None = None,
    kind: int = 1,
) -> NDArray[np.float64] | NDArray[np.float32]:
    """
    Take many 1d real to real FFTs.

    Parameters
    ----------
    dat : NDArray[np.float64] | NDArray[np.float32]
        Array to FFT.
        Should be (ndat, ntrans) where each row has a 1d FFT applied.
        If a 1d array is given then fft_r2r_1d is called instead.
    datft : NDArray[np.complex128] | NDArray[np.complex64] | None
        Array to store FFT in.
        If None an empty array is initialized, if not this one is filled in place.
        If not None then it should have the same dtype as dat.
    kind : int, default: 1
        The kind of r2r transform.
        See docstring for fft_r2r_1d for details.

    Returns
    -------
    datft : NDArray[np.float64] | NDArray[np.float32]
        The FFTed array.
    """
    if datft is not None:
        assert dat.dtype == datft.dtype
    if len(dat.shape) == 1:
        _datft = fft_r2r_1d(dat.astype(np.float64), kind)
        if dat.dtype == np.dtype("float32"):
            _datft = _datft.astype("float32")
        if datft is not None:
            datft[:] = _datft[:]
        return _datft

    ntrans = dat.shape[0]
    n = dat.shape[1]
    if datft is None:
        datft = np.empty([ntrans, n], dtype=type(dat[0, 0]))

    if type(dat[0, 0]) == np.dtype("float32"):
        many_fftf_r2r_1d_c(dat.ctypes.data, datft.ctypes.data, n, kind, ntrans)
    else:
        many_fft_r2r_1d_c(dat.ctypes.data, datft.ctypes.data, n, kind, ntrans)
    return datft


def read_wisdom(double_file: str = ".fftw_wisdom", single_file: str = ".fftwf_wisdom"):
    """
    Load FFTW wisdom files from disk.

    Parameters
    ----------
    double_file : str, default: ".fftw_wisdom"
        Path to wisdom file for the 64 bit version of fftw.
    single_file: str, default: ".fftwf_wisdom"
        Path to wisdom file for the 32 bit version of fftw.
    """
    df = np.zeros(len(double_file) + 1, dtype="int8")
    df[0:-1] = [ord(c) for c in double_file]

    sf = np.zeros(len(single_file) + 1, dtype="int8")
    sf[0:-1] = [ord(c) for c in single_file]

    read_wisdom_c(df.ctypes.data, sf.ctypes.data)


def write_wisdom(double_file: str = ".fftw_wisdom", single_file: str = ".fftwf_wisdom"):
    """
    Write FFTW wisdom files to disk.

    Parameters
    ----------
    double_file : str, default: ".fftw_wisdom"
        Path to wisdom file for the 64 bit version of fftw.
    single_file: str, default: ".fftwf_wisdom"
        Path to wisdom file for the 32 bit version of fftw.
    """
    df = np.zeros(len(double_file) + 1, dtype="int8")
    df[0:-1] = [ord(c) for c in double_file]

    sf = np.zeros(len(single_file) + 1, dtype="int8")
    sf[0:-1] = [ord(c) for c in single_file]

    write_wisdom_c(df.ctypes.data, sf.ctypes.data)
