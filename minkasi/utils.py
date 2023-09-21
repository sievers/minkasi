"""
Useful functions that don't have a better place to be.
"""
from typing import Iterable, Sequence
import numpy as np
from numpy.typing import NDArray
from maps import MapType

try:
    have_numba = True
    import numba as nb
except ImportError:
    have_numba = False
    from . import no_numba as nb

kb = 1.38064852e-16
h = 6.62607004e-27
T = 2.725


def y2rj(freq: float = 90) -> float:
    """
    Get the y to Rayleigh-Jeans conversion at a given frequency.

    Parameters
    ----------
    freq : float, default: 90
        The frequency to get the conversion at.

    Returns
    -------
    y2rj : float
        Conversion to multiply a y map by to get a Rayleigh-Jeans normalized map note
        that it doesn't have the T_cmb at the end, so the value for low frequencies is -2.
    ."""
    x = freq * 1e9 * h / kb / T
    ex = np.exp(x)
    f = x**2 * ex / (ex - 1) ** 2 * (x * (ex + 1) / (ex - 1) - 4)
    return f


def planck_g(freq: float = 90) -> float:
    """
    Conversion between T_CMB and T_RJ as a function of frequency.

    Parameters
    ----------
    freq : float, default: 90
        The frequency to get the conversion at.

    Returns
    -------
    planck_g : float
        The conversion factor.
    """
    x = freq * 1e9 * h / kb / T
    ex = np.exp(x)
    return x**2 * ex / ((ex - 1) ** 2)


def _nsphere_vol(npp: int) -> float:
    """
    Find volume of an nsphere.

    Parameters
    ----------
    npp : int
        Dimension of nsphere.

    Returns
    -------
    vol : float
        The volume of the nsphere.
    """
    if npp % 2:
        nn = (npp - 1) / 2
        vol = 2 ** (nn + 1) * np.pi**nn / np.prod(np.arange(1, npp + 1, 2))
    else:
        nn = npp / 2
        vol = (np.pi**nn) / np.prod(np.arange(1, nn + 1))
    return vol


def _prime_loop(
    ln: float,
    lp: NDArray[np.floating],
    icur: int,
    lcur: float,
    vals: NDArray[np.floating],
) -> int:
    """
    Loop through composites of primes in log2 space.

    Parameters
    ----------
    ln : float
        The limit of values in log2 space.
    lp : NDArray[np.floating]
        The primes in log2 space.
    icur : int
        The current index.
    lcur : float
        The current starting value in log2 space, start at 0.
    vals : NDArray[np.floating]
        The current composites in log2 space.
    """
    facs = np.arange(lcur, ln + 1e-3, lp[0])
    if len(lp) == 1:
        nfac = len(facs)
        if nfac > 0:
            vals[icur : (icur + nfac)] = facs
            icur = icur + nfac
        else:
            print("bad facs came from " + repr([2**lcur, 2**ln, 2 ** lp[0]]))
        return icur
    else:
        facs = np.arange(lcur, ln, lp[0])
        for fac in facs:
            icur = _prime_loop(ln, lp[1:], icur, fac, vals)
        return icur


def find_good_fft_lens(
    n: int, primes: Sequence[int] = [2, 3, 5, 7]
) -> NDArray[np.integer]:
    """
    Find a good FFT length.

    Parameters
    ----------
    n : int
        The length that we want to cut down to a good FFT length.
    primes : Sequence[int], default = [2,3,5,7]
        Prime numbers used as the radix.
        The length will be a composite of these numbers.
    """
    # lmax=np.log(n+0.5)
    npr = len(primes)
    vol = _nsphere_vol(npr)

    r = np.log2(n + 0.5)
    lp = np.log2(primes, dtype=float)
    int_max = (vol / 2**npr) * np.prod(
        r / lp
    ) + 30  # add a bit just to make sure we don't act up for small n
    int_max = int(int_max)

    vals = np.zeros(int_max)
    icur = 0
    icur = _prime_loop(r, lp, icur, 0.0, vals)
    assert icur <= int_max
    myvals = np.asarray(np.round(2 ** vals[:icur]), dtype="int")
    myvals = np.sort(myvals)
    return myvals


def _prep_rings(
    edges: Sequence[float] | NDArray[np.floating],
    cent: Sequence[float] | NDArray[np.floating],
    map: MapType,
    pixsize: float,
    fwhm: int | float | Sequence[int | float] | NDArray[np.integer | np.floating],
    amps: None | Sequence[float] | NDArray[np.floating],
    iswcs: bool,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.complex128],
]:
    """
    Helper function to store shared code for make_rings and make_rings_wSlope.
    See those functions for more details.
    """
    xvec = np.arange(map.nx)
    yvec = np.arange(map.ny)
    ix = int(map.nx / 2)
    iy = int(map.ny / 2)
    xvec[ix:] = xvec[ix:] - map.nx
    yvec[iy:] = yvec[iy:] - map.ny

    xmat = np.repeat([xvec], map.ny, axis=0).transpose()
    ymat = np.repeat([yvec], map.nx, axis=0)

    rmat = np.sqrt(xmat**2 + ymat**2) * pixsize
    if isinstance(fwhm, (float, int)):
        sig = fwhm / np.sqrt(8 * np.log(2))
        src_map = np.exp(-0.5 * rmat**2 / sig**2)
        src_map = src_map / src_map.sum()
    else:
        if amps is None:
            amps = np.ones_like(fwhm)
        if len(fwhm) != len(amps):
            raise ValueError("fwhm and amps are not the same length")
        sig = fwhm[0] / np.sqrt(8 * np.log(2))
        src_map = np.exp(-0.5 * rmat**2 / sig**2) * amps[0]
        for i in range(1, len(fwhm)):
            sig = fwhm[i] / np.sqrt(8 * np.log(2))
            src_map = src_map + np.exp(-0.5 * rmat**2 / sig**2) * amps[i]

        src_map = src_map / src_map.sum()
        beam_area = pixsize**2 / src_map.max()
        beam_area = beam_area / 3600**2 / (360**2 / np.pi)
        print("beam_area is ", beam_area * 1e9, " nsr")
    nring = len(edges) - 1
    rings = np.zeros([nring, map.nx, map.ny])
    if iswcs:
        mypix = map.wcs.wcs_world2pix(cent[0], cent[1], 1)
    else:
        mypix = cent
    mypix = np.array(mypix)

    print("mypix is ", mypix)

    xvec = np.arange(map.nx)
    yvec = np.arange(map.ny)
    xmat = np.repeat([xvec], map.ny, axis=0).transpose()
    ymat = np.repeat([yvec], map.nx, axis=0)

    srcft = np.fft.fft2(src_map)

    return rings, mypix, xmat, ymat, srcft


def make_rings(
    edges: Sequence[float] | NDArray[np.floating],
    cent: Sequence[float] | NDArray[np.floating],
    map: MapType,
    pixsize: float = 2.0,
    fwhm: int
    | float
    | Sequence[int | float]
    | NDArray[np.integer | np.floating] = 10.0,
    amps: None | Sequence[float] | NDArray[np.floating] = None,
    iswcs: bool = True,
) -> NDArray[np.floating]:
    """
    Make rings.

    Parameters
    ----------
    edges : Sequence[float] | NDArray[np.floating]
        The edges of the rings. Will produce len(edges)-1 rings.
    cent : Sequence[float] | NDArray[np.floating]
        The center of the rings.
        If iswcs is True these will be converted to pixels with the WCS.
        Otherwise are taken as is.
    map : MapType
        The map object to get coordinate information from.
    pixsize : float, default: 2.0
        The size of the ring pixels in units of map pixels.
    fwhm : int | float | Sequence[int | float] | NDArray[np.floating], default: 10.0
        the FWHM of the beam in pixels.
        If multiple are provided multiple amps should be provided as well.
    amps : None | Sequence[float] | NDArray[np.floating], default: None
        The beam amplitudes.
        If only a single FWHM is provided this should be None.
        Otherwise the length of this  should match the number of provided FWHMs.
    iswcs : bool, default: True
        If True then cent is converted to pixels using the map's WCS.

    Returns
    -------
    rings : NDArray[np.floating]
        A (nring, nx, ny) array of rings.
        nx and ny come from the map and nring is len(edges)-1.
    """
    rings, mypix, xmat, ymat, srcft = _prep_rings(
        edges, cent, map, pixsize, fwhm, amps, iswcs
    )
    rmat = np.sqrt((xmat - mypix[0]) ** 2 + (ymat - mypix[1]) ** 2) * pixsize
    for i, ring in enumerate(rings):
        # rings[i,(rmat>=edges[i])&(rmat<edges[i+1]=1.0
        ring[(rmat >= edges[i])] = 1.0
        ring[(rmat >= edges[i + 1])] = 0.0
        ring[:, :] = np.real(np.fft.ifft2(np.fft.fft2(ring[:, :]) * srcft))
    return rings


def make_rings_wSlope(
    edges: Sequence[float] | NDArray[np.floating],
    cent: Sequence[float] | NDArray[np.floating],
    vals: NDArray[np.floating],
    map: MapType,
    pixsize: float = 2.0,
    fwhm: int
    | float
    | Sequence[int | float]
    | NDArray[np.integer | np.floating] = 10.0,
    amps: None | Sequence[float] | NDArray[np.floating] = None,
    aa: float = 1.0,
    bb: float = 1.0,
    rot: float = 0.0,
) -> NDArray[np.floating]:
    """
    Make elliptical rings using the slope of vals.

    Parameters
    ----------
    edges : Sequence[float] | NDArray[np.floating]
        The edges of the rings. Will produce len(edges)-1 rings.
    cent : Sequence[float] | NDArray[np.floating]
        The center of the rings.
        If iswcs is True these will be converted to pixels with the WCS.
        Otherwise are taken as is.
    vals : NDArray[np.floating]
        The values to calculate slope from.
        Should be at least nring in length.
    map : MapType
        The map object to get coordinate information from.
    pixsize : float, default: 2.0
        The size of the ring pixels in units of map pixels.
    fwhm : int | float | Sequence[int | float] | NDArray[np.floating], default: 10.0
        the FWHM of the beam in pixels.
        If multiple are provided multiple amps should be provided as well.
    amps : None | Sequence[float] | NDArray[np.floating], default: None
        The beam amplitudes.
        If only a single FWHM is provided this should be None.
        Otherwise the length of this  should match the number of provided FWHMs.
    aa : float, default: 1.0
        Factor to scale in x.
    bb : float, default: 1.0
        Factor to scale in y.
    rot : float, default: 0.0
        Amount to rotate rings by.

    Returns
    -------
    rings : NDArray[np.floating]
        A (nring, nx, ny) array of rings.
        nx and ny come from the map and nring is len(edges)-1.
    """
    rings, mypix, xmat, ymat, srcft = _prep_rings(
        edges, cent, map, pixsize, fwhm, amps, True
    )
    nring = len(rings)

    xtr = (xmat - mypix[0]) * np.cos(rot) + (ymat - mypix[1]) * np.sin(
        rot
    )  # Rotate and translate x coords
    ytr = (ymat - mypix[1]) * np.cos(rot) - (xmat - mypix[0]) * np.sin(
        rot
    )  # Rotate and translate y coords
    rmat = (
        np.sqrt((xtr / aa) ** 2 + (ytr / bb) ** 2) * pixsize
    )  # Elliptically scale x,y
    myvals = vals[:nring] * 1.0  # Get just the values that correspond to rings
    myvals -= np.max(myvals)  # Set it such that the maximum value approaches 0
    pk2pk = np.max(myvals) - np.min(myvals)
    myvals -= (
        pk2pk / 50.0
    )  # Let's assume we're down about a factor of 50 at the outskirts.

    for i in range(nring):
        # rings[i,(rmat>=edges[i])&(rmat<edges[i+1]=1.0
        if i == nring - 1:
            slope = 0.0
        else:
            slope = (myvals[i] - myvals[i + 1]) / (
                edges[i + 1] - edges[i]
            )  # expect positve slope; want negative one.
        rgtinedge = rmat >= edges[i]
        rfromin = rmat - edges[i]
        initline = rfromin[rgtinedge] * slope
        if myvals[i] != 0:
            # Should be normalized to 1 now.
            rings[i, rgtinedge] = (myvals[i] - initline) / myvals[i]
        else:
            rings[i, rgtinedge] = 1.0
        rgtoutedge = rmat >= edges[i + 1]
        rings[i, rgtoutedge] = 0.0
        # The stuff computed below isn't currently used
        # myannul = [
        #     c1 and not (c2) for c1, c2 in zip(rgtinedge.ravel(), rgtoutedge.ravel())
        # ]
        # rannul = rmat.ravel()[myannul]
        # rmin    = (rmat == np.min(rannul))
        # rmout   = (rmat == np.max(rannul))
        rings[i, :, :] = np.real(np.fft.ifft2(np.fft.fft2(rings[i, :, :]) * srcft))
    return rings


def plot_ps(vec, downsamp=0):
    """
    This function isn't actually implemented yet.
    I assume its to plot a power spectrum,
    """
    return
    vecft = mkfftw.fft_r2r(vec)


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
