import numpy as np
from numpy.typing import NDArray


def zernike_column(
    m: int, nmax: int, rmat: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.int_]]:
    """
    Generate the radial part of zernike polynomials for all n from m up to nmax.

    Parameters
    ----------
    m : int
        The starting Zernike index.
    nmax : int
        The final Zernike index.
    rmat : NDArray[np.floating]
        The radii to compute the Zernikes at.

    Returns
    -------
    zn : NDArray[np.floating]
        The Zernikes evaluated at rmat.
        Set to 0 for rmat > 1.
    nn : NDArray[np.int_]
        Indices of each Zernike in the same order as zn.
    """
    if m > nmax:
        raise ValueError("m may not be larger than n")
    # if parity is wrong, then drop nmax by one.  makes external loop to generate all zns much simpler
    if (m - nmax) % 2 != 0:
        nmax = nmax - 1
    nm = int((nmax - m) / 2 + 1)

    mask = rmat > 1

    zn = np.zeros((nm,) + rmat.shape)
    nn = np.zeros(nm, dtype="int")

    zn[0] = rmat**m
    zn[0][mask] = 0
    nn[0] = m
    if nm == 1:
        return zn, nn

    rsqr = rmat**2
    zn[1] = ((m + 2) * rsqr - m - 1) * zn[0]
    zn[1][mask] = 0
    nn[1] = m + 2
    if nm == 2:
        return zn, nn

    ii = 2
    for n in range(m + 4, nmax + 1, 2):
        f1 = 2 * (n - 1) * (2 * n * (n - 2) * rsqr - m * m - n * (n - 2)) * zn[ii - 1]
        f2 = n * (n + m - 2) * (n - m - 2) * zn[ii - 2]
        f3 = 1.0 / ((n + m) * (n - m) * (n - 2))
        zn[ii] = (f1 - f2) * f3
        nn[ii] = n
        ii = ii + 1

    return zn, nn


def all_zernike(
    n: int, r: NDArray[np.floating], th: NDArray[np.floating]
) -> tuple[NDArray[np.floating], list]:
    """
    Compute full Zernikes from 0 to n.

    Parameters
    ----------
    n : int
        The maximum Zernike to compute.
    r : NDArray[np.floating]
        The radii to evaluate the Zernikes at.
    th : NDArray[np.floating]
        The angles to evaluate the Zernikes at.

    Returns
    -------
    zns : NDArray[np.floating]
        The Zernikes.
    znvec : list
        The radial part of each zernike.
    """
    znvec: list = [None] * (n + 1)
    nvec: list = [None] * (n + 1)
    mvec: list = [None] * (n + 1)
    nzer = 0
    for m in range(0, n + 1):
        znvec[m], nvec[m] = zernike_column(m, n, r)
        mvec[m] = 0 * nvec[m] + m
        if m == 0:
            nzer = nzer + len(znvec[m])
        else:
            nzer = nzer + 2 * len(znvec[m])

    shp = r.shape
    shp = np.append(nzer, shp)
    zns = np.zeros(shp)

    icur = 0
    for m in range(0, n + 1):
        ss = np.sin(m * th)
        cc = np.cos(m * th)
        zz = znvec[m]

        nn = len(zz)
        if m == 0:
            for i in range(nn):
                zns[icur, :] = zz[i]
                icur = icur + 1

        else:
            for i in range(nn):
                zns[icur, :] = zz[i] * cc
                icur = icur + 1
                zns[icur, :] = zz[i] * ss
                icur = icur + 1

    return zns, znvec
