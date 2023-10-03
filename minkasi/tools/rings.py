from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..maps import MapType


def _prep_rings(
    edges: Union[Sequence[float], NDArray[np.floating]],
    cent: Union[Sequence[float], NDArray[np.floating]],
    map: "MapType",
    pixsize: float,
    fwhm: Union[
        int, float, Sequence[Union[int, float]], NDArray[Union[np.integer, np.floating]]
    ],
    amps: Optional[Union[Sequence[float], NDArray[np.floating]]],
    iswcs: bool,
) -> Tuple[
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
    mypix = np.array(mypix, dtype=float)

    print("mypix is ", mypix)

    xvec = np.arange(map.nx, dtype=float)
    yvec = np.arange(map.ny, dtype=float)
    xmat = np.repeat([xvec], map.ny, axis=0).transpose()
    ymat = np.repeat([yvec], map.nx, axis=0)

    srcft = np.fft.fft2(src_map)

    return rings, mypix, xmat, ymat, srcft


def make_rings(
    edges: Union[Sequence[float], NDArray[np.floating]],
    cent: Union[Sequence[float], NDArray[np.floating]],
    map: "MapType",
    pixsize: float,
    fwhm: Union[
        int, float, Sequence[Union[int, float]], NDArray[Union[np.integer, np.floating]]
    ],
    amps: Optional[Union[Sequence[float], NDArray[np.floating]]],
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
    edges: Union[Sequence[float], NDArray[np.floating]],
    cent: Union[Sequence[float], NDArray[np.floating]],
    vals: NDArray[np.floating],
    map: "MapType",
    pixsize: float = 2.0,
    fwhm: Union[
        int, float, Sequence[Union[int, float]], NDArray[Union[np.integer, np.floating]]
    ] = 10,
    amps: Optional[Union[Sequence[float], NDArray[np.floating]]] = None,
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
