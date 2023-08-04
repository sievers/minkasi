from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from astropy import wcs
from astropy.io import fits

try:
    import numba as nb
except ImportError:
    from .. import no_numba as nb


def read_fits_map(
    fname: str, hdu: int = 0, do_trans: bool = True
) -> NDArray[np.floating]:
    """
    Read a map from a FITs file.

    Parameters
    ----------
    fname : str
        The path to the fits file.
    hdu : int, default: 0
        Index of the map in the file.
    do_trans : bool, default: True
        Transpose the map before returning.

    Returns
    -------
    opened_map : NDArray[np.floating]
        Array representing the map.
    """
    f: fits.HDUList = fits.open(fname)
    raw: NDArray[np.floating] = f[hdu].data
    tmp: NDArray[np.floating] = raw.copy()
    f.close()
    if do_trans:
        tmp = (tmp.T).copy()
    return tmp


def write_fits_map_wheader(
    map: NDArray[np.floating], fname: str, header: fits.Header, do_trans: bool = True
):
    """
    Write out a map to a fits file.

    Parameters
    ----------
    map : NDArray[np.floating]
        Array with the map data.
    fname : str
        Path to save map at.
    header : fits.Header
        Header to save with map.
    do_trans : bool, default: True
        Transpose the map before saving.
    """
    if do_trans:
        map = (map.T).copy()
    hdu: fits.PrimaryHDU = fits.PrimaryHDU(map, header=header)
    try:
        hdu.writeto(fname, overwrite=True)
    except:
        hdu.writeto(fname, clobber=True)


def get_wcs(
    lims: tuple[float, ...] | list[float] | NDArray[np.floating],
    pixsize: float,
    proj: str = "CAR",
    cosdec: float | None = None,
    ref_equ: bool = False,
) -> wcs.WCS:
    """
    Create a WCS object with given limits and pixsize.

    Parameters
    ----------
    lims : tuple[float, ...] | list[float] | NDArray[np.floating]
        The limits of ra/dec (ra_low, ra_high, dec_low, dec_high).
    pixsize : float
        The size of a pixel.
    proj : str, default: 'CAR'
        The projection to use.
    cosdec : float | None, default: None
        The cosine declination correction.
        Set to None to have this function calculate it.
    ref_equ : bool, default: False
        Use equtorial reference.

    Returns
    -------
    wcs : wcs.WCS
        The created WCS object.

    Raises
    ------
    ValueError
        If the projection is unknown.
    """
    w: wcs.WCS = wcs.WCS(naxis=2)
    dec: float = 0.5 * (lims[2] + lims[3])
    if cosdec is None:
        cosdec = np.cos(dec)
    if proj == "CAR":
        # CAR in FITS seems to already correct for cosin(dec), which has me confused, but whatever...
        cosdec = 1.0
        if ref_equ:
            w.wcs.crval = [0.0, 0.0]
            # this seems to be a needed hack if you want the position sent
            # in for the corner to actually be the corner.
            w.wcs.crpix = [lims[1] / pixsize + 1, -lims[2] / pixsize + 1]
            # w.wcs.crpix=[lims[1]/pixsize,-lims[2]/pixsize]
            # print 'crpix is ',w.wcs.crpix
        else:
            w.wcs.crpix = [1.0, 1.0]
            w.wcs.crval = [lims[1] * 180 / np.pi, lims[2] * 180 / np.pi]
        w.wcs.cdelt = [-pixsize / cosdec * 180 / np.pi, pixsize * 180 / np.pi]
        w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
        return w
    raise ValueError(f"Unknown projection type, {proj}, in get_wcs.")


def get_aligned_map_subregion_car(
    lims: tuple[float, ...] | list[float] | NDArray[np.floating],
    fname: str | None = None,
    big_wcs: wcs.WCS | None = None,
    osamp: int = 1,
) -> tuple[wcs.WCS, NDArray[np.floating], NDArray[np.integer]]:
    """
    Get a wcs for a subregion of a map, with optionally finer pixellization.
    Designed for use in e.g. combining ACT maps and Mustang data.

    Parameters
    ----------
    lims : tuple[float, ...] | list[float] | NDArray[np.floating]
        The limits of ra/dec for the subregion (ra_low, ra_high, dec_low, dec_high).
        Will be tweaked as needed.
    fname : str | None, default: None
        Path to a file with the larger WCS that contains this subregion.
        This is only used it big_wcs is None.
    big_wcs : wcs.WCS | None, default: None
        Larger WCS that contains thes subregion.
        If this is None then fname is used.
    osamp : int, default: 1
        Factor to divide pixsize of larger WCS to get pixsize of subregion.

    Returns
    -------
    small_wcs : wcs.WCS
        The WCS of the subregion.
    lims_use : NDArray[np.floating]
        The limits actually used when making the subregion.
    map_corner : NDArray[np.integer]
        The corner pixel of the map.

    Raises
    ------
    ValueError
        If both fname and big_wcs are None.
    """

    if big_wcs is None:
        if fname is None:
            raise ValueError(
                "Error in get_aligned_map_subregion_car.  Must send in either a file or a WCS."
            )
        big_wcs = wcs.WCS(fname)
    ll: NDArray[np.floating] = np.asarray(lims)
    ll = ll * 180 / np.pi

    # get the ra/dec limits in big pixel coordinates
    corner1: NDArray[np.floating] = np.asarray(big_wcs.wcs_world2pix(ll[0], ll[2], 0))
    corner2: NDArray[np.floating] = np.asarray(big_wcs.wcs_world2pix(ll[1], ll[3], 0))

    # get the pixel edges for the corners.  FITS works in
    # pixel centers, so edges are a half-pixel off
    corner1[0] = np.ceil(corner1[0]) + 0.5
    corner1[1] = np.floor(corner1[1]) - 0.5
    corner2[0] = np.floor(corner2[0]) - 0.5
    corner2[1] = np.ceil(corner2[1]) + 0.5

    corner1_radec: NDArray[np.floating] = np.asarray(
        big_wcs.wcs_pix2world(corner1[0], corner1[1], 0)
    )
    corner2_radec: NDArray[np.floating] = np.asarray(
        big_wcs.wcs_pix2world(corner2[0], corner2[1], 0)
    )

    dra: float = (corner1_radec[0] - corner2_radec[0]) / (corner1[0] - corner2[0])
    ddec: float = (corner1_radec[1] - corner2_radec[1]) / (corner1[1] - corner2[1])
    assert (
        np.abs(dra / ddec) - 1 < 1e-5
    )  # we are not currently smart enough to deal with rectangular pixels

    lims_use: NDArray[np.floating] = np.asarray(
        [corner1_radec[0], corner2_radec[0], corner1_radec[1], corner2_radec[1]]
    )
    pixsize: float = ddec / osamp
    lims_use = lims_use + np.asarray([0.5, -0.5, 0.5, -0.5]) * pixsize

    small_wcs: wcs.WCS = get_wcs(
        lims_use * np.pi / 180, pixsize * np.pi / 180, ref_equ=True
    )
    imin: int = int(np.round(corner2[0] + 0.5))
    jmin: int = int(np.round(corner1[1] + 0.5))
    map_corner: NDArray[np.integer] = np.asarray([imin, jmin], dtype="int")
    lims_use = lims_use * np.pi / 180

    return small_wcs, lims_use, map_corner


def get_ft_vec(n: int) -> NDArray[np.integer]:
    """
    Get an array where the first half spans the range (0, n/2) and the second (-n/2, -1).

    Parameters
    ----------
    n : int
        The length of the output vector.

    Returns
    -------
    x : NDArray[np.integer]
        The ft vector.
    """
    x = np.arange(n)
    x[x > n / 2] = x[x > n / 2] - n
    return x


@nb.njit(parallel=True)
def radec2pix_car(
    ra: NDArray[np.floating],
    dec: NDArray[np.floating],
    ipix: NDArray[np.integer],
    lims: Sequence[float],
    pixsize: float,
    cosdec: float,
    ny: int,
):
    """
    Convert RA/dec to pixelization.

    Parameters
    ----------
    ra : NDArray[np.floating]
        The the RA TOD.
    dec : NDArray[np.floating]
        The the dec TOD.
    ipix : NDArray[np.integer]
        The the pixellization.
        Modified inplace.
    lims : Sequence[float]
        The limits of ra/dec (ra_low, ra_high, dec_low, dec_high).
    pixsize : float
        The pixel size in the same units as RA and dec.
    cosdec : float
        The declination stretch term.
    ny : int
        The number of pixels in the y (dec) direction.
    """
    ra = np.ravel(ra)
    dec = np.ravel(dec)
    ipix = np.ravel(ipix)
    n = len(ipix)
    for i in nb.prange(n):
        xpix = int((ra[i] - lims[0]) * cosdec / pixsize + 0.5)
        ypix = int((dec[i] - lims[2]) / pixsize + 0.5)
        ipix[i] = xpix * ny + ypix
