import numpy as np
from astropy.io import fits
from numpy.typing import NDArray


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
