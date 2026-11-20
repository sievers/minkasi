"""
check_act_pixels: the ACT term models each ACT pixel as ACT_OSAMP x ACT_OSAMP map pixels, so 30" ACT pixels need
3" map pixels; any other map pixel must be refused (or, with strict=False, warned about).
"""

import numpy as np
import pytest
from astropy.wcs import WCS

from minkasi.needlet.act import ACT_OSAMP, check_act_pixels


def car_wcs(pix_arcsec: float, ra_sign: float = -1.0, pc: bool = False) -> WCS:
    """
    A CAR WCS near MOOJ1142.

    Parameters
    ----------
    pix_arcsec : float
        Pixel size in arcsec.
    ra_sign : float, default: -1.0
        Sign of the RA increment (ACT maps have +1, M2 maps -1).
    pc : bool, default: False
        Put the scale in the PC matrix with CDELT = 1, as jitkasi's JAXWCS.to_header (WITCH) writes it.

    Returns
    -------
    WCS
        The WCS.
    """
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    w.wcs.crval = [175.7, 0.0]
    w.wcs.crpix = [1.0, -1833.5]
    step = np.array([ra_sign, 1.0]) * pix_arcsec / 3600
    if pc:
        w.wcs.cdelt = [1.0, 1.0]
        w.wcs.pc = np.diag(step)
    else:
        w.wcs.cdelt = step
    return w


def test_act_osamp_is_ten():
    """The pixel sizes below assume the 30" / 3" pairing."""
    assert ACT_OSAMP == 10


@pytest.mark.parametrize("pc", [False, True])
def test_matching_pixels_pass(pc):
    """30" ACT pixels on a 3" map pass, whatever the RA sign and however the scale is stored."""
    check_act_pixels(
        [car_wcs(30.0, +1.0, pc), car_wcs(30.0, -1.0)], car_wcs(3.0, -1.0, pc)
    )


@pytest.mark.parametrize("map_pix", [4.0, 6.0, 2.0])
def test_mismatched_pixels_raise(map_pix):
    """Any other map pixel is refused, and the message names the pixel size to use."""
    with pytest.raises(ValueError, match=r"--pixsize 3\b"):
        check_act_pixels([car_wcs(30.0, +1.0)], car_wcs(map_pix))


def test_one_bad_map_raises():
    """A single ACT map with other pixels is enough to refuse."""
    with pytest.raises(ValueError, match=r'ACT pixels are 60" x 60"'):
        check_act_pixels([car_wcs(30.0, +1.0), car_wcs(60.0, +1.0)], car_wcs(3.0))


def test_not_strict_warns(capsys):
    """With strict=False (--debug) a mismatch is printed, not raised."""
    check_act_pixels([car_wcs(30.0, +1.0)], car_wcs(6.0), strict=False)
    assert "WARNING" in capsys.readouterr().out


def test_no_act_maps_pass():
    """Without ACT maps there is nothing to check."""
    check_act_pixels([], car_wcs(4.0))
