"""
Compare final maps band by band: rms of each map and of its difference with the first map in Gaussian
band-passes, plus the central decrement, e.g. the ACT weight variants of twores_ex.py.
"""

import argparse

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from cg_convergence import BANDS, band_rms


def load(path: str) -> np.ndarray:
    """
    Read a 2D FITS map, NaN set to 0.

    Parameters
    ----------
    path : str
        FITS file.

    Returns
    -------
    np.ndarray
        The map, shape (ny, nx).
    """
    return np.nan_to_num(fits.getdata(path).squeeze().astype(float))


def main(args: argparse.Namespace):
    """
    Print the tables.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    hits = load(args.hits)
    good = hits >= args.min_hits
    maps = dict(zip(args.labels, map(load, args.maps)))
    ref = maps[args.labels[0]]
    head = "  ".join(f"{lo:g}-{hi:g}".rjust(7) for lo, hi in BANDS)
    print(f"{'band sigma (px)':28}{head}")
    for k, m in maps.items():
        print(
            f"{'rms ' + k:28}"
            + "  ".join(f"{band_rms(m, good, lo, hi):7.1e}" for lo, hi in BANDS)
        )
    for k, m in list(maps.items())[1:]:
        print(
            f"{'rms diff/rms ' + args.labels[0]:28}"
            + "  ".join(
                f"{band_rms(m - ref, good, lo, hi) / band_rms(ref, good, lo, hi):7.2f}"
                for lo, hi in BANDS
            )
            + f"   ({k})"
        )
    yy, xx = np.indices(hits.shape)
    cy, cx = (np.average(a, weights=hits) for a in (yy, xx))
    near = np.hypot(yy - cy, xx - cx) < args.radius
    for k, m in maps.items():
        smooth = gaussian_filter(m, args.smooth) * good
        print(f"{k}: decrement {smooth[near].min():.2e}, pixel rms {m[good].std():.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", nargs="+", required=True, help="Final map FITS files")
    parser.add_argument("--labels", nargs="+", required=True, help="One label per map")
    parser.add_argument("--hits", required=True, help="Hits map FITS")
    parser.add_argument("--min-hits", type=float, default=20)
    parser.add_argument(
        "--radius", type=float, default=60, help="Pixels around the centre"
    )
    parser.add_argument("--smooth", type=float, default=4, help="Smoothing, pixels")
    main(parser.parse_args())
