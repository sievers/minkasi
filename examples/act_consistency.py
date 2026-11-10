"""
Does a joint M2 + ACT map agree with ACT where both measure? The joint map (twores_ex _final.fits) is seen as ACT sees it
(ACT beam, normalised to the M2 footprint, then averaged over each ACT pixel) and compared with the ACT coadd in the ACT
pixels fully inside the region where M2 has at least --min-hits hits, in units of the ACT noise per pixel. Prints the
rms of that difference after removing an offset (1 = consistent with ACT's white noise), its tilt and bowl at the edge of
the region, and the rms left after removing them. With --sim-sky the ACT map is simulated from the sky with the real ACT
noise, as a check of the method.
"""

import argparse

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from multiscale_map import fine_grid, read_act_coadd, sample, sample_array
from twores_ex import act_conversion

FWHM2SIGMA = 1 / np.sqrt(8 * np.log(2))


def act_view(
    m: np.ndarray, good: np.ndarray, shape: tuple[int, int], f: int, sig: float
) -> np.ndarray:
    """
    A fine-grid map as ACT sees it: smoothed with the ACT beam inside the footprint, then averaged over ACT pixels.

    Parameters
    ----------
    m : np.ndarray
        Map on the fine grid.
    good : np.ndarray
        Footprint on the fine grid; the smoothing is normalised to it.
    shape : tuple[int, int]
        Shape of the ACT map.
    f : int
        Sub-pixels per ACT pixel.
    sig : float
        Beam sigma in fine pixels.

    Returns
    -------
    np.ndarray
        Values per ACT pixel.
    """
    sm = gaussian_filter(m * good, sig) / np.maximum(
        gaussian_filter(good.astype(float), sig), 1e-3
    )
    return sm.reshape(shape[0], f, shape[1], f).mean((1, 3))


def fit_shapes(
    d: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray, bowl: bool
) -> tuple[np.ndarray, float]:
    """
    Least-squares fit of an offset, a plane and optionally a bowl (r^2).

    Parameters
    ----------
    d : np.ndarray
        Data.
    x, y : np.ndarray
        Coordinates scaled to 1 at the edge of the region.
    w : np.ndarray
        Noise weights (1 / sigma) for the residual rms.
    bowl : bool
        Include r^2.

    Returns
    -------
    coef : np.ndarray
        Offset, x slope, y slope (and bowl), in the units of d at the edge.
    rms : float
        Weighted rms of the residual.
    """
    cols = [np.ones_like(x), x, y] + ([x**2 + y**2] if bowl else [])
    a = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(a, d, rcond=None)
    return coef, float(((d - a @ coef) * w).std())


def main(args: argparse.Namespace):
    """
    Print the comparison.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    root = args.root
    hw = WCS(fits.getheader(f"{root}_hits.fits")).celestial
    ny, nx = fits.getdata(f"{root}_hits.fits").shape
    ra, dec = map(float, hw.wcs_pix2world((nx - 1) / 2, (ny - 1) / 2, 0))
    act, ivar = read_act_coadd(
        args.act_glob, ra, dec, args.size, act_conversion(args.freq)
    )
    f = args.factor
    fw, shape, _ = fine_grid(act, f)
    pix = abs(fw.wcs.cdelt[0]) * 3600
    hits = sample(f"{root}_hits.fits", fw, shape, 0)
    good = hits >= 20
    sig = args.act_beam * 60 * FWHM2SIGMA / pix
    jv = act_view(sample(f"{root}_final.fits", fw, shape, 1), good, act.shape, f, sig)
    a = np.asarray(act)
    if (
        args.sim_sky
    ):  # simulated ACT: the sky as ACT sees it plus white noise at the real level
        sky, sw = (
            fits.getdata(args.sim_sky).astype(float),
            WCS(fits.getheader(args.sim_sky)).celestial,
        )
        a = act_view(sample_array(sky, sw, fw, shape, 1), good, act.shape, f, sig)
        a = a + np.random.default_rng(args.seed).normal(size=a.shape) / np.sqrt(ivar)
    inm = (hits >= args.min_hits).reshape(act.shape[0], f, act.shape[1], f).all((1, 3))
    yy, xx = np.indices(inm.shape).astype(float)
    cy, cx = yy[inm].mean(), xx[inm].mean()
    r = np.sqrt(((yy - cy) ** 2 + (xx - cx) ** 2)[inm].max())
    x, y = ((xx - cx) / r)[inm], ((yy - cy) / r)[inm]
    d, w = (jv - a)[inm], np.sqrt(ivar[inm])
    plane, rms_plane = fit_shapes(d, x, y, w, False)
    both, rms_both = fit_shapes(d, x, y, w, True)
    print(
        f"{args.label}: {inm.sum()} ACT pixels, ACT noise {np.median(1 / w) * 1e6:.1f} uK/pixel; "
        f"(joint seen by ACT - ACT) / noise: rms {((d - np.median(d)) * w).std():.2f} after an offset, "
        f"{rms_plane:.2f} after a plane, {rms_both:.2f} after a plane and a bowl"
    )
    print(
        f"  tilt at the edge: x {plane[1] * 1e6:+.0f} uK, y {plane[2] * 1e6:+.0f} uK, "
        f"|tilt| {np.hypot(plane[1], plane[2]) * 1e6:.0f} uK; bowl with the plane {both[3] * 1e6:+.0f} uK"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="twores_ex outroot (uses _final.fits and _hits.fits)",
    )
    parser.add_argument("--label", default="map")
    parser.add_argument(
        "--act-glob",
        default="/mnt/welch/USERS/jorlo/maps/dr6_20230316_splits/cmb_night_*_f090_3pass_4way_set*_ivar.fits",
    )
    parser.add_argument("--act-beam", type=float, default=2.1, help="arcmin")
    parser.add_argument("--freq", type=float, default=90e9)
    parser.add_argument("--size", type=float, default=18.0, help="arcmin")
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument(
        "--min-hits",
        type=float,
        default=120,
        help="ACT pixels compared: all sub-pixels this many hits",
    )
    parser.add_argument(
        "--sim-sky",
        default=None,
        help="Simulate ACT from this sky instead of using the ACT maps",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="One ACT split set")
    args = parser.parse_args()
    if args.debug:
        args.act_glob = args.act_glob.replace("set*", "set0")
    main(args)
