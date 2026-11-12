"""
Two-resolution display maps (multiscale_map.py) of a simulation: truth, the simulated data and their difference,
for one or more twores_ex runs on the simulated sky. Truth and data go through the same display steps, so the
residual compares like with like:
  - truth: the simulated sky (make_sim_sky.py) at sub-pixel resolution where M2 has data, and outside the sky as a
    noiseless ACT sees it (ACT beam, then averaged over each ACT pixel);
  - data: the run's joint map where M2 has data, and outside a simulated ACT map, the same ACT view of the sky plus
    white noise from the real ACT inverse variance (as twores_ex.sim_act does on the mapmaking grid).
For visualization only.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import Normalize
from minkasi.needlet.act import act_conversion
from scipy.ndimage import gaussian_filter

from multiscale_map import (
    caption,
    compose,
    draw,
    fine_grid,
    make_norm,
    read_act_coadd,
    sample,
    sample_array,
    ticks,
)

FWHM2SIGMA = 1 / np.sqrt(8 * np.log(2))


def act_view(sky: np.ndarray, sky_wcs: WCS, act, args) -> np.ndarray:
    """
    The sky as a noiseless ACT sees it: smoothed with the ACT beam, then averaged over each ACT pixel.

    Parameters
    ----------
    sky : np.ndarray
        Sky image, [y, x].
    sky_wcs : WCS
        Its WCS.
    act : enmap.ndmap
        Map on the ACT pixelization, for the geometry.
    args : argparse.Namespace
        Parsed arguments (act_beam, factor).

    Returns
    -------
    np.ndarray
        Values per ACT pixel.
    """
    pix = abs(sky_wcs.wcs.cdelt[0]) * 3600
    smooth = gaussian_filter(sky, args.act_beam * 60 * FWHM2SIGMA / pix, mode="nearest")
    wcs, shape, _ = fine_grid(act, args.factor)
    fine = sample_array(smooth, sky_wcs, wcs, shape, 1)
    f = args.factor
    return fine.reshape(act.shape[0], f, act.shape[1], f).mean((1, 3))


def main(args: argparse.Namespace):
    """
    Make the figure.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    with fits.open(args.sky) as hdul:
        sky, sky_wcs = hdul[0].data.astype(float), WCS(hdul[0].header).celestial
    ra, dec = sky_wcs.wcs.crval
    conv = act_conversion(args.freq, unit="K_RJ")
    act, ivar = read_act_coadd(args.act_glob, float(ra), float(dec), args.size, conv)
    truth_act = act_view(sky, sky_wcs, act, args)
    noise = np.random.default_rng(args.seed).normal(size=act.shape)
    data_act = truth_act + np.divide(
        noise, np.sqrt(ivar), out=np.zeros(act.shape), where=ivar > 0
    )

    wcs, shape, _ = fine_grid(act, args.factor)
    pix = abs(wcs.wcs.cdelt[0]) * 3600
    up = lambda m: np.repeat(np.repeat(m, args.factor, 0), args.factor, 1)
    truth_fine = sample_array(sky, sky_wcs, wcs, shape, 1)
    rows = []
    for run in args.runs:
        label, root = run.split("=", 1)
        hits = sample(f"{root}_hits.fits", wcs, shape, 0)
        truth, split, m2 = compose(args, up(truth_act), truth_fine, hits, pix)
        data = compose(
            args, up(data_act), sample(f"{root}_final.fits", wcs, shape, 1), hits, pix
        )[0]
        res = data - truth
        if (
            args.res_smooth > 0
        ):  # smooth the M2 part of the residual further, normalised to the M2 region
            sig = args.res_smooth * FWHM2SIGMA / pix
            res = np.where(
                m2,
                gaussian_filter(res * m2, sig)
                / np.maximum(gaussian_filter(m2.astype(float), sig), 1e-3),
                res,
            )
        print(
            f"{label}: residual rms {res[m2].std() * 1e6:.1f} uK where M2 shows, "
            f"{res[~split].std() * 1e6:.1f} uK in the ACT pixels"
        )
        rows.append((label, truth, data, res, m2))

    # one scale for truth and data, from both, so that the noise in the data stays inside it
    norm = make_norm(np.concatenate([np.concatenate([r[1], r[2]]) for r in rows]), args)
    rlim = (
        args.res_lim
        or np.percentile(np.abs(np.concatenate([r[3].ravel() for r in rows])), 99.5)
        * 1e6
    )
    rnorm = Normalize(-rlim, rlim)
    fig = plt.figure(figsize=(17, 5.4 * len(rows)))
    for k, (label, truth, data, res, m2) in enumerate(rows):
        for j, (img, nm, title) in enumerate(
            (
                (truth, norm, "truth"),
                (data, norm, f"simulated data, {label}"),
                (
                    res,
                    rnorm,
                    f"data minus truth, {label}",
                ),
            )
        ):
            ax = fig.add_subplot(len(rows), 3, 3 * k + j + 1, projection=wcs)
            im = draw(ax, img, m2, nm, args.cmap, args.outline, args.grid)
            ax.set_title(title, fontsize=11)
            if j > 0:
                ax.coords[1].set_ticklabel_visible(False)
                ax.coords[1].set_axislabel("")
            fig.colorbar(
                im, ax=ax, label=r"$\mu$K$_{\rm RJ}$", shrink=0.8, ticks=ticks(nm)
            )
    print("caption:", caption(args, pix))
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    S = "/mnt/welch/USERS/jorlo/needlets/sim"
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sky", default=f"{S}/sky_s0.fits", help="Simulated sky (make_sim_sky.py)"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[f"needlet CG={S}/coef/MOOJ1142", f"outside prior={S}/prior/MOOJ1142"],
        help="label=outroot of twores_ex runs on the simulated sky (uses _final.fits and _hits.fits)",
    )
    parser.add_argument(
        "--act-glob",
        default="/mnt/welch/USERS/jorlo/maps/dr6_20230316_splits/cmb_night_*_f090_3pass_4way_set*_ivar.fits",
        help="ACT inverse variance maps, for the geometry and the noise level",
    )
    parser.add_argument(
        "--act-beam", type=float, default=2.1, help="ACT beam FWHM, arcmin"
    )
    parser.add_argument("--freq", type=float, default=90e9)
    parser.add_argument("--size", type=float, default=18.0, help="arcmin")
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument(
        "--min-hits", type=float, default=20, help="Hits for an ACT pixel to be split"
    )
    parser.add_argument(
        "--show-hits",
        type=float,
        default=None,
        help="Hits for a sub-pixel to show the joint map",
    )
    parser.add_argument("--smooth", type=float, default=10.0, help="arcsec")
    parser.add_argument(
        "--smooth-ref-hits",
        type=float,
        default=None,
        help="Below this many hits the smoothing FWHM grows as sqrt(ref / hits), up to --smooth-max (default: fixed FWHM)",
    )
    parser.add_argument(
        "--smooth-max",
        type=float,
        default=30.0,
        help="arcsec, largest FWHM with --smooth-ref-hits",
    )
    parser.add_argument("--feather-hits", type=float, default=200.0)
    parser.add_argument(
        "--res-smooth",
        type=float,
        default=30.0,
        help="arcsec, extra smoothing of the residual's M2 part",
    )
    parser.add_argument("--seed", type=int, default=0, help="ACT noise")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument(
        "--res-lim", type=float, default=None, help="uK, residual colour range"
    )
    parser.add_argument("--stretch", choices=["twoslope", "linear"], default="twoslope")
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--outline", action="store_true")
    parser.add_argument("--grid", action="store_true", help="Draw the RA/Dec grid")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument(
        "--debug", action="store_true", help="One ACT split set, 12 arcmin, one run"
    )
    args = parser.parse_args()
    args.match = "none"  # truth and data are compared as they are
    if args.debug:
        args.act_glob = args.act_glob.replace("set*", "set0")
        args.size, args.dpi, args.runs = 12.0, 60, args.runs[:1]
    main(args)
