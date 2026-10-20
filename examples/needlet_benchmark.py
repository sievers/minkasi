"""
Benchmark and accuracy check of the probed (sparse) ANA against the exact one, with figures for reports/needlet.html.
"""

import argparse
import json
import os
import time

import minkasi.minkasi_all as minkasi
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from twores_ex import read_tods, setup_wmap


def plot_maps(files, out, smooth=None):
    """
    Plot FITS maps side by side, titled by the suffix of the file name (e.g. need, act, combined).

    Parameters
    ----------
    files : list[str]
        FITS maps to plot.
    out : str
        Output figure path.
    smooth : dict | None
        Gaussian smoothing (pixels) by file name suffix, e.g. {"need": 8}. Default is no smoothing.
    """
    smooth = {} if smooth is None else smooth
    fig, axs = plt.subplots(1, len(files), figsize=(5 * len(files), 4.5), squeeze=False)
    for ax, f in zip(axs[0], files):
        name = os.path.basename(f).rsplit("_", 1)[-1][:-5]
        sig = smooth.get(name, 0)
        im = ax.imshow(
            gaussian_filter(fits.getdata(f), sig) if sig else fits.getdata(f),
            origin="lower",
        )
        ax.set_title(name + (f" (smoothed {sig} px)" if sig else ""))
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main(args):
    """
    Time exact and probed ANA for a few filters and save figures and a json summary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    os.makedirs(args.odir, exist_ok=True)
    todvec = read_tods(args.idir, 1)
    pix = args.pixsize
    wmap = setup_wmap(todvec, pix, args.nfilt)
    need = wmap.needlet
    for tod in todvec.tods:
        tod.info["ipix"] = wmap.get_pix(tod)
        tod.set_noise(minkasi.NoiseSmoothedSVD)
    wmap.get_downsamps()
    need.plot_bands()
    plt.savefig(f"{args.odir}/bands.png", dpi=120)
    plt.close("all")

    res = {
        "downsamps": wmap.downsamps.tolist(),
        "shape": [wmap.nx, wmap.ny],
        "filters": {},
    }
    fig, axs = plt.subplots(
        len(args.filters), 3, figsize=(10, 3.3 * len(args.filters)), squeeze=False
    )
    for row, i in enumerate(args.filters):
        n = int(wmap.nxs_red[i] * wmap.nys_red[i])
        t0 = time.time()
        exact = wmap.get_response_matrix(todvec, nfilts=[i])[i]
        t_exact = time.time() - t0
        info = {"npix_red": n, "t_exact": t_exact, "radius": {}}
        for r in args.radii:
            t0 = time.time()
            approx = wmap.get_response_matrix(
                todvec, nfilts=[i], radius=r, exact_below=0
            )[i]
            dt = time.time() - t0
            err = abs(approx - exact).max() / abs(exact).max()
            info["radius"][r] = {
                "t": dt,
                "max_rel_err": float(err),
                "nnz_frac": approx.nnz / n**2,
            }
            last = approx
        res["filters"][i] = info
        e, a = exact.toarray(), last.toarray()
        c = n // 2
        side = int(wmap.nys_red[i])
        axs[row, 0].imshow(e[c].reshape(wmap.nxs_red[i], side), origin="lower")
        axs[row, 0].set_title(f"filter {i}: exact row")
        axs[row, 1].imshow(a[c].reshape(wmap.nxs_red[i], side), origin="lower")
        axs[row, 1].set_title(f"radius={args.radii[-1]}")
        axs[row, 2].imshow((a[c] - e[c]).reshape(wmap.nxs_red[i], side), origin="lower")
        axs[row, 2].set_title("difference")
    fig.tight_layout()
    fig.savefig(f"{args.odir}/rows.png", dpi=120)
    if args.maps:
        plot_maps(args.maps, f"{args.odir}/maps.png")
    with open(f"{args.odir}/summary.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--idir", default="/mnt/welch/MUSTANG/M2-TODs/MOOJ1142/mustang2/"
    )
    parser.add_argument("--odir", default="../reports/figs")
    parser.add_argument("--pixsize", type=float, default=6.0)
    parser.add_argument("--nfilt", type=int, default=6)
    parser.add_argument("--filters", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--radii", type=float, nargs="+", default=[10.0, 20.0])
    parser.add_argument("--maps", nargs="*", default=[], help="FITS maps to plot")
    parser.add_argument("--maps-only", action="store_true", help="Only plot --maps")
    parser.add_argument("--debug", action="store_true", help="Only the coarsest filter")
    args = parser.parse_args()
    if args.debug:
        args.filters, args.radii = [2], [10.0]
    if args.maps_only:
        os.makedirs(args.odir, exist_ok=True)
        plot_maps(args.maps, f"{args.odir}/maps.png")
    else:
        main(args)
