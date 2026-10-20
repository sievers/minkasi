"""
Benchmark and accuracy check of the probed (sparse) ANA against the exact one, with figures for reports/needlet.html.
"""

import argparse
import json
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits

import minkasi.minkasi_all as minkasi
from minkasi.maps.skymap import SkyMap
from minkasi.needlet.needlet import Needlet, WavSkyMap
from twores_ex import read_tods


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
    ps = pix / 3600 * np.pi / 180
    lims = todvec.lims()
    box = SkyMap(lims, ps, square=True, multiple=2).map
    need = Needlet(
        np.arange(args.nfilt),
        lightcone=box,
        L=box.shape[0] * pix * np.sqrt(2),
        pixsize=pix,
    )
    need.get_needlet_filters_2d(
        need.lightcone_box.get_grid_dimless_2d(return_grid=True)
    )
    wmap = WavSkyMap(need, lims, ps, square=True, multiple=2)
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
            approx = wmap.get_response_matrix(todvec, nfilts=[i], radius=r)[i]
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
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, (f, t) in zip(
            axs, zip(args.maps, ["PCG (M2 only)", "combined M2+ACT"])
        ):
            ax.imshow(fits.getdata(f), origin="lower")
            ax.set_title(t)
        fig.savefig(f"{args.odir}/maps.png", dpi=120)
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
    parser.add_argument(
        "--maps", nargs="*", default=[], help="PCG and combined FITS maps to plot"
    )
    parser.add_argument("--debug", action="store_true", help="Only the coarsest filter")
    args = parser.parse_args()
    if args.debug:
        args.filters, args.radii = [2], [10.0]
    main(args)
