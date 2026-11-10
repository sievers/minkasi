"""
Where the simulation's map error lives, per needlet band: the error (map - truth) of saved CG maps split into the
bands (component F_i^2 m of band i, of the error inside the footprint; the saved maps are zero outside), measured in
the footprint interior and in the edge strip inside the footprint. Also the share of each band in the error of the 16-64 px Gaussian band-pass at the edge
(Figure 6 of the summary page), and the prior strength a signal prior would give for the bands the outside prior
acts on: eps = 1 / (sky variance per pixel of the sum of those bands), from the simulated sky components.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from minkasi.needlet.needlet import Needlet
from scipy.ndimage import distance_transform_edt

from sim_compare import bandpass, load_map


def filters2(n: int, pixsize: float, nfilt: int) -> np.ndarray:
    """
    Squared needlet filters of an n x n map in numpy FFT order, built as in twores_ex.setup_wmap.

    Parameters
    ----------
    n : int
        Map side in pixels.
    pixsize : float
        Pixel size in arcsec.
    nfilt : int
        Number of filters.

    Returns
    -------
    np.ndarray
        Shape (nfilt, n, n); they sum to 1 over the filters.
    """
    need = Needlet(
        np.arange(nfilt),
        lightcone=np.zeros((n, n)),
        L=n * pixsize * np.sqrt(2),
        pixsize=pixsize,
    )
    need.get_needlet_filters_2d(
        need.lightcone_box.get_grid_dimless_2d(return_grid=True)
    )
    return np.fft.ifftshift(need.filters**2, axes=(1, 2))


def split(m: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """
    Band components F_i^2 m of a map.

    Parameters
    ----------
    m : np.ndarray
        Map.
    f2 : np.ndarray
        Output of filters2.

    Returns
    -------
    np.ndarray
        Shape (nfilt, n, n), summing to m.
    """
    return np.fft.ifft2(f2 * np.fft.fft2(m)[None]).real


def main(args: argparse.Namespace):
    """
    Print the tables and make the figure.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    good = load_map(f"{args.sim}/coef/MOOJ1142_hits.fits") >= args.min_hits
    inside, outside = distance_transform_edt(good), distance_transform_edt(~good)
    regions = {
        "interior": good & (inside > args.edge),
        "edge in": good & (inside <= args.edge),
    }
    truth = load_map(f"{args.sim}/coef/MOOJ1142_truth.fits")
    f2 = filters2(truth.shape[0], args.pixsize, args.nfilt)
    tb = split(truth, f2)
    runs = {
        r: np.load(f"{args.sim}/{r.split('@')[0]}/MOOJ1142_cg{r.split('@')[1]}.npy")
        for r in args.runs
    }

    print(
        "rms of the error per band, in units of the truth's rms in that band inside the footprint"
    )
    print("run          region     " + " ".join(f"{i:>7d}" for i in range(args.nfilt)))
    eb = {}
    for r, m in runs.items():
        eb[r] = split(
            (m - truth) * good, f2
        )  # saved maps are zero outside the footprint
        for name, sel in regions.items():
            row = [eb[r][i][sel].std() / tb[i][good].std() for i in range(args.nfilt)]
            print(f"{r:12s} {name:10s} " + " ".join(f"{v:7.2f}" for v in row))

    print(
        "\nshare of each band in the squared error of the 16-64 px band-pass, edge strip inside the footprint"
    )
    sel = regions["edge in"]
    for r, m in runs.items():
        tot = bandpass((m - truth) * good, good, 16, 64)
        share = [
            (bandpass(eb[r][i], good, 16, 64) * tot)[sel].sum() / (tot[sel] ** 2).sum()
            for i in range(args.nfilt)
        ]
        print(f"{r:12s} " + " ".join(f"{v:7.2f}" for v in share))

    # signal prior for the bands the outside prior acts on, from the simulated sky components (own 800 x 800 grid)
    print(
        f"\nsignal prior for bands {args.prior_bands[0]}-{args.prior_bands[-1]}: eps = 1 / sky variance per pixel"
    )
    with fits.open(args.sky) as hdul:
        comps = {h.name: h.data.astype(float) for h in hdul[1:]}
    fs = filters2(next(iter(comps.values())).shape[0], args.pixsize, args.nfilt)
    total = 0.0
    for name, c in comps.items():
        fine = split(c, fs)[args.prior_bands].sum(0)
        var = fine.var()
        total += var
        print(f"  {name:8s} rms {np.sqrt(var):.2e} K_RJ  eps {1 / var:.2e}")
    fine_t = tb[args.prior_bands].sum(0)
    print(
        f"  all      rms {np.sqrt(total):.2e} K_RJ  eps {1 / total:.2e}  (whole sky map)"
    )
    edge_out = ~good & (outside <= args.edge)
    print(
        f"  truth just outside the footprint: rms {fine_t[edge_out].std():.2e} K_RJ, "
        f"eps {1 / fine_t[edge_out].var():.2e}"
    )
    print(
        f"  used: eps {args.eps_used:.2e} (1e-2 of the M2 pixel weight); M2 pixel weight {args.eps_used / 1e-2:.2e}, "
        f"i.e. M2 noise {np.sqrt(1e-2 / args.eps_used):.2e} K_RJ per pixel"
    )

    # figure: error per band for the runs, cropped to the footprint plus the outer strip
    ii, jj = np.where(good)
    box = (slice(ii.min(), ii.max() + 1), slice(jj.min(), jj.max() + 1))
    bands = list(range(args.nfilt - 2))
    fig, ax = plt.subplots(
        len(runs),
        len(bands),
        figsize=(2.1 * len(bands), 2.3 * len(runs) + 0.6),
        squeeze=False,
    )
    for j, i in enumerate(bands):
        lim = max(np.abs(eb[r][i][good]).max() for r in runs)
        for k, r in enumerate(runs):
            a = ax[k, j]
            a.imshow(
                eb[r][i][box].T * 1e6,
                origin="lower",
                cmap="RdBu_r",
                vmin=-lim * 1e6,
                vmax=lim * 1e6,
            )
            a.contour(good[box].T.astype(float), [0.5], colors="k", linewidths=0.6)
            a.set_xticks([])
            a.set_yticks([])
            if k == 0:
                a.set_title(f"band {i}\n±{lim * 1e6:.0f} µK", fontsize=9)
            if j == 0:
                a.set_ylabel(r.replace("@", ", it "), fontsize=9)
    fig.suptitle(
        "map minus truth, split by needlet band (black: footprint edge)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print("wrote", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", default="/mnt/welch/USERS/jorlo/needlets/sim")
    parser.add_argument(
        "--sky", default="/mnt/welch/USERS/jorlo/needlets/sim/sky_s0.fits"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["coef@40", "coef@600", "prior@600"],
        help="run@iteration",
    )
    parser.add_argument("--nfilt", type=int, default=10)
    parser.add_argument("--pixsize", type=float, default=3.0)
    parser.add_argument("--min-hits", type=float, default=20)
    parser.add_argument(
        "--edge", type=float, default=20, help="Width of the edge strips, pixels"
    )
    parser.add_argument(
        "--prior-bands", type=int, nargs="+", default=list(range(4, 10))
    )
    parser.add_argument(
        "--eps-used", type=float, default=2.12e5, help="eps of the prior run (its log)"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument(
        "--debug", action="store_true", help="One run, low resolution figure"
    )
    args = parser.parse_args()
    if args.debug:
        args.runs, args.dpi = args.runs[-1:], 60
    main(args)
