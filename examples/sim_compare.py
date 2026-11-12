"""
Compare twores_ex.py solutions of a simulation (--sim-sky) with the input sky: for each run and saved CG
iteration, the error and the transfer (recovered over input) in Gaussian band-passes inside the footprint,
split into the edge and the interior, and the central decrement.
"""

import argparse

import numpy as np
from astropy.io import fits
from scipy.ndimage import distance_transform_edt, gaussian_filter

from cg_convergence import BANDS


def load_map(path: str) -> np.ndarray:
    """
    Read a map written by minkasi in the orientation of the saved .npy arrays (FITS maps are transposed).

    Parameters
    ----------
    path : str
        FITS file.

    Returns
    -------
    m : np.ndarray
    """
    return fits.getdata(path).squeeze().T.astype(float)


def bandpass(x: np.ndarray, good: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Gaussian band-pass of a map inside a footprint, smoothing normalised by the footprint.

    Parameters
    ----------
    x : np.ndarray
        Map.
    good : np.ndarray
        Boolean footprint.
    lo, hi : float
        Gaussian sigmas in pixels; the band is smooth(lo) - smooth(hi).

    Returns
    -------
    b : np.ndarray
    """
    w = (
        np.maximum(gaussian_filter(good.astype(float), lo), 1e-3),
        np.maximum(gaussian_filter(good.astype(float), hi), 1e-3),
    )
    xm = np.where(good, x, 0)
    return gaussian_filter(xm, lo) / w[0] - gaussian_filter(xm, hi) / w[1]


def main(args: argparse.Namespace):
    """
    Print the tables.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    good = load_map(args.hits) >= args.min_hits
    dist = distance_transform_edt(good)
    edge, inner = good & (dist <= args.edge), good & (dist > args.edge)
    truth = load_map(args.truth)
    cut = np.zeros_like(good)
    c = np.array(good.shape) // 2
    r = int(args.radius)
    cut[c[0] - r : c[0] + r, c[1] - r : c[1] + r] = True

    def decrement(m):
        sm = gaussian_filter(np.where(good, m, 0), 4) / np.maximum(
            gaussian_filter(good.astype(float), 4), 1e-3
        )
        return sm[good & cut].min()

    tb = {b: bandpass(truth, good, *b) for b in BANDS}
    head = " ".join(f"{lo:g}-{hi:g}".rjust(11) for lo, hi in BANDS)
    print(f"truth decrement {decrement(truth):.3e}")
    print(f"band sigma (px)            {head}")
    print(
        "rms truth                  "
        + " ".join(f"{tb[b][good].std():11.2e}" for b in BANDS)
    )
    for run in args.runs:
        label, root = run.split("=", 1)
        for k in args.iters:
            try:
                m = np.load(f"{root}_cg{k}.npy")
            except FileNotFoundError:
                continue
            print(f"== {label}, iteration {k}: decrement {decrement(m):.3e}")
            for name, sel in (("all", good), ("edge", edge), ("inner", inner)):
                err, tr = [], []
                for b in BANDS:
                    mb = bandpass(m, good, *b)
                    err.append((mb - tb[b])[sel].std() / tb[b][sel].std())
                    tr.append((mb * tb[b])[sel].sum() / (tb[b][sel] ** 2).sum())
                print(
                    f"  error/truth {name:5s}       "
                    + " ".join(f"{e:11.2f}" for e in err)
                )
                print(
                    f"  transfer    {name:5s}       "
                    + " ".join(f"{t:11.2f}" for t in tr)
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth", required=True, help="Truth map FITS (twores_ex _truth.fits)"
    )
    parser.add_argument("--hits", required=True, help="Hits map FITS")
    parser.add_argument(
        "--runs", nargs="+", required=True, help="label=prefix of the _cg<N>.npy maps"
    )
    parser.add_argument("--iters", type=int, nargs="+", default=[40, 200, 400, 600])
    parser.add_argument("--min-hits", type=float, default=20)
    parser.add_argument(
        "--edge", type=float, default=20, help="Width of the edge region, pixels"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=60,
        help="Half side of the box for the decrement, pixels",
    )
    parser.add_argument("--debug", action="store_true", help="Only the last iteration")
    args = parser.parse_args()
    if args.debug:
        args.iters = args.iters[-1:]
    main(args)
