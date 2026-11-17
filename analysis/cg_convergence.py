"""
Plot how much the needlet CG map still changes between saved iterations, per angular scale, relative to
the rms of the last saved map in that band (maps saved by twores_ex.py --cg-save).
"""

import argparse

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

BANDS = [(0.01, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)]


def band_rms(x: np.ndarray, good: np.ndarray, lo: float, hi: float) -> float:
    """
    rms in the footprint of a Gaussian band-pass, smoothing normalised by the footprint.

    Parameters
    ----------
    x : np.ndarray
        Map.
    good : np.ndarray
        Boolean footprint mask.
    lo, hi : float
        Gaussian sigmas in pixels; the band is smooth(lo) - smooth(hi).

    Returns
    -------
    rms : float
    """
    w = good.astype(float)

    def smooth(s):
        return gaussian_filter(np.where(good, x, 0), s) / np.maximum(
            gaussian_filter(w, s), 1e-3
        )

    return (smooth(lo) - smooth(hi))[good].std()


def main(args: argparse.Namespace):
    """
    Make the figure and print the table.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    # minkasi writes FITS maps transposed with respect to the saved .npy arrays
    good = fits.getdata(args.hits).squeeze().T >= args.min_hits
    maps = {k: np.load(f"{args.root}_cg{k}.npy") for k in args.iters}
    last = args.iters[-1]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=70 if args.debug else 120)
    x = np.arange(len(BANDS))
    for a, b in zip(args.iters[:-1], args.iters[1:]):
        y = [
            band_rms(maps[b] - maps[a], good, lo, hi)
            / band_rms(maps[last], good, lo, hi)
            for lo, hi in BANDS
        ]
        ax.plot(x, y, "o-", label=f"{a} to {b}")
        print(f"{a}->{b}:", " ".join(f"{v:.2f}" for v in y))
    ax.set_xticks(x, [f"{lo:g}-{hi:g}" for lo, hi in BANDS])
    ax.set(
        xlabel="Gaussian band-pass sigma (pixels)",
        ylabel=f"rms change / rms of iteration {last} map",
        yscale="log",
    )
    ax.legend(title="iterations")
    fig.tight_layout()
    fig.savefig(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Prefix of the _cg<N>.npy maps")
    parser.add_argument("--hits", required=True, help="Hits map FITS")
    parser.add_argument(
        "--iters", type=int, nargs="+", default=[10, 20, 30, 40, 60, 80]
    )
    parser.add_argument("--min-hits", type=float, default=20)
    parser.add_argument("--out", required=True, help="Output figure")
    parser.add_argument("--debug", action="store_true", help="Lower resolution figure")
    main(parser.parse_args())
