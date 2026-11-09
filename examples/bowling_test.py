"""
Test whether large scale "bowling" (radial gradient / saddle shape) builds up with CG iterations. A
quadratic surface is fitted to each saved map (maps of twores_ex.py --cg-save) in the footprint, outside
the central cluster, and the amplitude of the fitted surface is followed against the iteration for each
run, e.g. with and without ACT. Also shows the maps smoothed to a few arcmin.
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # no display needed
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

FWHM2SIGMA = 1 / np.sqrt(8 * np.log(2))


def design(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Quadratic design matrix on a pixel grid centred on the map, with x, y in units of the half width.

    Parameters
    ----------
    shape : tuple[int, int]
        Map shape (ny, nx).

    Returns
    -------
    basis : np.ndarray
        Shape (6, ny, nx): 1, x, y, x^2, x y, y^2.
    r : np.ndarray
        Radius in pixels, shape (ny, nx).
    """
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    yy, xx = yy - ny / 2, xx - nx / 2
    half = max(nx, ny) / 2
    x, y = xx / half, yy / half
    return np.array([np.ones_like(x), x, y, x**2, x * y, y**2]), np.hypot(xx, yy)


def fit_quadratic(
    m: np.ndarray, basis: np.ndarray, use: np.ndarray, good: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Least squares quadratic fit to the pixels in use.

    Parameters
    ----------
    m : np.ndarray
        Map.
    basis : np.ndarray
        Design matrix from design.
    use : np.ndarray
        Boolean mask of pixels to fit.
    good : np.ndarray
        Boolean footprint, over which the amplitude is measured.

    Returns
    -------
    coef : np.ndarray
        Six coefficients.
    amp : np.ndarray
        (rms over the footprint of the fitted surface, bowl: its rise from the centre to the footprint edge).
    """
    coef = np.linalg.lstsq(basis[:, use].T, m[use], rcond=None)[0]
    surf = np.tensordot(coef, basis, axes=1)
    return coef, np.array([surf[good].std(), coef[3] + coef[5]])


def main(args: argparse.Namespace):
    """
    Make the figures and print the table.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    good = fits.getdata(args.hits).squeeze() >= args.min_hits
    basis, r = design(good.shape)
    use = good & (r > args.exclude)
    runs = dict(zip(args.labels, args.roots))
    act = fits.getdata(args.act).squeeze() if args.act else None
    sig = args.act_beam * 60 / args.pixsize * FWHM2SIGMA
    w = np.maximum(gaussian_filter(good.astype(float), sig), 1e-3)

    def load(root: str, k: int) -> np.ndarray:
        """Saved map, or with --act its difference with the ACT map after smoothing to the ACT beam."""
        m = np.load(f"{root}_cg{k}.npy")
        return (
            m if act is None else gaussian_filter(np.where(good, m, 0), sig) / w - act
        )

    amps = {}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=70 if args.debug else 120)
    for label, root in runs.items():
        amps[label] = np.array(
            [fit_quadratic(load(root, k), basis, use, good)[1] for k in args.iters]
        )
        print(label)
        for k, (rms, bowl) in zip(args.iters, amps[label]):
            print(f"  {k:4d}  surface rms {rms:.2e}  x^2+y^2 coefficient {bowl:+.2e}")
        for ax, y, name in zip(
            axs, amps[label].T, ("rms of fitted surface", "x^2+y^2 coefficient")
        ):
            ax.plot(args.iters, y, "o-", label=label)
            ax.set(xlabel="CG iteration", ylabel=f"{name} (Jy/beam)")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    show = [args.iters[0], args.iters[-1]]
    fig, axs = plt.subplots(
        len(runs),
        len(show),
        figsize=(4.5 * len(show), 4 * len(runs)),
        squeeze=False,
        dpi=70 if args.debug else 120,
    )
    sm = {
        (lab, k): np.where(
            good,
            gaussian_filter(
                np.where(good, np.load(f"{root}_cg{k}.npy"), 0), args.smooth
            )
            / np.maximum(gaussian_filter(good.astype(float), args.smooth), 1e-3),
            np.nan,
        )
        for lab, root in runs.items()
        for k in show
    }
    lim = np.nanpercentile(np.concatenate([v[good] for v in sm.values()]), [2, 98])
    for i, lab in enumerate(runs):
        for j, k in enumerate(show):
            im = axs[i, j].imshow(sm[lab, k], origin="lower", vmin=lim[0], vmax=lim[1])
            axs[i, j].set_title(f"{lab}, iteration {k}", fontsize=9)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    fig.colorbar(im, ax=axs, label="Jy/beam")
    fig.savefig(args.out.replace(".png", "_maps.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots", nargs="+", required=True, help="Prefixes of the _cg<N>.npy maps"
    )
    parser.add_argument("--labels", nargs="+", required=True, help="One label per root")
    parser.add_argument("--hits", required=True, help="Hits map FITS")
    parser.add_argument("--iters", type=int, nargs="+", default=[40, 80, 120, 160, 200])
    parser.add_argument("--min-hits", type=float, default=20)
    parser.add_argument(
        "--exclude",
        type=float,
        default=60,
        help="Radius around the centre left out of the fit, pixels",
    )
    parser.add_argument(
        "--smooth", type=float, default=20, help="Smoothing of the map panels, pixels"
    )
    parser.add_argument(
        "--out", required=True, help="Output figure (.png); maps go to _maps.png"
    )
    parser.add_argument(
        "--act",
        help="ACT map FITS on the same grid. Fit the difference of each map, smoothed to the ACT beam, "
        "and the ACT map instead, so that the cluster cancels and what is left is large scale disagreement with ACT",
    )
    parser.add_argument("--act-beam", type=float, default=2.1, help="ACT FWHM, arcmin")
    parser.add_argument("--pixsize", type=float, default=3.0, help="Pixel size, arcsec")
    parser.add_argument("--debug", action="store_true", help="Lower resolution figures")
    main(parser.parse_args())
