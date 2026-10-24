"""
Blink GIF (and optionally a side by side PNG) comparing the final direct needlet map with the maps of minkasi_mpi_example.py (plain PCG)
and minkasi_mask_example.py (hits-masked PCG). The PCG maps are reprojected onto the grid of the
needlet map and every frame shows only the pixels that are good in all three, on a shared colour scale.
"""

import argparse
import io

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from PIL import Image
from reproject import reproject_interp

from plot_final_aplpy import load, panel


def regrid(path: str, ref: tuple[np.ndarray, fits.Header], order: str) -> np.ndarray:
    """
    Reproject a FITS map onto the grid of a reference map.

    Parameters
    ----------
    path : str
        FITS map to reproject.
    ref : tuple[np.ndarray, fits.Header]
        Data and header of the reference map.
    order : str
        Interpolation order, e.g. "bilinear" for maps and "nearest-neighbor" for hit counts.

    Returns
    -------
    out : np.ndarray
        The map on the reference grid, NaN outside its footprint.
    """
    dat, hdr = load(path)
    return reproject_interp(
        (dat, WCS(hdr, naxis=2)), WCS(ref[1], naxis=2), ref[0].shape, order=order
    )[0]


def main(args: argparse.Namespace):
    """
    Make the GIF.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    need = load(args.need)
    hits = load(args.need_hits)[0]
    maps = {"need": need[0]}
    good = hits >= args.min_hits
    for key, thresh in (("mpi", 0), ("mask", args.min_hits - 1)):
        maps[key] = regrid(getattr(args, key), need, "bilinear")
        good &= regrid(getattr(args, f"{key}_hits"), need, "nearest-neighbor") > thresh
    if args.extra:  # same grid as the needlet map
        maps["extra"] = load(args.extra)[0]
    good &= np.all([np.isfinite(m) for m in maps.values()], axis=0)
    titles = {
        "extra": args.extra_label,
        "need": "needlet, CG on full operator",
        "mpi": "minkasi_mpi_example (no mask)",
        "mask": f"minkasi_mask_example (mask={args.min_hits:g})",
    }
    ref = [maps["need"]] if args.scale_need else maps.values()
    lim = np.percentile(np.concatenate([m[good] for m in ref]), [1, 99])
    dpi = 70 if args.debug else 110

    def draw(fig: plt.Figure, pos: tuple[int, int, int], key: str):
        m = maps[key]
        panel(
            fig,
            pos,
            np.where(good, m, np.nan),
            need[1],
            f"{titles[key]}\nrms {m[good].std():.2e} Jy/beam ({good.sum()} common px)",
            tuple(lim),
            beam_arcmin=args.m2_beam / 60,
        )

    if args.png:
        fig = plt.figure(figsize=(5.7 * len(maps), 5.5), dpi=dpi)
        for j, key in enumerate(maps):
            draw(fig, (1, len(maps), j + 1), key)
        fig.tight_layout()
        fig.savefig(args.png)
        plt.close(fig)
    frames = []
    for key in maps:
        fig = plt.figure(figsize=(6, 5), dpi=dpi)
        draw(fig, (1, 1, 1), key)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--need", required=True, help="Final needlet map FITS")
    parser.add_argument("--need-hits", required=True, help="Hits map FITS of --need")
    parser.add_argument("--mpi", required=True, help="Final map of minkasi_mpi_example")
    parser.add_argument("--mpi-hits", required=True, help="Hits map of --mpi")
    parser.add_argument(
        "--mask", required=True, help="Final map of minkasi_mask_example"
    )
    parser.add_argument("--mask-hits", required=True, help="Hits map of --mask")
    parser.add_argument("--out", required=True, help="Output GIF")
    parser.add_argument(
        "--extra-label", default="needlet, other", help="Title of --extra"
    )
    parser.add_argument(
        "--extra",
        help="Another final needlet map FITS to show, same grid as --need",
    )
    parser.add_argument("--png", help="Also save the three maps side by side here")
    parser.add_argument("--min-hits", type=float, default=20)
    parser.add_argument("--m2-beam", type=float, default=9.0, help="M2 FWHM, arcsec")
    parser.add_argument("--duration", type=int, default=1500, help="ms per frame")
    parser.add_argument(
        "--scale-need",
        action="store_true",
        help="Colour scale from the needlet map alone, not all three maps",
    )
    parser.add_argument("--debug", action="store_true", help="Lower resolution frames")
    main(parser.parse_args())
