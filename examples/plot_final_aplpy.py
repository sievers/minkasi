"""
Plot the final M2 map, the final map at ACT resolution on the ACT scale and the ACT map with aplpy
(sky coordinates, colour bars in Jy/beam). With --compare, plot a second final map and the
difference between them in percent of the rms of the first instead.
"""

import argparse

import aplpy
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

FWHM2SIGMA = 1 / np.sqrt(8 * np.log(2))


def load(path: str) -> tuple[np.ndarray, fits.Header]:
    """
    Read a 2D FITS map.

    Parameters
    ----------
    path : str
        FITS file.

    Returns
    -------
    data : np.ndarray
        Map as float64, shape (ny, nx).
    header : fits.Header
        Header with a 2D celestial WCS.
    """
    with fits.open(path) as hdul:
        return hdul[0].data.squeeze().astype(float), hdul[0].header


def panel(
    fig: plt.Figure,
    pos: tuple[int, int, int],
    data: np.ndarray,
    header: fits.Header,
    title: str,
    limits: tuple[float, float],
    label: str = "Jy/beam",
    cmap: str = "viridis",
    beam_arcmin: float | None = None,
) -> aplpy.FITSFigure:
    """
    Add one aplpy panel to a figure.

    Parameters
    ----------
    fig : plt.Figure
        Figure to draw into.
    pos : tuple[int, int, int]
        Subplot position (nrows, ncols, index).
    data : np.ndarray
        Map to show, NaN pixels are drawn white.
    header : fits.Header
        Header of the map.
    title : str
        Panel title.
    limits : tuple[float, float]
        Colour scale limits.
    label : str, default: "Jy/beam"
        Colour bar label.
    cmap : str, default: "viridis"
        Colour map.
    beam_arcmin : float | None, default: None
        FWHM of a beam to draw in the corner, in arcmin.

    Returns
    -------
    f : aplpy.FITSFigure
    """
    wcs_hdr = fits.Header(header)
    # the maps are pixelised with astropy's WCS, which follows Calabretta for -CAR with CRVAL2 != 0
    f = aplpy.FITSFigure(
        fits.PrimaryHDU(data, wcs_hdr),
        figure=fig,
        subplot=pos,
        north=False,
        convention="calabretta",
    )
    f.show_colorscale(vmin=limits[0], vmax=limits[1], cmap=cmap)
    f.set_nan_color("white")
    f.add_colorbar()
    f.colorbar.set_axis_label_text(label)
    f.set_title(title, fontsize=10)
    f.tick_labels.set_xformat("hh:mm:ss")
    f.tick_labels.set_yformat("dd:mm")
    if beam_arcmin is not None:
        f.add_beam(major=beam_arcmin / 60, minor=beam_arcmin / 60, angle=0)
        f.beam.set_facecolor("white")
        f.beam.set_alpha(0.4)
        f.beam.set_edgecolor("black")
    return f


def main(args: argparse.Namespace):
    """
    Make the figure.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    final, hdr = load(args.final)
    bad = load(args.hits)[0] < args.min_hits
    pct = lambda x: tuple(np.percentile(x[~bad], [1, 99]))
    shown = np.where(bad, np.nan, final)
    fig = plt.figure(figsize=(5.5 * 3, 5), dpi=100 if args.debug else 150)
    if args.compare:
        other = load(args.compare)[0]
        diff = 100 * (final - other) / final[~bad].std()
        lim = np.nanpercentile(np.abs(np.where(bad, np.nan, diff)), 99)
        lo, hi = pct(final)
        panel(
            fig,
            (1, 3, 1),
            shown,
            hdr,
            args.label,
            (lo, hi),
            beam_arcmin=args.m2_beam / 60,
        )
        panel(
            fig,
            (1, 3, 2),
            np.where(bad, np.nan, other),
            hdr,
            args.compare_label,
            (lo, hi),
        )
        panel(
            fig,
            (1, 3, 3),
            np.where(bad, np.nan, diff),
            hdr,
            f"({args.label} - {args.compare_label}) / rms({args.label}) [%]",
            (-lim, lim),
            label="%",
            cmap="RdBu_r",
        )
    else:
        act = load(args.act)[0]
        pix_arcmin = abs(hdr["CDELT2"]) * 60
        sigma = args.act_beam / pix_arcmin * FWHM2SIGMA
        smooth = gaussian_filter(np.where(bad, 0, final), sigma)
        smooth[bad] = act[bad]
        alim = tuple(np.percentile(act, [1, 99]))
        panel(
            fig,
            (1, 3, 1),
            shown,
            hdr,
            args.label,
            pct(final),
            beam_arcmin=args.m2_beam / 60,
        )
        panel(
            fig,
            (1, 3, 2),
            smooth,
            hdr,
            f"{args.label}, ACT beam (ACT scale)",
            alim,
            beam_arcmin=args.act_beam,
        )
        panel(fig, (1, 3, 3), act, hdr, "ACT", alim, beam_arcmin=args.act_beam)
    fig.tight_layout()
    fig.savefig(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", required=True, help="Final map FITS")
    parser.add_argument("--hits", required=True, help="Hits map FITS used for the mask")
    parser.add_argument("--act", help="ACT map FITS (not needed with --compare)")
    parser.add_argument("--compare", help="Second final map FITS to compare to --final")
    parser.add_argument("--label", default="final")
    parser.add_argument("--compare-label", default="other")
    parser.add_argument("--out", required=True, help="Output figure")
    parser.add_argument("--min-hits", type=float, default=20)
    parser.add_argument("--act-beam", type=float, default=2.1, help="ACT FWHM, arcmin")
    parser.add_argument("--m2-beam", type=float, default=9.0, help="M2 FWHM, arcsec")
    parser.add_argument("--debug", action="store_true", help="Lower resolution figure")
    args = parser.parse_args()
    if not args.compare and not args.act:
        parser.error("--act is required unless --compare is given")
    main(args)
