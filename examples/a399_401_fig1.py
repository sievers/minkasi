"""
Figure 1 (right) of Hincks et al. 2022 (arXiv:2107.04611) redone with the joint M2 + ACT two-resolution map of
A399-401: the 90 GHz multiscale display map (multiscale_map.py, same colour scale) of the joint solve, with their
ACT + Planck y-map as contours at 3, 5, ..., 15 sigma, the clusters' R500c circles (their Tables 1 and 4), the ACT
compact sources coloured by band and sized by the beam, and the beams. Next to it, cutouts of the M2 region in ACT
f090 alone and in the joint map, same pixels and colour scale, to show what M2 adds.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.wcs import WCS
from matplotlib.patches import Rectangle
from minkasi.needlet.act import act_conversion
from multiscale_map import (
    compose,
    fine_grid,
    make_norm,
    read_act_coadd,
    sample,
    sample_array,
    ticks,
)
from scipy.ndimage import gaussian_filter

COSMO = FlatLambdaCDM(H0=67.6, Om0=0.31)  # the paper's cosmology
# name: RA, Dec (Table 1, NED), z, R500c in Mpc (Table 4)
CLUSTERS = {
    "A399": ("02h57m56.4s", "+13d00m59s", 0.071806, 1.45),
    "A401": ("02h58m57.5s", "+13d34m46s", 0.073664, 1.53),
}
BANDS = {
    "f090": ("red", 2.1),
    "f150": ("blue", 1.4),
    "f220": ("green", 1.0),
}  # colour, ACT FWHM (arcmin)


def standard_orientation(wcs: WCS, *maps: np.ndarray):
    """
    Flip CAR maps so that RA increases to the left, as in the paper (pixell cutouts have RA increasing along x).

    Parameters
    ----------
    wcs : WCS
        WCS of the maps.
    *maps : np.ndarray
        Maps on that grid, [y, x].

    Returns
    -------
    wcs : WCS
        The matching WCS.
    *maps : np.ndarray
        The maps, flipped if needed.
    """
    if wcs.wcs.cdelt[0] < 0:
        return wcs, *maps
    wcs = wcs.deepcopy()
    wcs.wcs.crpix[0] = maps[0].shape[1] + 1 - wcs.wcs.crpix[0]
    wcs.wcs.cdelt[0] *= -1
    return wcs, *(m[:, ::-1] for m in maps)


def y_sigma(ymap: np.ndarray) -> float:
    """
    Noise of the y-map per pixel, robustly (1.4826 x median absolute deviation over the whole map, which is mostly
    noise; the clusters and sources are a small fraction of it).

    Parameters
    ----------
    ymap : np.ndarray
        The y-map.

    Returns
    -------
    float
        Standard deviation estimate.
    """
    v = ymap[ymap != 0]
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def read_sources(pattern: str, min_snr: float) -> dict[str, np.ndarray]:
    """
    ACT compact sources per band (all_<band>_cat.txt: ra, dec, SNR, ...).

    Parameters
    ----------
    pattern : str
        Catalogue path with {band} in place of the band name.
    min_snr : float
        Smallest S/N kept.

    Returns
    -------
    dict[str, np.ndarray]
        Band: (n, 2) array of RA, Dec in degrees.
    """
    out = {}
    for band in BANDS:
        for path in glob.glob(pattern.format(band=band)):
            cat = np.loadtxt(path, usecols=(0, 1, 2), ndmin=2)
            out[band] = cat[cat[:, 2] >= min_snr, :2]
    return out


def r500_arcmin(z: float, r500: float) -> float:
    """
    Angular radius of R500c.

    Parameters
    ----------
    z : float
        Redshift.
    r500 : float
        R500c in proper Mpc.

    Returns
    -------
    float
        Radius in arcmin.
    """
    return (r500 * u.Mpc / COSMO.angular_diameter_distance(z) * u.rad).to_value(
        u.arcmin
    )


def draw_clusters(ax):
    """
    R500c circles (dashed) and names of A399 and A401.

    Parameters
    ----------
    ax : WCSAxes
        The map axes.
    """
    tr = ax.get_transform("icrs")
    for name, (ra, dec, z, r500) in CLUSTERS.items():
        c = SkyCoord(ra, dec)
        r = r500_arcmin(z, r500) * u.arcmin
        ax.add_patch(
            SphericalCircle(
                (c.ra, c.dec), r, ec="0.4", fc="none", ls="--", lw=1, transform=tr
            )
        )
        ax.text(
            c.ra.deg,
            (c.dec + 0.8 * r).deg,
            name,
            color="0.4",
            ha="center",
            fontsize=10,
            transform=tr,
            clip_on=True,
        )


def draw_sources(ax, sources: dict[str, np.ndarray], box: tuple, lw: float = 1.2):
    """
    ACT compact sources inside a pixel box, coloured by band and sized by the beam FWHM.

    Parameters
    ----------
    ax : WCSAxes
        The map axes.
    sources : dict[str, np.ndarray]
        Output of read_sources.
    box : tuple
        Pixel limits (x0, x1, y0, y1) of the panel.
    lw : float, default: 1.2
        Line width.
    """
    x0, x1, y0, y1 = box
    for band, pos in sources.items():
        colour, fwhm = BANDS[band]
        px, py = ax.wcs.wcs_world2pix(pos[:, 0], pos[:, 1], 0)
        for ra, dec in pos[(px > x0) & (px < x1) & (py > y0) & (py < y1)]:
            ax.add_patch(
                SphericalCircle(
                    (ra * u.deg, dec * u.deg),
                    fwhm / 2 * u.arcmin,
                    ec=colour,
                    fc="none",
                    lw=lw,
                    transform=ax.get_transform("icrs"),
                )
            )


def draw_beams(ax, box: tuple, m2_fwhm: float, gap: float):
    """
    The ACT f090 beam and the displayed M2 resolution in the lower left corner of a panel.

    Parameters
    ----------
    ax : WCSAxes
        The map axes.
    box : tuple
        Pixel limits (x0, x1, y0, y1) of the panel.
    m2_fwhm : float
        Displayed M2 FWHM in arcsec.
    gap : float
        Spacing of the two beams in arcmin.
    """
    x0, x1, y0, y1 = box
    tr = ax.get_transform("icrs")
    lon, lat = ax.wcs.wcs_pix2world(
        [[x0 + 0.12 * (x1 - x0), y0 + 0.06 * (y1 - y0)]], 0
    )[0]
    for k, (label, fwhm) in enumerate(
        [("ACT", BANDS["f090"][1]), ("M2", m2_fwhm / 60)]
    ):
        x = lon - k * gap / 60 / np.cos(np.radians(lat))  # M2 to the east (left)
        ax.add_patch(
            SphericalCircle(
                (x * u.deg, lat * u.deg),
                fwhm / 2 * u.arcmin,
                ec="k",
                fc="none",
                lw=1,
                transform=tr,
            )
        )
        ax.text(x, lat + gap / 3.5 / 60, label, ha="center", fontsize=9, transform=tr)


def outline(ax, m2: np.ndarray, **kw):
    """
    Outline of the region showing the joint map.

    Parameters
    ----------
    ax : WCSAxes
        The map axes.
    m2 : np.ndarray
        True where the joint map is shown (compose).
    **kw
        Passed to contour.
    """
    ax.contour(
        gaussian_filter(m2.astype(float), 2.0),
        [0.5],
        colors="k",
        linewidths=0.6,
        alpha=0.6,
        **kw,
    )


def panel(fig, spec, wcs, data, norm, args, box, title=None):
    """
    A WCS map panel limited to a pixel box.

    Parameters
    ----------
    fig : Figure
        The figure.
    spec : SubplotSpec
        Where the panel goes.
    wcs : WCS
        WCS of the data.
    data : np.ndarray
        Map in K.
    norm : Normalize
        Colour normalisation in uK.
    args : argparse.Namespace
        Parsed arguments (cmap).
    box : tuple
        Pixel limits (x0, x1, y0, y1).
    title : str | None, default: None
        Panel title.

    Returns
    -------
    ax : WCSAxes
        The panel.
    im : AxesImage
        The image, for a colour bar.
    """
    ax = fig.add_subplot(spec, projection=wcs)
    im = ax.imshow(
        data * 1e6, origin="lower", cmap=args.cmap, norm=norm, interpolation="nearest"
    )
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_xlabel("RA (ICRS)")
    ax.set_ylabel("Dec (ICRS)")
    if title:
        ax.set_title(title, fontsize=10)
    return ax, im


def zoom_box(wcs: WCS, center, size: float, pix: float) -> tuple:
    """
    Pixel limits of a square cutout.

    Parameters
    ----------
    wcs : WCS
        WCS of the map.
    center : sequence of float
        RA, Dec of the cutout centre in degrees.
    size : float
        Side in arcmin.
    pix : float
        Pixel size in arcsec.

    Returns
    -------
    tuple
        (x0, x1, y0, y1).
    """
    cx, cy = wcs.wcs_world2pix([center], 0)[0]
    h = size * 60 / 2 / pix
    return cx - h, cx + h, cy - h, cy + h


def main(args: argparse.Namespace):
    """
    Make the figure: the paper's panel on the left, and cutouts of the M2 region in ACT alone and in the joint map
    on the right, with the same pixels and colour scale.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    ra, dec = args.center
    conv = act_conversion(args.freq, unit="K_RJ")
    act = read_act_coadd(args.act_glob, ra, dec, args.size, conv)[0]
    wcs, shape, act_up = fine_grid(act, args.factor)
    pix = abs(wcs.wcs.cdelt[0]) * 3600
    hits = sample(args.root + "_hits.fits", wcs, shape, 0)
    joint = sample(args.root + "_final.fits", wcs, shape, 1)
    image, _, m2 = compose(args, act_up, joint, hits, pix)
    wcs, image, m2, act_up = standard_orientation(wcs, image, m2, act_up)
    norm = make_norm(image, args)
    sources = read_sources(args.cat, args.min_snr)
    full = (-0.5, image.shape[1] - 0.5, -0.5, image.shape[0] - 0.5)
    zbox = zoom_box(wcs, args.zoom_center, args.zoom_size, pix)

    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.05, 1], wspace=0.25, hspace=0.25)
    ax, im = panel(fig, gs[:, 0], wcs, image, norm, args, full)
    with fits.open(args.ymap) as hdul:
        ymap, ywcs = hdul[0].data.astype(float), WCS(hdul[0].header).celestial
    if args.y_smooth > 0:  # contours of per-pixel noise are speckle
        pixy = abs(ywcs.wcs.cdelt[0]) * 60
        ymap = gaussian_filter(ymap, args.y_smooth / np.sqrt(8 * np.log(2)) / pixy)
    sig = args.y_sigma or y_sigma(ymap)  # of the smoothed map
    yfine = sample_array(ymap, ywcs, wcs, image.shape, 1)
    ax.contour(
        yfine, np.arange(3, 16, 2) * sig, colors="green", linewidths=0.8, alpha=0.9
    )
    outline(ax, m2, linestyles=":")
    draw_clusters(ax)
    draw_sources(ax, sources, full)
    draw_beams(ax, full, args.m2_fwhm, 7)
    x0, x1, y0, y1 = zbox
    ax.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, ec="k", fc="none", lw=1.2, ls="-")
    )
    ax.coords[0].set_major_formatter("hh:mm")
    ax.coords[1].set_major_formatter("dd:mm")

    for row, (data, title) in enumerate(
        [(act_up, "ACT f090 alone"), (image, "M2 + ACT joint map")]
    ):
        z, _ = panel(fig, gs[row, 1], wcs, data, norm, args, zbox, title)
        outline(z, m2)
        draw_sources(z, sources, zbox, lw=1.0)
        draw_beams(z, zbox, args.m2_fwhm, 4)
        z.coords[0].set_major_formatter("hh:mm:ss")
        z.coords[0].set_ticks(number=3)
        z.coords[1].set_major_formatter("dd:mm")
        if row == 0:
            z.coords[0].set_axislabel("")
            z.coords[0].set_ticklabel_visible(False)

    fig.colorbar(
        im,
        ax=fig.axes,
        label=r"$\mu$K$_{\rm RJ}$",
        shrink=0.8,
        pad=0.02,
        ticks=ticks(norm),
    )
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(
        f"y-map smoothed to {args.y_smooth:g}' FWHM, sigma {sig:.2e} per pixel, contours at 3..15 sigma; "
        f"zoom {args.zoom_size:g}' around {args.zoom_center}; wrote {args.out}"
    )


if __name__ == "__main__":
    bridge = "/scratch/jorlo/MUSTANG/M2-TODs/A399_401/bridge"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/scratch/jorlo/needlets/a399_401/r14/full/A399_401",
        help="twores_ex outroot (_final.fits and _hits.fits)",
    )
    parser.add_argument(
        "--act-glob", default=os.path.join(bridge, "all_f090_ivar.fits")
    )
    parser.add_argument(
        "--ymap",
        default=os.path.join(bridge, "comptonY/act_planck_sr_br_1.0_s19_y_map.fits"),
        help="ACT + Planck y-map of the paper, for the contours",
    )
    parser.add_argument(
        "--y-smooth",
        type=float,
        default=1.65,
        help="arcmin, Gaussian FWHM applied to --ymap before the contours (0: none)",
    )
    parser.add_argument(
        "--y-sigma",
        type=float,
        default=None,
        help="Noise per pixel of --ymap (default: robust estimate)",
    )
    parser.add_argument(
        "--cat",
        default=os.path.join(bridge, "all_{band}_cat.txt"),
        help="Source catalogues, {band} = f090, f150, f220",
    )
    parser.add_argument("--min-snr", type=float, default=5.0)
    # colour scale of multiscale_map.py (make_norm)
    parser.add_argument(
        "--vmin", type=float, default=None, help="uK_RJ (default the map minimum)"
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="uK_RJ (default the 99.7th percentile)",
    )
    parser.add_argument("--stretch", choices=["twoslope", "linear"], default="twoslope")
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--freq", type=float, default=90e9, help="Hz")
    parser.add_argument(
        "--center",
        type=float,
        nargs=2,
        default=[44.59, 13.37],
        metavar=("RA", "DEC"),
        help="Degrees; the paper's panel",
    )
    parser.add_argument(
        "--zoom-center",
        type=float,
        nargs=2,
        default=[44.662, 13.419],
        metavar=("RA", "DEC"),
        help="Degrees; centre of the M2 region (hits >= 120)",
    )
    parser.add_argument(
        "--zoom-size", type=float, default=34.0, help="arcmin, side of the cutouts"
    )
    parser.add_argument(
        "--size", type=float, default=88.0, help="arcmin; the paper's panel"
    )
    # display settings of multiscale_map.py (compose)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--min-hits", type=float, default=20)
    parser.add_argument("--show-hits", type=float, default=120)
    parser.add_argument(
        "--smooth", type=float, default=10.0, help="arcsec FWHM of the M2 part"
    )
    parser.add_argument("--smooth-ref-hits", type=float, default=3000)
    parser.add_argument("--smooth-max", type=float, default=30.0)
    parser.add_argument("--feather-hits", type=float, default=300)
    parser.add_argument("--match", choices=["offset", "none"], default="offset")
    parser.add_argument(
        "--m2-fwhm",
        type=float,
        default=13.5,
        help='arcsec, displayed M2 resolution (9" beam and 10" smoothing)',
    )
    parser.add_argument("--out", default="a399_401_fig1.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--debug", action="store_true", help="Smaller field, coarser sub-pixels"
    )
    args = parser.parse_args()
    if args.debug:
        args.size, args.factor, args.dpi, args.zoom_size = 40.0, 4, 70, 20.0
    main(args)
