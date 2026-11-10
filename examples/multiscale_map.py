"""
Two-resolution display map (for visualization only, not for science). The base grid is the ACT pixelization. Every
ACT pixel containing M2 data is split into factor x factor sub-pixels (factor ideally 2^n). Sub-pixels with enough
M2 hits show the M2 + ACT joint map, the others keep their ACT pixel's value. ACT pixels without M2 data show the
ACT map. The two regions share one colour scale; the joint map's offset can be matched to ACT where they overlap, the
M2 part lightly smoothed, and the joint map feathered into ACT by hits so that the noisy low-hit edge fades into ACT.
"""

import argparse
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import Normalize, TwoSlopeNorm
from pixell import enmap, utils
from scipy.ndimage import gaussian_filter, map_coordinates

from twores_ex import act_conversion


def read_act_coadd(
    pattern: str, ra: float, dec: float, size: float, conv: float
) -> tuple[enmap.ndmap, enmap.ndmap]:
    """
    Inverse-variance weighted coadd of the ACT split maps (temperature) in a box, converted to the M2 units.

    Parameters
    ----------
    pattern : str
        Glob of the ACT inverse variance maps; the maps are found by replacing _ivar with _map.
    ra, dec : float
        Box centre in degrees.
    size : float
        Box side in arcmin.
    conv : float
        Factor from ACT map units to the M2 units.

    Returns
    -------
    coadd, ivar : enmap.ndmap
        The coadd and its inverse variance (in the M2 units) on the ACT pixelization, [dec, ra] order.
    """
    half = size / 120 * utils.degree
    c = np.array([dec, ra]) * utils.degree
    box = np.array([c - half, c + half])
    num = den = 0
    for path in sorted(glob.glob(pattern)):
        ivar = enmap.read_map(path, box=box)
        m = enmap.read_map(path.replace("_ivar", "_map"), box=box)
        m = m[0] if m.ndim == 3 else m
        num, den = num + ivar * m, den + ivar
    coadd = np.divide(num, den, out=np.zeros(den.shape), where=den > 0) * conv
    return enmap.ndmap(coadd, den.wcs), enmap.ndmap(den / conv**2, den.wcs)


def subgrid_wcs(wcs: WCS, factor: int) -> WCS:
    """
    WCS of a grid with each pixel split into factor x factor sub-pixels.

    Parameters
    ----------
    wcs : WCS
        Coarse WCS.
    factor : int
        Sub-pixels per pixel along each axis.

    Returns
    -------
    WCS
        Fine WCS whose pixel blocks tile the coarse pixels exactly.
    """
    fine = wcs.deepcopy()
    fine.wcs.cdelt = wcs.wcs.cdelt / factor
    fine.wcs.crpix = (wcs.wcs.crpix - 0.5) * factor + 0.5
    return fine


def sample(path: str, wcs: WCS, shape: tuple[int, int], order: int) -> np.ndarray:
    """
    Sample a FITS image at the pixel centres of another grid.

    Parameters
    ----------
    path : str
        FITS file, image in the primary HDU.
    wcs : WCS
        Target grid.
    shape : tuple[int, int]
        Target shape, [y, x].
    order : int
        Interpolation order (0 nearest, 1 bilinear).

    Returns
    -------
    np.ndarray
        Samples, 0 outside the image.
    """
    with fits.open(path) as hdul:
        data, src = hdul[0].data.astype(float).squeeze(), WCS(hdul[0].header).celestial
    return sample_array(data, src, wcs, shape, order)


def sample_array(
    data: np.ndarray, src: WCS, wcs: WCS, shape: tuple[int, int], order: int
) -> np.ndarray:
    """
    Sample an image with a WCS at the pixel centres of another grid.

    Parameters
    ----------
    data : np.ndarray
        Image, [y, x].
    src : WCS
        Its WCS.
    wcs : WCS
        Target grid.
    shape : tuple[int, int]
        Target shape, [y, x].
    order : int
        Interpolation order (0 nearest, 1 bilinear).

    Returns
    -------
    np.ndarray
        Samples, 0 outside the image.
    """
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    px, py = src.wcs_world2pix(*wcs.wcs_pix2world(xx, yy, 0), 0)
    return map_coordinates(data, [py, px], order=order, cval=0.0)


def fine_grid(act: enmap.ndmap, factor: int) -> tuple[WCS, tuple[int, int], np.ndarray]:
    """
    The sub-pixel grid of an ACT map and the map repeated onto it.

    Parameters
    ----------
    act : enmap.ndmap
        Map on the ACT pixelization.
    factor : int
        Sub-pixels per ACT pixel along each axis.

    Returns
    -------
    wcs : WCS
        WCS of the fine grid.
    shape : tuple[int, int]
        Its shape.
    up : np.ndarray
        The ACT map as constant blocks on the fine grid.
    """
    wcs = subgrid_wcs(WCS(act.wcs.to_header()), factor)
    up = np.repeat(np.repeat(np.asarray(act), factor, 0), factor, 1)
    return wcs, up.shape, up


def smooth_by_hits(
    joint: np.ndarray,
    m2: np.ndarray,
    hits: np.ndarray,
    pix: float,
    fwhm: float,
    ref_hits: float | None,
    fwhm_max: float,
    nlevel: int = 6,
) -> np.ndarray:
    """
    Gaussian smoothing inside a region, normalised so that it does not pull in the values outside, with a width that
    can grow where the hits are low: FWHM = fwhm * sqrt(ref_hits / hits), between fwhm and fwhm_max. The M2 noise per
    pixel goes as 1 / sqrt(hits) and after smoothing as 1 / (FWHM sqrt(hits)), so this keeps the displayed noise about
    constant below ref_hits.

    Parameters
    ----------
    joint : np.ndarray
        Map to smooth.
    m2 : np.ndarray
        Region to smooth within.
    hits : np.ndarray
        Hits per pixel.
    pix : float
        Pixel size in arcsec.
    fwhm : float
        FWHM in arcsec where hits >= ref_hits (everywhere if ref_hits is None).
    ref_hits : float | None
        Hits above which the FWHM is `fwhm`.
    fwhm_max : float
        Largest FWHM in arcsec.
    nlevel : int, default: 6
        Number of smoothing widths computed; each pixel interpolates between the two that bracket its FWHM.

    Returns
    -------
    np.ndarray
        The smoothed map.
    """

    def smooth(f):
        sig = f / np.sqrt(8 * np.log(2)) / pix
        return gaussian_filter(joint * m2, sig) / np.maximum(
            gaussian_filter(m2.astype(float), sig), 1e-3
        )

    if not ref_hits or fwhm_max <= fwhm:
        return smooth(fwhm)
    target = np.clip(fwhm * np.sqrt(ref_hits / np.maximum(hits, 1e-3)), fwhm, fwhm_max)
    levels = np.geomspace(fwhm, fwhm_max, nlevel)
    maps = np.array([smooth(f) for f in levels])
    t = np.interp(np.log(target), np.log(levels), np.arange(nlevel))
    lo = np.floor(t).astype(int).clip(0, nlevel - 2)
    frac = t - lo
    rows = np.indices(t.shape)
    return (1 - frac) * maps[lo, rows[0], rows[1]] + frac * maps[
        lo + 1, rows[0], rows[1]
    ]


def compose(
    args, act_up: np.ndarray, joint: np.ndarray, hits: np.ndarray, pix: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the two-resolution map on the sub-pixel grid.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (factor, min_hits, show_hits, smooth, smooth_ref_hits, smooth_max, match, feather_hits).
    act_up : np.ndarray
        ACT map as constant blocks on the fine grid (fine_grid).
    joint : np.ndarray
        Joint map sampled on the fine grid.
    hits : np.ndarray
        M2 hits sampled on the fine grid.
    pix : float
        Sub-pixel size in arcsec.

    Returns
    -------
    image : np.ndarray
        Display map; split ACT pixels hold sub-pixel values, the others are constant blocks.
    split : np.ndarray
        True on the sub-pixels of split ACT pixels.
    m2 : np.ndarray
        True on the sub-pixels showing the joint map.
    """
    f = args.factor
    show = max(args.min_hits, args.show_hits or 0)
    m2 = hits >= show  # sub-pixels showing the joint map; the others keep the ACT value
    if args.smooth > 0:
        joint = smooth_by_hits(
            joint, m2, hits, pix, args.smooth, args.smooth_ref_hits, args.smooth_max
        )
    has = hits >= args.min_hits  # ACT pixels with any of these are split
    blocks = has.reshape(has.shape[0] // f, f, has.shape[1] // f, f).any((1, 3))
    split = np.repeat(np.repeat(blocks, f, 0), f, 1)
    if args.match == "offset":
        off = np.median((joint - act_up)[m2])
        print(
            f"offset of the joint map from ACT in the M2 region: {off * 1e6:.1f} uK_RJ, subtracted"
        )
        joint = joint - off
    w = (
        np.clip((hits - show) / args.feather_hits, 0, 1)
        if args.feather_hits > 0
        else 1.0
    )
    image = np.where(m2, w * joint + (1 - w) * act_up, act_up)
    return image, split, m2


def make_norm(image: np.ndarray, args) -> Normalize:
    """
    Colour normalisation of a display map, in uK.

    Parameters
    ----------
    image : np.ndarray
        Display map in K.
    args : argparse.Namespace
        Parsed arguments (vmin, vmax, stretch).

    Returns
    -------
    Normalize
        twoslope: 0 at the centre of the colour map, the negative (decrement) and positive (CMB) sides scaled
        separately so that both use the full colour range; linear: symmetric about 0.
    """
    lo = image.min() * 1e6 if args.vmin is None else args.vmin
    hi = np.percentile(image, 99.7) * 1e6 if args.vmax is None else args.vmax
    if args.stretch == "linear":
        lo, hi = -max(-lo, hi), max(-lo, hi)
    return (
        TwoSlopeNorm(0.0, lo, hi) if args.stretch == "twoslope" else Normalize(lo, hi)
    )


def smoothing_label(args) -> str:
    """
    Short description of the display settings for a figure title.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (smooth, smooth_ref_hits, smooth_max, show_hits, min_hits, feather_hits).

    Returns
    -------
    str
        For example 'M2 part smoothed to 10" (above 3000 hits, up to 30" below), shown above 120 hits'.
    """
    if args.smooth <= 0:
        sm = "M2 part unsmoothed"
    elif args.smooth_ref_hits:
        sm = (
            f'M2 part smoothed to {args.smooth:g}" above {args.smooth_ref_hits:g} hits, '
            f'up to {args.smooth_max:g}" below'
        )
    else:
        sm = f'M2 part smoothed to {args.smooth:g}"'
    show = max(args.min_hits, args.show_hits or 0)
    fe = f", feathered over {args.feather_hits:g}" if args.feather_hits > 0 else ""
    return f"{sm}; shown above {show:g} hits{fe}"


def ticks(norm: Normalize) -> np.ndarray:
    """
    Colour bar ticks with both sides of a two-slope scale labelled.

    Parameters
    ----------
    norm : Normalize
        Colour normalisation.

    Returns
    -------
    np.ndarray
        Ticks: the ends, halves and 0, rounded toward 0 to 2 significant figures so they stay in range.
    """
    t = np.array([norm.vmin, norm.vmin / 2, 0.0, norm.vmax / 2, norm.vmax])
    step = 10.0 ** (np.floor(np.log10(np.maximum(np.abs(t), 1e-30))) - 1)
    return np.trunc(t / step) * step


def draw(
    ax, image: np.ndarray, split: np.ndarray, norm: Normalize, cmap: str, outline: bool
):
    """
    Draw a display map on WCS axes.

    Parameters
    ----------
    ax : WCSAxes
        Axes with the fine-grid projection.
    image, split : np.ndarray
        Output of compose.
    norm : Normalize
        Colour normalisation in uK.
    cmap : str
        Colour map.
    outline : bool
        Outline the split ACT pixels.

    Returns
    -------
    AxesImage
        The image, for a colour bar.
    """
    im = ax.imshow(
        image * 1e6, origin="lower", cmap=cmap, norm=norm, interpolation="nearest"
    )
    if outline:
        ax.contour(split.astype(float), [0.5], colors="k", linewidths=0.6, alpha=0.6)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.coords.grid(color="0.5", alpha=0.25, linestyle=":")
    return im


def plot(args, image, split, m2, wcs):
    """
    Plot the map with sky coordinates.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    image, split, m2 : np.ndarray
        Output of compose.
    wcs : WCS
        WCS of the fine grid.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection=wcs)
    im = draw(ax, image, split, make_norm(image, args), args.cmap, args.outline)
    fig.colorbar(
        im, ax=ax, label=r"$\mu$K$_{\rm RJ}$", shrink=0.85, ticks=ticks(im.norm)
    )
    pix = abs(wcs.wcs.cdelt[0]) * 3600
    ax.set_title(
        f'{args.title}\nACT {pix * args.factor:g}" pixels, split into {args.factor}x{args.factor} '
        f'({pix:g}") where M2 has data\n{smoothing_label(args)}',
        fontsize=10,
    )
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print("wrote", args.out)


def main(args: argparse.Namespace):
    """
    Make the map.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    with fits.open(args.joint) as hdul:
        hw = WCS(hdul[0].header).celestial
        ny, nx = hdul[0].data.squeeze().shape
    ra, dec = hw.wcs_pix2world((nx - 1) / 2, (ny - 1) / 2, 0)
    conv = act_conversion(args.freq, unit="K_RJ")
    act = read_act_coadd(args.act_glob, float(ra), float(dec), args.size, conv)[0]
    wcs, shape, act_up = fine_grid(act, args.factor)
    pix = abs(wcs.wcs.cdelt[0]) * 3600
    hits, joint = sample(args.hits, wcs, shape, 0), sample(args.joint, wcs, shape, 1)
    image, split, m2 = compose(args, act_up, joint, hits, pix)
    print(
        f"ACT map {act.shape}, fine grid {image.shape}; {split.mean():.1%} of the area split, "
        f"{m2.mean():.1%} showing the joint map"
    )
    if args.fits:
        fits.writeto(args.fits, image, wcs.to_header(), overwrite=True)
    plot(args, image, split, m2, wcs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint",
        default="/mnt/welch/USERS/jorlo/needlets/lat40/krj600/MOOJ1142_final.fits",
        help="Joint M2 + ACT map (twores_ex _final.fits, K_RJ)",
    )
    parser.add_argument(
        "--hits",
        default="/mnt/welch/USERS/jorlo/needlets/lat40/krj600/MOOJ1142_hits.fits",
    )
    parser.add_argument(
        "--act-glob",
        default="/mnt/welch/USERS/jorlo/maps/dr6_20230316_splits/cmb_night_*_f090_3pass_4way_set*_ivar.fits",
    )
    parser.add_argument(
        "--freq", type=float, default=90e9, help="Hz, for the ACT unit conversion"
    )
    parser.add_argument(
        "--size", type=float, default=30.0, help="Side of the map, arcmin"
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=8,
        help="Sub-pixels per ACT pixel along each axis (2^n)",
    )
    parser.add_argument(
        "--min-hits",
        type=float,
        default=20,
        help="Hits for an ACT pixel to be split (and the default of --show-hits)",
    )
    parser.add_argument(
        "--show-hits",
        type=float,
        default=None,
        help="Hits for a sub-pixel to show the joint map (default --min-hits); sub-pixels of split ACT pixels "
        "with fewer show the ACT value. Display only: the solve can still use every sample",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Gaussian FWHM for the M2 part, arcsec",
    )
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
    parser.add_argument(
        "--feather-hits",
        type=float,
        default=0.0,
        help="Blend joint into ACT over this many hits above --show-hits (0: no blending)",
    )
    parser.add_argument("--match", choices=["offset", "none"], default="offset")
    parser.add_argument(
        "--vmin", type=float, default=None, help="uK_RJ (default the map minimum)"
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="uK_RJ (default the 99.7th percentile)",
    )
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument(
        "--stretch",
        choices=["twoslope", "linear"],
        default="twoslope",
        help="twoslope: separate linear scales below and above 0; linear: symmetric about 0",
    )
    parser.add_argument(
        "--outline", action="store_true", help="Outline the split ACT pixels"
    )
    parser.add_argument("--title", default="MOOJ1142, M2 + ACT")
    parser.add_argument(
        "--fits", default=None, help="Also write the fine-grid display map to this FITS"
    )
    parser.add_argument("--out", required=True, help="Output PNG")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--debug", action="store_true", help="Two ACT splits, 12 arcmin, low resolution"
    )
    args = parser.parse_args()
    if args.debug:
        args.act_glob = args.act_glob.replace("set*", "set0")
        args.size, args.dpi = 12.0, 70
    main(args)
