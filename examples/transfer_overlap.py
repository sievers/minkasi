"""
Overlap of the M2 and ACT transfer functions and sensitivities inside the M2 footprint, per Fourier mode.

Random white probes s restricted to the footprint P are passed through
  - the M2 data filter: bin(fit_cm_plus_poly(A s)) / hits, against bin(A s) / hits (the transfer of the per detector
    polynomial it subtracts from the data; the common mode is fitted jointly but not removed. The mapmaking
    operator does not model this filter),
  - the M2 operator P A^T N^-1 A P (the information the mapmaker assigns to each mode, noise model of the data),
  - the ACT forward operator (beam and ACT pixel) and the ACT term P F^T ivar F P of twores_ex.py.
The transfer is <conj(s_k) (O s)_k> / <|s_k|^2> for the filters and the same ratio gives the Fourier diagonal of
the information operators, azimuthally averaged. Both information terms are in the map basis of twores_ex
(the M2 beam convolved sky, K_RJ^-2 per pixel), so their ratio is the share of each experiment in each mode.
"""

import argparse
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import minkasi.minkasi_all as minkasi
import numpy as np
from minkasi.needlet.needlet import apply_beam, block_average, gaussian_beam_ft

from twores_ex import (
    ACT_OSAMP,
    act_conversion,
    read_act,
    read_tods,
    set_noise,
    setup_wmap,
)


def radial_bins(shape: tuple[int, int], pix_arcsec: float, nbin: int):
    """
    Multipole of every rfft2 mode and bins in it: one fundamental wide up to 16 fundamentals, logarithmic above.

    Parameters
    ----------
    shape : tuple[int, int]
        Map shape.
    pix_arcsec : float
        Pixel size in arcsec.
    nbin : int
        Number of bins.

    Returns
    -------
    ell : np.ndarray
        Multipole of each mode, rfft2 layout.
    edges : np.ndarray
        Bin edges in ell.
    """
    pix = np.deg2rad(pix_arcsec / 3600)
    k = np.hypot(np.fft.fftfreq(shape[0])[:, None], np.fft.rfftfreq(shape[1])[None, :])
    ell = 2 * np.pi * k / pix
    dl = 2 * np.pi / (shape[0] * pix)  # fundamental
    lin = dl * (np.arange(16) + 0.5)
    return ell, np.concatenate([lin, np.geomspace(lin[-1], ell.max(), nbin + 1)[1:]])


def binned(values: np.ndarray, ell: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Mean of a quantity in rfft2 layout in each multipole bin.

    Parameters
    ----------
    values : np.ndarray
        Quantity on the rfft2 grid.
    ell : np.ndarray
        Multipole of each mode.
    edges : np.ndarray
        Bin edges.

    Returns
    -------
    np.ndarray
        Bin means (nan for empty bins).
    """
    idx = np.digitize(ell.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < edges.size - 1)
    num = np.bincount(idx[ok], values.ravel()[ok], edges.size - 1)
    cnt = np.bincount(idx[ok], minlength=edges.size - 1)
    return np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)


def m2_operators(todvec, smap, s: np.ndarray, hits: np.ndarray):
    """
    Apply the M2 data filter and the M2 mapmaking operator to a probe.

    Parameters
    ----------
    todvec : TodVec
        TODs with their noise models.
    smap : SkyMap
        Map giving the pixelization (its contents are overwritten).
    s : np.ndarray
        Probe map.
    hits : np.ndarray
        Hit map.

    Returns
    -------
    filt : np.ndarray
        Binned map of the polynomial filtered TODs of s, over the binned map of the TODs of s.
    info : np.ndarray
        A^T N^-1 A s.
    """
    out = []
    for kind in ("raw", "filt", "info"):
        acc = smap.copy()
        acc.clear()
        for tod in todvec.tods:
            smap.map[:] = s
            dat = np.zeros(tod.info["dat_calib"].shape)
            smap.map2tod(tod, dat)
            if kind == "filt":
                dat = minkasi.fit_cm_plus_poly(np.ascontiguousarray(dat))
            elif kind == "info":
                dat = tod.apply_noise(dat)
            acc.tod2map(tod, dat)
        acc.mpi_reduce()
        out.append(acc.map.copy())
    raw, filt, info = out
    norm = np.divide(1.0, hits, out=np.zeros_like(hits), where=hits > 0)
    return filt * norm, info, raw * norm


def main(args: argparse.Namespace):
    """
    Measure the transfer functions and information per mode and plot them.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    todvec = read_tods(args.idir, args.ntods, args.name, args.ntods_total)
    wmap = setup_wmap(todvec, args.pixsize, args.nfilt, args.pad_frac)
    for tod in todvec.tods:
        tod.info["ipix"] = wmap.get_pix(tod)
        set_noise(tod, args.deproject_poly)
    hits = minkasi.make_hits(todvec, wmap.real_map).map
    fp = (hits >= args.min_hits).astype(float)
    shape = fp.shape

    paths = sorted(glob.glob(args.act_glob))[: 2 if args.debug else None]
    conv = act_conversion(args.freq, args.m2_beam, args.m2_unit)
    ivar = read_act(
        paths, wmap.lims, wmap, minkasi.make_hits(todvec, wmap.real_map), conv=conv
    )[0]
    ivar /= ACT_OSAMP**2  # weight per fine pixel, as in twores_ex
    act_beam = gaussian_beam_ft(shape, args.act_beam * 60 / args.pixsize)

    def act_forward(m):
        return block_average(apply_beam(m, act_beam), ACT_OSAMP)

    def act_info(m):
        return fp * apply_beam(
            block_average(ivar * act_forward(m), ACT_OSAMP), act_beam
        )

    ell, edges = radial_bins(shape, args.pixsize, args.nbin)
    acc = {
        k: np.zeros(edges.size - 1)
        for k in ("auto", "m2_filt", "m2_info", "act_tr", "act_info")
    }
    rng = np.random.default_rng(args.seed)  # same probes on every rank
    for j in range(args.nprobe):
        s = rng.normal(size=shape) * fp
        filt, info, raw = m2_operators(todvec, wmap.real_map.copy(), s, hits)
        sk = np.fft.rfft2(s)
        for key, o in (
            ("auto", s),
            ("m2_filt", filt),
            ("m2_info", info),
            ("act_tr", fp * act_forward(s)),
            ("act_info", act_info(s)),
        ):
            acc[key] += binned((np.conj(sk) * np.fft.rfft2(o)).real, ell, edges)
        if minkasi.myrank == 0:
            print(
                f"probe {j + 1}/{args.nprobe}, binned check {np.abs(raw - s)[fp > 0].max():.1e}"
            )
    res = {k: v / acc["auto"] for k, v in acc.items() if k != "auto"}
    res["m2_beam"] = binned(
        gaussian_beam_ft(shape, args.m2_beam / args.pixsize), ell, edges
    )
    lc = np.sqrt(edges[1:] * edges[:-1])
    # band profiles: squared filters (fftshifted layout) against ell
    kx = np.fft.fftshift(np.fft.fftfreq(shape[0]))[:, None]
    ky = np.fft.fftshift(np.fft.fftfreq(shape[1]))[None, :]
    ell_full = 2 * np.pi * np.hypot(kx, ky) / np.deg2rad(args.pixsize / 3600)
    filt2 = wmap.needlet.filters**2

    if minkasi.myrank != 0:
        return
    np.savez(args.out + ".npz", ell=lc, edges=edges, **res)
    share = res["m2_info"] / (res["m2_info"] + res["act_info"])
    print(
        "band  ell_peak  scale(arcmin)  M2_filter  ACT_transfer  M2_info   ACT_info  M2_share"
    )
    for i, f2 in enumerate(filt2):
        w = binned(np.fft.ifftshift(f2)[:, : shape[1] // 2 + 1], ell, edges)
        w = np.nan_to_num(w)
        lpk = (f2 * ell_full).sum() / f2.sum()
        avg = {k: np.nansum(w * v) / w[np.isfinite(v)].sum() for k, v in res.items()}
        print(
            f"{i:4d} {lpk:9.0f} {2 * np.pi / lpk * 180 / np.pi * 60 / 2:13.2f}"
            f" {avg['m2_filt']:10.3f} {avg['act_tr']:13.3f} {avg['m2_info']:9.2e}"
            f" {avg['act_info']:10.2e} {avg['m2_info'] / (avg['m2_info'] + avg['act_info']):9.3f}"
        )

    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    ax[0].plot(lc, res["m2_filt"], label="M2 polynomial filter (fit_cm_plus_poly)")
    ax[0].plot(lc, res["m2_filt"] * res["m2_beam"], label="M2 filter x beam")
    ax[0].plot(lc, res["act_tr"], label="ACT beam x pixel")
    ax[0].set_ylabel("transfer")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[1].loglog(lc, np.abs(res["m2_info"]), label=r"M2 $A^TN^{-1}A$")
    ax[1].loglog(lc, np.abs(res["act_info"]), label=r"ACT $F^T N^{-1} F$")
    ax[1].set_ylabel(r"information per mode (K$_{RJ}^{-2}$)")
    ax[2].semilogx(lc, share, "k", label="M2 share")
    ax[2].set_ylabel("M2 / (M2 + ACT)")
    ax[2].set_ylim(0, 1)
    for i, f2 in enumerate(filt2):
        w = binned(np.fft.ifftshift(f2)[:, : shape[1] // 2 + 1], ell, edges)
        ax[2].fill_between(lc, 0, np.nan_to_num(w), alpha=0.15)
        k = np.nanargmax(w)
        ax[2].text(lc[k], 0.02, str(i), ha="center", fontsize=8)
    ax[2].set_xlabel(r"$\ell$")
    top = ax[0].secondary_xaxis(
        "top",
        functions=(
            lambda l: 10800 / np.maximum(l, 1),
            lambda t: 10800 / np.maximum(t, 1e-9),
        ),
    )
    top.set_xlabel("half wavelength (arcmin)")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=120)
    print("wrote", args.out + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="MOOJ1142")
    parser.add_argument(
        "--idir", default="/mnt/welch/MUSTANG/M2-TODs/MOOJ1142/mustang2/"
    )
    parser.add_argument("--ntods", type=int, default=3, help="Per rank")
    parser.add_argument("--ntods-total", type=int, default=40)
    parser.add_argument("--nfilt", type=int, default=10)
    parser.add_argument("--pixsize", type=float, default=3.0, help="Arcsec")
    parser.add_argument("--pad-frac", type=float, default=0.25)
    parser.add_argument("--min-hits", type=float, default=20)
    parser.add_argument(
        "--act-glob",
        default="/mnt/welch/USERS/jorlo/maps/dr6_20230316_splits/cmb_night_*_f090_3pass_4way_set*_ivar.fits",
    )
    parser.add_argument("--act-beam", type=float, default=2.1, help="Arcmin")
    parser.add_argument("--m2-beam", type=float, default=9.0, help="Arcsec")
    parser.add_argument("--freq", type=float, default=90e9, help="Hz")
    parser.add_argument("--m2-unit", default="K_RJ", choices=["K_RJ", "Jy", "mJy"])
    parser.add_argument(
        "--deproject-poly", action="store_true", help="As in twores_ex.py"
    )
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--nbin", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output prefix (.png, .npz)")
    parser.add_argument(
        "--debug", action="store_true", help="2 TODs, 2 probes, 2 ACT maps"
    )
    args = parser.parse_args()
    if args.debug:
        args.ntods, args.ntods_total, args.nprobe = 2, 2, 2
    main(args)
