"""
Radial profiles around the cluster, to see the M2 bowl and how much of it the joint map keeps. All maps are seen at ACT
resolution (ACT beam, 30" pixels) inside the M2 region, with each map's tilt over the region removed and zero at its
outer edge. Top row: real data, against ACT. Bottom row: the simulation, against the truth (which shows that the
method recovers the cluster and does not make a bowl). Right column: the cluster profile itself at --fine-fwhm
resolution, real joint map and simulated joint map against the simulated truth.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from minkasi.needlet.act import act_conversion
from scipy.ndimage import gaussian_filter

from act_consistency import act_view
from multiscale_map import fine_grid, read_act_coadd, sample, sample_array

FWHM2SIGMA = 1 / np.sqrt(8 * np.log(2))


class Field:
    """
    The ACT grid around a run's M2 region, its sub-pixel grid, and helpers to see maps at ACT resolution.

    Parameters
    ----------
    root : str
        twores_ex outroot whose _hits.fits sets the region.
    args : argparse.Namespace
        Parsed arguments (act_glob, size, factor, min_hits, act_beam, edges).
    """

    def __init__(self, root: str, args: argparse.Namespace):
        hw = WCS(fits.getheader(f"{root}_hits.fits")).celestial
        ny, nx = fits.getdata(f"{root}_hits.fits").shape
        ra, dec = map(float, hw.wcs_pix2world((nx - 1) / 2, (ny - 1) / 2, 0))
        self.act, self.ivar = read_act_coadd(
            args.act_glob, ra, dec, args.size, act_conversion(90e9)
        )
        self.f = args.factor
        self.fw, self.shape, _ = fine_grid(self.act, self.f)
        self.pix = abs(self.fw.wcs.cdelt[0]) * 3600
        self.hits = sample(f"{root}_hits.fits", self.fw, self.shape, 0)
        self.good = self.hits >= 20
        self.sig = args.act_beam * 60 * FWHM2SIGMA / self.pix
        self.inm = (
            (self.hits >= args.min_hits)
            .reshape(self.act.shape[0], self.f, self.act.shape[1], self.f)
            .all((1, 3))
        )
        self.edges = np.arange(0, args.rmax + 1e-9, args.dr)
        self.rc = 0.5 * (self.edges[1:] + self.edges[:-1])

    def fine(self, path: str) -> np.ndarray:
        """
        A FITS map on the sub-pixel grid.

        Parameters
        ----------
        path : str
            FITS file.

        Returns
        -------
        np.ndarray
            The map.
        """
        return sample(path, self.fw, self.shape, 1)

    def view(self, m: np.ndarray) -> np.ndarray:
        """
        A sub-pixel map at ACT resolution.

        Parameters
        ----------
        m : np.ndarray
            Map on the sub-pixel grid.

        Returns
        -------
        np.ndarray
            Values per ACT pixel.
        """
        return act_view(m, self.good, self.act.shape, self.f, self.sig)

    def set_centre(self, m: np.ndarray, fwhm: float) -> None:
        """
        Put the centre at the minimum of a map smoothed to fwhm (arcsec), among pixels with at least 500 hits.

        Parameters
        ----------
        m : np.ndarray
            Map on the sub-pixel grid.
        fwhm : float
            Smoothing FWHM in arcsec.
        """
        sm = self.smooth(m, fwhm)
        self.cy, self.cx = np.unravel_index(
            np.argmin(np.where(self.hits >= 500, sm, np.inf)), self.shape
        )
        yy, xx = np.indices(self.act.shape)
        self.r_act = (
            np.hypot(
                (yy + 0.5) * self.f - 0.5 - self.cy, (xx + 0.5) * self.f - 0.5 - self.cx
            )
            * self.pix
            / 60
        )
        yy, xx = np.indices(self.shape)
        self.r_fine = np.hypot(yy - self.cy, xx - self.cx) * self.pix / 60
        self.yy, self.xx = np.indices(self.act.shape)

    def smooth(self, m: np.ndarray, fwhm: float) -> np.ndarray:
        """
        Gaussian smoothing inside the M2 footprint, normalised to it.

        Parameters
        ----------
        m : np.ndarray
            Map on the sub-pixel grid.
        fwhm : float
            FWHM in arcsec.

        Returns
        -------
        np.ndarray
            Smoothed map.
        """
        s = fwhm * FWHM2SIGMA / self.pix
        return gaussian_filter(m * self.good, s) / np.maximum(
            gaussian_filter(self.good.astype(float), s), 1e-3
        )

    def deplane(self, m: np.ndarray) -> np.ndarray:
        """
        Remove a map's plane (tilt) over the M2 region, keeping its mean.

        Parameters
        ----------
        m : np.ndarray
            Values per ACT pixel.

        Returns
        -------
        np.ndarray
            m without its tilt.
        """
        sel = self.inm
        a = np.column_stack([np.ones(sel.sum()), self.xx[sel], self.yy[sel]])
        c, *_ = np.linalg.lstsq(a, m[sel], rcond=None)
        return m - c[1] * self.xx - c[2] * self.yy

    def profile(self, m: np.ndarray, fine: bool = False) -> np.ndarray:
        """
        Mean in radial bins, in uK, inside the M2 region.

        Parameters
        ----------
        m : np.ndarray
            Values per ACT pixel, or on the sub-pixel grid with fine.
        fine : bool, default: False
            m is on the sub-pixel grid (bins over hits >= 20).

        Returns
        -------
        np.ndarray
            Profile per bin (nan where empty).
        """
        r, sel = (self.r_fine, self.good) if fine else (self.r_act, self.inm)
        out = []
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            b = sel & (r >= lo) & (r < hi)
            out.append(m[b].mean() * 1e6 if b.any() else np.nan)
        return np.array(out)


def zero_edge(p: np.ndarray, n: int = 3) -> np.ndarray:
    """
    Shift a profile so that the mean of its last n finite bins is zero.

    Parameters
    ----------
    p : np.ndarray
        Profile.
    n : int, default: 3
        Number of outer bins.

    Returns
    -------
    np.ndarray
        Shifted profile.
    """
    return p - np.nanmean(p[np.isfinite(p)][-n:])


def main(args: argparse.Namespace):
    """
    Make the figure and print the profiles.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.4))
    # real data: ACT, joint, M2 only
    real = Field(args.real_joint, args)
    jf = real.fine(f"{args.real_joint}_final.fits")
    real.set_centre(jf, 30)
    a = np.asarray(real.act)
    rows = {
        "ACT": real.profile(a),
        "joint": real.profile(real.deplane(real.view(jf))),
        "M2 only": real.profile(
            real.deplane(real.view(real.fine(f"{args.real_m2only}_final.fits")))
        ),
    }
    # simulation: truth, joint, M2 only
    sim = Field(args.sim_joint, args)
    sky, sw = (
        fits.getdata(args.sim_sky).astype(float),
        WCS(fits.getheader(args.sim_sky)).celestial,
    )
    truth = sample_array(sky, sw, sim.fw, sim.shape, 1)
    sim.set_centre(truth, 30)
    sj = sim.fine(f"{args.sim_joint}_final.fits")
    srows = {
        "truth": sim.profile(sim.view(truth)),
        "joint": sim.profile(sim.deplane(sim.view(sj))),
        "M2 only": sim.profile(
            sim.deplane(sim.view(sim.fine(f"{args.sim_m2only}_final.fits")))
        ),
    }
    for k, (field, prof, ref, lab) in enumerate(
        (
            (real, rows, "ACT", "real data, 106 TODs"),
            (sim, srows, "truth", "simulation, 40 TODs"),
        )
    ):
        for name, p in prof.items():
            ax[k, 0].plot(field.rc, zero_edge(p), "o-", ms=3, label=name)
            if name != ref:
                ax[k, 1].plot(
                    field.rc,
                    zero_edge(p - prof[ref]),
                    "o-",
                    ms=3,
                    label=f"{name} - {ref}",
                )
        ax[k, 0].set_title(f"{lab}: profiles at ACT resolution", fontsize=10)
        ax[k, 1].set_title(f"{lab}: difference from {ref}", fontsize=10)
        print(f"{lab}: radius (') " + " ".join(f"{x:6.2f}" for x in field.rc))
        for name, p in prof.items():
            print(f"  {name:8s} " + " ".join(f"{v:6.0f}" for v in zero_edge(p)))
            if name != ref:
                print(
                    f"  {name + ' - ' + ref:8s} "
                    + " ".join(f"{v:6.0f}" for v in zero_edge(p - prof[ref]))
                )
    # the cluster itself at fine resolution
    fw = args.fine_fwhm
    ax[0, 2].plot(
        real.rc,
        zero_edge(real.profile(real.smooth(jf, fw), fine=True)),
        "o-",
        ms=3,
        label="real joint",
    )
    ax[1, 2].plot(
        sim.rc,
        zero_edge(sim.profile(sim.smooth(truth, fw), fine=True)),
        "o-",
        ms=3,
        label="sim truth",
    )
    ax[1, 2].plot(
        sim.rc,
        zero_edge(sim.profile(sim.smooth(sj, fw), fine=True)),
        "o-",
        ms=3,
        label="sim joint",
    )
    ax[0, 2].set_title(f'real data: cluster profile at {fw:g}"', fontsize=10)
    ax[1, 2].set_title(f'simulation: cluster profile at {fw:g}"', fontsize=10)
    for x in ax.flat:
        x.axhline(0, color="k", lw=0.5)
        x.set_xlabel("radius from the cluster centre (arcmin)")
        x.set_ylabel(r"$\mu$K$_{\rm RJ}$, 0 at the outer edge")
        x.grid(alpha=0.3)
        x.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print("wrote", args.out)


if __name__ == "__main__":
    D = "/mnt/welch/USERS/jorlo/needlets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-joint", default=f"{D}/all106_nomask/run600_wcs/MOOJ1142"
    )
    parser.add_argument(
        "--real-m2only", default=f"{D}/all106_nomask/m2only300/MOOJ1142"
    )
    parser.add_argument("--sim-joint", default=f"{D}/sim/nomask/MOOJ1142")
    parser.add_argument("--sim-m2only", default=f"{D}/sim/m2only/MOOJ1142")
    parser.add_argument("--sim-sky", default=f"{D}/sim/sky_s0.fits")
    parser.add_argument(
        "--act-glob",
        default="/mnt/welch/USERS/jorlo/maps/dr6_20230316_splits/cmb_night_*_f090_3pass_4way_set*_ivar.fits",
    )
    parser.add_argument("--size", type=float, default=18.0, help="arcmin")
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument(
        "--min-hits",
        type=float,
        default=120,
        help="ACT pixels used: all sub-pixels this many hits",
    )
    parser.add_argument("--act-beam", type=float, default=2.1, help="arcmin")
    parser.add_argument("--rmax", type=float, default=6.0, help="arcmin")
    parser.add_argument("--dr", type=float, default=0.5, help="arcmin")
    parser.add_argument(
        "--fine-fwhm", type=float, default=20.0, help="arcsec, right column"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--debug", action="store_true", help="Coarse bins")
    args = parser.parse_args()
    if args.debug:
        args.dr = 1.0
    main(args)
