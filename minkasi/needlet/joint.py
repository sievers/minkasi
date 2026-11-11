"""
Joint MUSTANG-2 (TODs) + ACT (maps) needlet mapmaking.

The map is a stack of needlet bands. The M2 term is the TOD operator A^T N^-1 A in the needlet basis, the ACT term
F_i B^T s W s B (sum_j F_j x_j) on the bands ACT's beam transmits, and an optional prior acts on chosen bands in a chosen
region. The combined system is solved by CG on the full (band coupled) operator, with the block solve of the probed
diagonal blocks as preconditioner, or by the block solve alone. JointMapper(parse_args()).run() does everything;
examples/twores_ex.py is the command line front end.
"""

import argparse
import glob
import os
import time
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ..mapmaking.noise import NoiseSmoothedSVD
from ..mapmaking.pcg import pcg, run_pcg_wprior
from ..mapmaking.tod2map import make_hits
from ..maps.mapset import Mapset
from ..maps.utils import sample_image
from ..parallel import barrier, myrank
from ..tods.io import read_tods
from .act import ACT_OSAMP, ActTerm, act_conversion, edge_taper, read_act, sim_act
from .cmb import cmb_fourier_weight, lensed_cmb_cl
from .diagnostics import (
    coefficient_support,
    footprint_of,
    pixel_solve,
    residual_diagnostics,
    ritz_analysis,
    symmetry_check,
)
from .needlet import cached_responses, map2wav_real, wav2map_real, wavskymap_for_tods
from .sim import inject_sim


class JointMapper:
    """
    The joint M2 + ACT needlet mapmaking pipeline.

    Parameters
    ----------
    cfg : argparse.Namespace
        Options as returned by parse_args (see its help for each).

    Attributes
    ----------
    todvec : TodVec
        The TODs of this rank.
    wmap : WavSkyMap
        The needlet map.
    hits : SkyMap
        M2 hits.
    mask : NDArray | float
        Hits mask of the M2 term (1.0 without a mask).
    act : ActTerm | None
        The ACT term, None without ACT maps.
    """

    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.sky = None
        self.act = None
        self.act_bands: list[int] = []
        self.act_rhs: NDArray | float = 0
        self.prior_rhs: NDArray | float = 0
        self.prior = None
        self.keep = None
        self.t0 = time.time()

    @property
    def filt(self) -> NDArray:
        """The needlet filters."""
        return self.wmap.needlet.filters

    def read(self) -> None:
        """
        Read the TODs, build the needlet map and, with --sim-sky, replace the data by a simulation of that sky.
        """
        cfg = self.cfg
        self.todvec = read_tods(
            cfg.idir,
            cfg.ntods,
            cfg.name,
            cfg.ntods_total,
            raw=bool(cfg.sim_sky),
            tod_list=cfg.tod_list,
        )
        self.wmap = wavskymap_for_tods(
            self.todvec, cfg.pixsize, cfg.nfilt, cfg.pad_frac
        )
        if cfg.sim_sky:  # replace the data by a simulation of a known sky
            self.sky = sample_image(cfg.sim_sky, self.wmap.real_map)
            inject_sim(self.todvec, self.wmap, self.sky, cfg.sim_noise, cfg.sim_seed)
            if myrank == 0:
                truth = self.wmap.real_map.copy()
                truth.map = self.sky
                truth.write(cfg.outroot + "_truth.fits")

    def prepare(self) -> None:
        """
        Pointing, noise models, hits (written to _hits.fits) and the hits mask.
        """
        cfg = self.cfg
        for tod in self.todvec.tods:
            tod.info["ipix"] = self.wmap.get_pix(tod)
            tod.set_noise(NoiseSmoothedSVD)
        self.hits = make_hits(self.todvec, self.wmap.real_map)
        if myrank == 0:
            self.hits.write(cfg.outroot + "_hits.fits")
        barrier()
        if cfg.min_hits > 0:
            self.wmap.set_mask(
                self.hits.map, cfg.min_hits - 1
            )  # mask pixels with hits < min_hits
        self.mask = 1.0 if self.wmap.mask is None else self.wmap.mask

    def initial_pcg(self) -> None:
        """
        A few PCG iterations of the M2 problem (--maxiter, preconditioned by 1 / hits). With --direct the model only
        cleans the noise estimate; otherwise it is subtracted from the TODs and added back at the end.
        """
        cfg = self.cfg
        self.mapset = Mapset()
        self.mapset.add_map(self.wmap)
        rhs = self.mapset.copy()
        self.todvec.make_rhs(rhs)
        x0 = rhs.copy()
        x0.clear()
        precon = self.mapset.copy()
        hm = self.hits.map
        precon.maps[0].map[:] = np.divide(1.0, hm, out=np.zeros_like(hm), where=hm > 0)
        print("Saving to ", cfg.outroot)
        out = run_pcg_wprior(
            rhs,
            x0,
            self.todvec,
            precon=precon,
            maxiter=cfg.maxiter,
            save_iters=[cfg.maxiter],
            outroot=cfg.outroot,
        )
        self.pcg_map = wav2map_real(out.maps[0].map, self.filt)[0] * self.mask
        if myrank == 0:
            out.maps[0].write(cfg.outroot + "_need.fits")
        barrier()
        for tod in self.todvec.tods:
            tmp = np.zeros(tod.info["dat_calib"].shape)
            out.maps[0].map2tod(tod, tmp)
            if (
                cfg.direct
            ):  # the PCG model only cleans the noise estimate, the data keep the signal
                tod.set_noise(NoiseSmoothedSVD, dat=tod.info["dat_calib"] - tmp)
            else:
                tod.info["dat_calib"] -= tmp
                tod.set_noise(NoiseSmoothedSVD)
        self.rhs = self.mapset.copy()
        self.todvec.make_rhs(
            self.rhs
        )  # wavelet basis rhs, of the residual unless --direct

    def m2_response(self) -> None:
        """
        The lattices of the bands and the M2 response matrices (cached).
        """
        cfg = self.cfg
        wmap, need = self.wmap, self.wmap.needlet
        wmap.get_downsamps()
        self.cache = os.path.join(cfg.cache, cfg.name)
        if (
            cfg.lattice_tol is not None
        ):  # the matrices depend on the lattice, keep them apart
            self.cache += f"_nyq{cfg.lattice_tol:g}"
        # window of the response matrices: a few filter extents, never more than --radius
        radii = [
            min(
                cfg.radius or np.inf,
                2 * need.get_need_lims(i, real_space=True)[1] / cfg.pixsize,
            )
            for i in range(need.nfilt)
        ]
        self.radius = None if cfg.radius is None else radii
        self.bands = wmap.select_filters(None if cfg.direct else cfg.max_res)
        if cfg.lattice_tol is not None:
            wmap.get_downsamps(cfg.lattice_tol)
            print(
                "lattice from the filter support: downsamps",
                wmap.downsamps,
                "points",
                wmap.nxs_red,
            )
        t0 = time.time()
        self.m2_resp = cached_responses(
            self.cache + "_m2",
            lambda b: wmap.get_response_matrix(
                self.todvec,
                nfilts=b,
                downsamps=None if cfg.lattice_tol is None else wmap.downsamps,
                radius=self.radius,
                exact_below=cfg.exact_below,
            ),
            cfg.redo,
            need.nfilt,
            self.bands,
        )
        print(f"M2 response took {time.time() - t0:.1f} s")
        self.responses = [self.m2_resp]

    def act_term(self) -> None:
        """
        Read ACT (or simulate it with --sim-sky), write the ACT-only map to _act.fits, and build the ACT term, its
        right hand side and its response matrices (cached).
        """
        cfg = self.cfg
        t0 = time.time()
        paths = sorted(glob.glob(cfg.act_glob))[: 2 if cfg.debug else None]
        if not paths:
            return
        # TODO: ivar is treated as a diagonal map space inverse covariance. Replace this with a
        # robust map space noise covariance estimate once we have one (see tools.noise_estimation).
        conv = cfg.act_conv or act_conversion(cfg.freq, cfg.m2_beam, cfg.m2_unit)
        ivar, wdata = read_act(paths, self.wmap.lims, self.wmap, conv=conv)
        fwhm_pix = cfg.act_beam * 60 / cfg.pixsize
        if self.sky is not None:
            wdata = ivar * sim_act(self.sky, ivar, fwhm_pix, cfg.sim_seed)
        if myrank == 0:  # ACT only (inverse variance weighted coadd) for comparison
            self.wmap.real_map.map = np.divide(
                wdata, ivar, out=np.zeros_like(wdata), where=ivar > 0
            )
            self.wmap.real_map.write(cfg.outroot + "_act.fits")
        ivar_pix = np.median(
            ivar[ivar > 0]
        )  # of one ACT pixel, before the weights below
        # The filters are periodic but the ACT map is not: taper the weights at the edges. ivar is that of an ACT
        # pixel repeated on its ACT_OSAMP**2 fine pixels, which the sum over fine pixels would count that many
        # times: weight every fine pixel by 1/ACT_OSAMP**2
        norm = edge_taper(ivar.shape, cfg.taper) * cfg.act_ivar_scale / ACT_OSAMP**2
        ivar, wdata = ivar * norm, wdata * norm
        # ACT only constrains the bands its beam transmits; deconvolving the others amplifies noise
        transfer = self.wmap.band_transfer(fwhm_pix)
        self.act_bands = [i for i in self.bands if transfer[i] >= cfg.act_min_transfer]
        print("ACT used in bands", self.act_bands, "beam transfer", transfer.round(3))
        # The CMB is noise for M2 but not white: weight the Fourier modes by N_l / (N_l + C_l B_l^2) (--act-cmb),
        # on top of the white pixel noise: precision = diag(s) W diag(s), s = sqrt(ivar)
        weight = None
        if cfg.act_cmb > 0:
            pix = cfg.pixsize / 3600 * np.pi / 180
            cl = lensed_cmb_cl(cache=os.path.join(cfg.cache, "cmb_cl.npy")) * conv**2
            weight = cmb_fourier_weight(
                ivar.shape,
                pix,
                cl,
                cfg.act_beam / 60 * np.pi / 180,
                ivar_pix,
                (ACT_OSAMP * pix) ** 2,
                cfg.act_cmb,
            )
        self.act = ActTerm(ivar, wdata, self.wmap, fwhm_pix, self.act_bands, weight)
        # ACT sees the sky through its beam; without --direct the PCG model is removed from the M2 TODs, so ACT
        # enters through its residual with respect to the beam smoothed PCG map
        resid = (
            self.act.act_map
            if cfg.direct
            else self.act.act_map - self.act.forward(self.pcg_map)
        )
        self.act_rhs = self.act.rhs(resid)
        tag = (
            f"_act_t{cfg.taper}_c{conv:.3g}_b{cfg.act_beam}_pw_tr{cfg.act_min_transfer}"
            + f"_n{ACT_OSAMP}_wcs"  # _wcs: ACT placed by sky position (fix of 2026-11-10)
            + (f"_s{cfg.act_ivar_scale:g}" if cfg.act_ivar_scale != 1 else "")
            + (f"_cmb{cfg.act_cmb:g}" if cfg.act_cmb > 0 else "")
        )
        self.responses.append(
            cached_responses(
                self.cache + tag,
                self.act.responses(self.radius, cfg.exact_below),
                cfg.redo,
                self.wmap.needlet.nfilt,
                self.act_bands,
            )
        )
        print(f"ACT took {time.time() - t0:.1f} s")

    def prior_term(self) -> None:
        """
        With --outside-prior, the prior eps |P (m_B - m0_B)|^2: region P (hits < --prior-min-hits, or --prior-mask),
        bands B (--prior-bands, or the non-ACT bands with --prior-non-act, else all), mean m0 (--prior-mean, else 0).
        eps is --outside-prior times the M2 weight per pixel. Adds its responses and right hand side.
        """
        cfg = self.cfg
        if cfg.outside_prior is None:
            return
        if cfg.act_only:
            raise SystemExit("--outside-prior needs the M2 term")
        nfilt = self.wmap.needlet.nfilt
        # the region: nonzero pixels of --prior-mask, or hits < --prior-min-hits (independent of the M2 mask)
        if cfg.prior_mask:
            foot = sample_image(cfg.prior_mask, self.wmap.real_map) <= 0.5
            tag = "_m" + os.path.splitext(os.path.basename(cfg.prior_mask))[0]
        else:
            foot = self.hits.map >= cfg.prior_min_hits
            tag = f"_h{cfg.prior_min_hits:g}" if cfg.prior_min_hits != 20 else ""
        outside = 1.0 - foot
        prior_resp = cached_responses(
            f"{self.cache}_outside" + tag,
            lambda bands: self.wmap.get_response_matrix_map(
                outside.ravel(),
                nfilts=bands,
                radius=self.radius,
                exact_below=cfg.exact_below,
            ),
            cfg.redo,
            nfilt,
            self.bands,
        )
        # weight relative to the M2 weight per pixel, from the finest band (one lattice point per pixel)
        last = self.bands[-1]
        pts = np.ix_(self.wmap.nx_space[last], self.wmap.ny_space[last])
        infoot = foot[pts].ravel()
        eps = cfg.outside_prior * (
            np.median(self.m2_resp[last].diagonal()[infoot])
            / np.median(prior_resp[last].diagonal()[~infoot])
        )
        print(
            f"outside prior: eps = {eps:.3e} ({cfg.outside_prior:g} of the M2 pixel weight)"
        )
        if cfg.prior_bands is not None:
            poff = [i for i in range(nfilt) if i not in cfg.prior_bands]
        else:
            poff = list(self.act_bands) if cfg.prior_non_act else []
        print("outside prior in bands", [i for i in range(nfilt) if i not in poff])
        self.responses.append(
            np.array(
                [
                    None if g is None or i in poff else eps * g
                    for i, g in enumerate(prior_resp)
                ],
                dtype=object,
            )
        )
        self.prior = {"eps": eps, "outside": outside, "poff": poff}
        if cfg.prior_mean:  # pull toward the prior bands of a map, not toward zero
            wm = map2wav_real(
                sample_image(cfg.prior_mean, self.wmap.real_map) * cfg.prior_mean_scale,
                self.filt,
            )[0]
            wm[poff] = 0
            mean = wav2map_real(wm, self.filt)[0]
            self.prior_rhs = self.prior_dot_map(mean)
            print(f"prior mean from {cfg.prior_mean}, prior-band rms {mean.std():.2e}")

    def prior_dot_map(self, m: NDArray) -> NDArray:
        """
        F_i eps P m on the prior bands.

        Parameters
        ----------
        m : NDArray
            Real space map.

        Returns
        -------
        NDArray
            Wavelet array, zero outside the prior bands.
        """
        yp = map2wav_real(self.prior["eps"] * self.prior["outside"] * m, self.filt)[0]
        yp[self.prior["poff"]] = 0
        return yp

    def apply_full(self, x: Mapset) -> Mapset:
        """
        The full operator: M2 (unless --act-only) + ACT + prior.

        Parameters
        ----------
        x : Mapset
            Needlet coefficients.

        Returns
        -------
        Mapset
            The operator applied to x.
        """
        if self.cfg.act_only:  # the ACT term alone, no TODs
            y = x.copy()
            y.maps[0].map[:] = self.act.dot(x.maps[0].map)
            return y
        y = self.todvec.dot(x)
        if self.act is not None:
            y.maps[0].map += self.act.dot(x.maps[0].map)
        if self.prior is not None:
            xp = x.maps[0].map.copy()
            xp[self.prior["poff"]] = 0
            y.maps[0].map += self.prior_dot_map(wav2map_real(xp, self.filt)[0])
        return y

    def project(self, m: Mapset) -> Mapset:
        """
        With --restrict, zero the coefficients whose needlet does not reach the footprint (in place).

        Parameters
        ----------
        m : Mapset
            Needlet coefficients.

        Returns
        -------
        Mapset
            m.
        """
        if self.keep is not None:
            m.maps[0].map *= self.keep
        return m

    def apply(self, x: Mapset) -> Mapset:
        """
        The operator CG solves with (the full operator, restricted with --restrict).

        Parameters
        ----------
        x : Mapset
            Needlet coefficients.

        Returns
        -------
        Mapset
            The operator applied to x.
        """
        if self.keep is None:
            return self.apply_full(x)
        return self.project(self.apply_full(self.project(x.copy())))

    def precon(self, r: Mapset) -> Mapset:
        """
        The block solve as preconditioner (restricted with --restrict).

        Parameters
        ----------
        r : Mapset
            Residual.

        Returns
        -------
        Mapset
            Preconditioned residual.
        """
        z = self.project(r.copy())
        z.maps[0].map[:] = self.solve(z.maps[0].map)
        return self.project(z)

    def report(self, i: int, x: Mapset, info: dict, pixel: bool = False) -> None:
        """
        CG callback: print the residual and save the real space map at the --cg-save iterations.

        Parameters
        ----------
        i : int
            Iteration (from 0).
        x : Mapset
            Current solution (needlet coefficients, or the real space map with pixel).
        info : dict
            From pcg, holds "relres".
        pixel : bool, default: False
            x is already a real space map.
        """
        print(
            f"CG iter {i + 1}: |r|/|b| = {info['relres']:.2e} ({time.time() - self.t0:.0f} s)"
        )
        if (i + 1) in self.cfg.cg_save and myrank == 0:
            m = x.maps[0].map if pixel else wav2map_real(x.maps[0].map, self.filt)[0]
            np.save(f"{self.cfg.outroot}_cg{i + 1}.npy", m * self.mask)

    def cg_solve(self) -> NDArray | None:
        """
        CG on the full (band coupled) operator with the block solve as preconditioner, and the diagnostics asked for
        (--symcheck, --restrict, --consistent, --ritz, --resdiag, --pixel-cg).

        Returns
        -------
        NDArray | None
            The solution as a wavelet array, None after --symcheck.
        """
        cfg = self.cfg
        self.t0 = time.time()
        self.solve = self.wmap.block_solver(
            self.responses[-1:] if cfg.act_only else self.responses,
            ridge=cfg.ridge,
            inner=cfg.inner,
            lmin_cache=self.cache + "_lmin.json",
        )
        print(f"factorized the block preconditioner in {time.time() - self.t0:.1f} s")
        for resp in self.responses:  # the solver holds what it needs, free the rest
            resp[:] = None
        act_dot = None if self.act is None else self.act.dot
        if cfg.symcheck:
            symmetry_check(self.todvec.dot, act_dot, self.rhs, self.solve)
            return None
        if (
            cfg.restrict is not None
        ):  # only the coefficients whose needlet reaches the footprint
            self.keep = coefficient_support(
                self.filt, footprint_of(self.mask, self.filt.shape[1:]), cfg.restrict
            )
            print("coefficients kept per band:", self.keep.mean(axis=(1, 2)).round(3))
        b = self.rhs.copy()
        if cfg.act_only:
            b.maps[0].map[:] = 0
        b.maps[0].map += self.act_rhs + self.prior_rhs
        self.project(b)
        x0 = b.copy()
        x0.clear()
        if (
            cfg.consistent
        ):  # a rhs in the range of the operator, to see if the real one is not
            b = self.apply(pcg(self.apply, b, x0, self.precon, cfg.consistent))
        if cfg.pixel_cg:
            return pixel_solve(
                cfg.pixel_cg,
                self.apply,
                self.todvec.dot,
                act_dot,
                self.solve,
                b,
                self.wmap,
                self.hits.map,
                None if self.act is None else self.act.ivar,
                self.mask,
                cfg.cg,
                cfg.cg_tol,
                lambda i, x, info: self.report(i, x, info, pixel=True),
            )
        lan = {} if cfg.ritz else None
        xsol = pcg(
            self.apply,
            b,
            x0,
            self.precon,
            cfg.cg,
            cfg.cg_tol,
            self.report,
            cfg.flexible,
            lan,
        )
        if cfg.ritz:
            ritz_analysis(lan, self.filt, self.mask, cfg.outroot, xsol.maps[0].map)
        if cfg.resdiag:
            residual_diagnostics(
                self.apply,
                b,
                xsol,
                self.act_rhs,
                self.act_bands,
                cfg.outroot,
                self.filt,
                self.mask,
            )
        return xsol.maps[0].map

    def write(self, wsol: NDArray) -> None:
        """
        Write the solution: _combined.fits (the solve) and _final.fits (with the PCG map added back unless --direct).

        Parameters
        ----------
        wsol : NDArray
            Solution as a wavelet array.
        """
        mmap = wav2map_real(wsol, self.filt)[0] * self.mask
        self.wmap.real_map.map = mmap
        if myrank == 0:
            self.wmap.real_map.write(self.cfg.outroot + "_combined.fits")
            # Unless --direct, the combined solution is for the residual after PCG: add the PCG map back.
            self.wmap.real_map.map = mmap if self.cfg.direct else mmap + self.pcg_map
            self.wmap.real_map.write(self.cfg.outroot + "_final.fits")

    def run(self) -> None:
        """
        The whole pipeline.
        """
        self.read()
        self.prepare()
        self.initial_pcg()
        self.m2_response()
        self.act_term()
        t0 = time.time()
        if self.cfg.cg:
            self.prior_term()
            wsol = self.cg_solve()
            if wsol is None:
                return
        else:
            wsol = self.wmap.solve_bands(
                self.rhs.maps[0].map + self.act_rhs, self.responses
            )
        print(f"solve took {time.time() - t0:.1f} s")
        self.write(wsol)


def parse_args(
    description: str | None = None, argv: Sequence[str] | None = None
) -> argparse.Namespace:
    """
    Command line options of JointMapper, parsed and normalised (--cg implies --direct, --radius < 0 means exact,
    --debug makes a small problem).

    Parameters
    ----------
    description : str | None, default: None
        Help text of the program.
    argv : Sequence[str] | None, default: None
        Arguments (default sys.argv).

    Returns
    -------
    argparse.Namespace
        The options.
    """
    p = argparse.ArgumentParser(description=description)
    a = p.add_argument
    a("--name", default="MOOJ1142")
    a("--idir", default="/mnt/welch/MUSTANG/M2-TODs/MOOJ1142/mustang2/")
    a("--outroot", default="/mnt/welch/USERS/jorlo/needlets/MOOJ1142")
    a("--cache", default=".", help="Directory of the response matrix caches")
    a(
        "--act-glob",
        default="/mnt/welch/USERS/jorlo/maps/dr6_20230316_splits/cmb_night_*_f090_3pass_4way_set*_ivar.fits",
    )
    a(
        "--act-conv",
        type=float,
        default=None,
        help="ACT map units (uK_CMB) to M2 units; default from --freq, --m2-unit",
    )
    a("--freq", type=float, default=90e9, help="Hz, for the unit conversion")
    a(
        "--m2-unit",
        default="K_RJ",
        choices=["K_RJ", "Jy", "mJy"],
        help="Unit of the M2 TODs (K_RJ: Rayleigh-Jeans temperature; Jy, mJy: per M2 beam)",
    )
    a("--m2-beam", type=float, default=9.0, help="M2 beam FWHM in arcsec")
    a("--taper", type=int, default=0, help="Pixels; edge taper of the ACT weights")
    a(
        "--pad-frac",
        type=float,
        default=0.25,
        help="Map padding, fraction of data extent",
    )
    a(
        "--act-ivar-scale",
        type=float,
        default=1.0,
        help="Factor on the ACT inverse variance (and its rhs), to test the weight of ACT against M2; 0 for M2 only",
    )
    a(
        "--act-cmb",
        type=float,
        default=0.0,
        help="Count this multiple of the lensed CMB power (CAMB) as noise in the ACT map on top of its white noise",
    )
    a(
        "--ridge",
        type=float,
        default=1e-3,
        help="Diagonal shift of the block preconditioner, relative to each band",
    )
    a(
        "--lattice-tol",
        type=float,
        default=None,
        help="Choose each band's lattice from the support of its filter (window above this) so that it is not "
        "undersampled; default keeps the lattice of get_need_lims. The caches get a tag.",
    )
    a(
        "--sim-sky",
        default=None,
        help="Simulate: replace the M2 and ACT data by this sky (FITS) plus noise",
    )
    a(
        "--sim-noise",
        choices=["white", "flip"],
        default="flip",
        help="Simulated M2 noise: white at the level of each detector, or the data with alternating signs",
    )
    a("--sim-seed", type=int, default=0, help="Seed of the simulated noise")
    a(
        "--outside-prior",
        type=float,
        default=None,
        help="Add a prior eps |P (m_B - m0_B)|^2: region P (hits < --prior-min-hits, or --prior-mask), bands B "
        "(--prior-bands or --prior-non-act, default all), mean m0 (--prior-mean, default 0); eps is this fraction "
        "of the M2 weight per pixel (e.g. 1e-2). Default: none",
    )
    a(
        "--pixel-cg",
        choices=["needlet", "jacobi"],
        default=None,
        help="With --cg: solve for the real space map instead of the needlet coefficients, with the needlet block "
        "solve or a Jacobi (hits and ACT inverse variance) preconditioner",
    )
    a(
        "--prior-non-act",
        action="store_true",
        help="Apply --outside-prior only to the bands without ACT",
    )
    a(
        "--prior-bands",
        type=int,
        nargs="+",
        default=None,
        help="Apply --outside-prior only to these bands",
    )
    a(
        "--prior-min-hits",
        type=float,
        default=20,
        help="--outside-prior acts where hits < this",
    )
    a(
        "--prior-mask",
        default=None,
        help="FITS image; --outside-prior acts where it is > 0.5",
    )
    a(
        "--prior-mean",
        default=None,
        help="FITS map (M2 units) whose prior bands --outside-prior pulls toward",
    )
    a(
        "--prior-mean-scale",
        type=float,
        default=1.0,
        help="Factor applied to --prior-mean (e.g. a unit conversion)",
    )
    a(
        "--restrict",
        type=float,
        default=None,
        help="Solve only for the coefficients whose needlet reaches the footprint; the needlet extent is where its "
        "kernel exceeds this fraction of its peak (e.g. 1e-2). Default: all coefficients",
    )
    a(
        "--ritz",
        action="store_true",
        help="Spectrum of the preconditioned operator from the CG Lanczos data",
    )
    a("--flexible", action="store_true", help="Flexible (Polak-Ribiere) CG")
    a("--act-only", action="store_true", help="With --cg: solve the ACT term alone")
    a(
        "--resdiag",
        action="store_true",
        help="After the CG, print the true residual per band and save it as a map",
    )
    a("--act-beam", type=float, default=2.1, help="ACT beam FWHM in arcmin")
    a(
        "--act-min-transfer",
        type=float,
        default=0.05,
        help="Minimum ACT beam transfer of a band for ACT to be used",
    )
    a("--ntods", type=int, default=2, help="TODs per MPI rank")
    a(
        "--tod-list",
        default=None,
        help="Text file of TOD paths to use instead of --idir and the blacklist",
    )
    a("--nfilt", type=int, default=10)
    a("--pixsize", type=float, default=3.0, help="Arcseconds")
    a("--maxiter", type=int, default=50, help="Iterations of the initial M2 PCG")
    a(
        "--min-hits",
        type=float,
        default=20,
        help="Mask pixels with fewer hits from mapmaking (0 = no mask)",
    )
    a("--max-res", type=float, default=5.0, help="Arcsec; smallest scale for ANA")
    a("--radius", type=float, default=30.0, help="Pixels; ANA window, <0 for exact")
    a(
        "--exact-below",
        type=int,
        default=5000,
        help="Bands with at most this many lattice points computed exactly",
    )
    a(
        "--direct",
        action="store_true",
        help="Final map = ANA^-1 rhs of the full M2 (and ACT) data in all bands; PCG only cleans the noise estimate",
    )
    a(
        "--cg",
        type=int,
        default=0,
        help="CG iterations on the full operator (block solve as preconditioner)",
    )
    a(
        "--symcheck",
        action="store_true",
        help="With --cg: print the symmetry of the operator and exit",
    )
    a(
        "--consistent",
        type=int,
        default=0,
        metavar="N",
        help="With --cg: replace the rhs by A x, x after N CG",
    )
    a(
        "--cg-save",
        type=int,
        nargs="*",
        default=[],
        help="Iterations at which to save <outroot>_cg<N>.npy",
    )
    a("--cg-tol", type=float, default=1e-3, help="Relative residual to stop at")
    a(
        "--inner",
        type=int,
        default=20,
        help="Jacobi CG iterations of the large bands in the preconditioner",
    )
    a(
        "--ntods-total",
        type=int,
        help="Use only this many TODs in total (after the blacklist)",
    )
    a("--redo", action="store_true", help="Recompute cached responses")
    a("--debug", action="store_true", help="Fewer TODs/iterations, coarser pixels")
    cfg = p.parse_args(argv)
    cfg.direct |= cfg.cg > 0
    if cfg.radius < 0:
        cfg.radius = None
    if cfg.debug:
        cfg.ntods, cfg.maxiter, cfg.pixsize, cfg.nfilt = 1, 3, 6.0, 6
        cfg.max_res = None if cfg.max_res is None else min(cfg.max_res, 60.0)
        cfg.cache = os.path.join(cfg.cache, "debug")
        os.makedirs(cfg.cache, exist_ok=True)
    return cfg
