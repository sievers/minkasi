"""
Test the block diagonal solve of WavSkyMap.solve_bands against PCG on a small M2-only problem.

1. Ridge sweep: solve each band with its exact response matrix G_i and a range of ridges, and compare
   the maps to a converged PCG map of the same TODs and noise model.
2. Preconditioned CG on the full (band coupled) operator A^T N^-1 A, with the block solve as the
   preconditioner, compared to the same reference.
Needs the exact M2 response caches of twores_ex.py (computed here if missing).
"""

import argparse
import os

import minkasi.minkasi_all as minkasi
import numpy as np
from minkasi.maps.mapset import Mapset
from minkasi.needlet.needlet import band2map, map2band, wav2map_real
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter

from twores_ex import cached, pcg, read_tods, setup_wmap


class BlockSolver:
    """
    Dense eigendecomposition version of WavSkyMap.solve_bands for one set of response matrices, so
    that the ridge can be changed for free.

    Parameters
    ----------
    wmap : WavSkyMap
        Wavelet map the matrices belong to.
    resp : NDArray[object]
        Response matrix (sparse) per band, None for bands to skip.
    """

    def __init__(self, wmap, resp):
        self.wmap, self.bands = wmap, [i for i, g in enumerate(resp) if g is not None]
        self.eig = {}
        for i in self.bands:
            w, v = eigh(resp[i].toarray())
            self.eig[i] = (w, v, resp[i].diagonal().mean())

    def lmin(self) -> dict[int, float]:
        """Smallest eigenvalue of each G_i in units of its mean diagonal."""
        return {i: w[0] / d for i, (w, _, d) in self.eig.items()}

    def solve(self, rhs: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
        """
        Block diagonal solve F_i^2 (G_i + shift)^-1 F_i^2 r as in solve_bands.

        Parameters
        ----------
        rhs : np.ndarray
            Wavelet space rhs, shape (nfilt, nx, ny).
        ridge : float, default: 1e-3
            Ridge relative to the mean diagonal of G_i; G_i is also shifted by its most negative
            eigenvalue if it is indefinite.

        Returns
        -------
        wsol : np.ndarray
            Wavelet map, shape (nfilt, nx, ny).
        """
        wm = self.wmap
        out = np.zeros((wm.needlet.nfilt, wm.nx, wm.ny))
        for i, (w, v, dmean) in self.eig.items():
            filt = wm.needlet.filters[i]
            xs, ys = wm.nx_space[i], wm.ny_space[i]
            b = map2band(rhs[i], filt)[np.ix_(xs, ys)].ravel()
            shift = ridge * dmean + max(0, -1.1 * w[0])
            c = v @ ((v.T @ b) / (w + shift))
            coeffs = np.zeros((wm.nx, wm.ny))
            coeffs[np.ix_(xs, ys)] = c.reshape(len(xs), len(ys))
            out[i] = band2map(coeffs, filt)
        return out


def metrics(maps: dict[str, np.ndarray], ref: str, good: np.ndarray):
    """
    Print noise and agreement with a reference map in the good pixels.

    Parameters
    ----------
    maps : dict[str, np.ndarray]
        Real space maps by name.
    ref : str
        Key of the reference map.
    good : np.ndarray
        Boolean mask of the pixels to use.
    """

    w = good.astype(float)

    def smooth(m, s):  # footprint normalised Gaussian smoothing
        return gaussian_filter(np.where(good, m, 0), s) / np.maximum(
            gaussian_filter(w, s), 1e-3
        )

    def hp(m):  # 1-5 pixel band-pass
        return (smooth(m, 1) - smooth(m, 5))[good]

    r = hp(maps[ref])
    print(
        f"{'map':>22} {'rms':>9} {'pixdiff':>9} {'bp rms':>9} {'bp corr':>8} {'bp slope':>9}"
    )
    for k, m in maps.items():
        d = (m[:, 1:] - m[:, :-1])[good[:, 1:] & good[:, :-1]]
        b = hp(m)
        print(
            f"{k:>22} {m[good].std():9.2e} {d.std() / np.sqrt(2):9.2e} {b.std():9.2e} "
            f"{np.corrcoef(r, b)[0, 1]:8.3f} {np.polyfit(r, b, 1)[0]:9.3f}"
        )


def say(*a):
    """Print on MPI rank 0 only."""
    if minkasi.myrank == 0:
        print(*a, flush=True)


def probe_masked(wmap, todvec, mask, bands: list[int]):
    """
    Exact response matrices of the masked operator M A^T N^-1 A M, the one the PCG solves and
    whose rhs get_response_matrix does not see (it probes the unmasked real space map).

    Parameters
    ----------
    wmap : WavSkyMap
        Wavelet map, with downsamps set.
    todvec : TodVec
        TODs with noise models.
    mask : np.ndarray
        Real space boolean mask.
    bands : list[int]
        Bands to compute.

    Returns
    -------
    resp : NDArray[object]
        Sparse matrix per band, None for the others.
    """
    mapset = Mapset()
    mapset.add_map(wmap.real_map.copy())

    def apply(imap):
        mapset.maps[0].map[:] = imap * mask
        return todvec.dot(mapset, cache_maps=False).maps[0].map * mask

    resp = np.full(wmap.needlet.nfilt, None, dtype=object)
    for i in bands:
        resp[i] = wmap.probe_band(apply, i, None)
    return resp


def main(args: argparse.Namespace):
    """
    Run both tests.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    todvec = read_tods(args.idir, args.ntods, args.name)
    wmap = setup_wmap(todvec, args.pixsize, args.nfilt, args.pad_frac)
    filt = wmap.needlet.filters
    for tod in todvec.tods:
        tod.info["ipix"] = wmap.get_pix(tod)
        tod.set_noise(minkasi.NoiseSmoothedSVD)
    hits = minkasi.make_hits(todvec, wmap.real_map)
    wmap.set_mask(hits.map, args.min_hits - 1)
    mask = wmap.mask
    good = mask > 0
    wmap.get_downsamps()

    mapset = Mapset()
    mapset.add_map(wmap)
    rhs = mapset.copy()
    todvec.make_rhs(rhs)
    x0 = rhs.copy()
    x0.clear()
    inv = hits.map.copy()
    inv[inv > 0] = 1 / inv[inv > 0]
    hit_precon = mapset.copy()
    hit_precon.maps[0].map[:] = inv

    # noise model cleaned of signal, as twores_ex --direct
    model = minkasi.run_pcg_wprior(
        rhs, x0, todvec, precon=hit_precon, maxiter=args.maxiter
    )
    for tod in todvec.tods:
        tmp = np.zeros(tod.info["dat_calib"].shape)
        model.maps[0].map2tod(tod, tmp)
        tod.set_noise(minkasi.NoiseSmoothedSVD, dat=tod.info["dat_calib"] - tmp)
    rhs = mapset.copy()
    todvec.make_rhs(rhs)

    def real(x):
        return wav2map_real(x.maps[0].map, filt)[0] * mask

    more = minkasi.run_pcg_wprior(
        rhs, model, todvec, precon=hit_precon, maxiter=args.nref
    )
    maps = {"pcg 50": real(model), f"pcg {args.maxiter + args.nref}": real(more)}

    bands = wmap.select_filters(None)
    resp = {
        "unmasked G": cached(
            f"{args.cache}/{args.name}_m2",
            lambda b: wmap.get_response_matrix(
                todvec, nfilts=b, radius=None, exact_below=10**9
            ),
            False,
            wmap.needlet.nfilt,
            bands,
        ),
        "masked G": cached(
            f"{args.cache}/{args.name}_m2masked_p{args.pad_frac}",
            lambda b: probe_masked(wmap, todvec, mask, b),
            False,
            wmap.needlet.nfilt,
            bands,
        ),
    }
    solvers = {k: BlockSolver(wmap, r) for k, r in resp.items()}
    i = solvers["masked G"].bands[-1]
    xs, ys = wmap.nx_space[i], wmap.ny_space[i]
    c = np.random.default_rng(0).normal(size=len(xs) * len(ys))
    coeffs = np.zeros((wmap.nx, wmap.ny))
    coeffs[np.ix_(xs, ys)] = c.reshape(len(xs), len(ys))
    probe = mapset.copy()
    probe.clear()
    probe.maps[0].map[i] = coeffs  # wav2map_real applies F_i
    y = (
        todvec.dot(probe).maps[0].map[i][np.ix_(xs, ys)].ravel()
    )  # map2wav_real applied F_i
    for k, s in solvers.items():
        say(f"{k}: smallest eigenvalue / mean diagonal per band")
        say({b: f"{v:+.1e}" for b, v in s.lmin().items()})
        gc = resp[k][i] @ c
        say(
            f"  band {i} |A x - G x|/|G x| = {np.linalg.norm(y - gc) / np.linalg.norm(gc):.2e}"
        )

    # Test 2 first: the converged CG on the full operator is the reference of both tests
    ref = "CG ref (masked G, converged)"
    for k, s in solvers.items():
        say(
            f"\nTest 2: CG on the full operator, block solve with {k} (ridge 1e-3) as preconditioner"
        )
        maps2 = {}

        def block_precon(r):
            z = r.copy()
            z.maps[0].map[:] = s.solve(r.maps[0].map, 1e-3)
            return z

        def report(i, x, relres):
            if (i + 1) in args.cg_report:
                say(f"  iter {i + 1}: |r|/|b| = {relres:.2e}")
                maps2[f"block-CG {i + 1}"] = real(x)

        pcg(todvec.dot, rhs, x0, block_precon, max(args.cg_report), 0.0, report)
        last = maps2[f"block-CG {max(args.cg_report)}"]
        if k == "masked G":
            maps[ref] = last
        maps2 = {ref: maps.get(ref, last), **maps2}
        if minkasi.myrank == 0:
            metrics(maps2, ref, good)

    say("\nTest 1: block solve, exact G_i, ridge sweep (M2 only), vs the converged CG")
    for k, s in solvers.items():
        for ridge in args.ridges:
            maps[f"{k[:2]} G ridge {ridge:g}"] = (
                wav2map_real(s.solve(rhs.maps[0].map, ridge), filt)[0] * mask
            )
    if minkasi.myrank == 0:
        metrics({ref: maps.pop(ref), **maps}, ref, good)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="MOOJ1142")
    parser.add_argument(
        "--idir", default="/mnt/welch/MUSTANG/M2-TODs/MOOJ1142/mustang2/"
    )
    parser.add_argument(
        "--cache", required=True, help="Directory of the exact M2 response cache"
    )
    parser.add_argument("--ntods", type=int, default=4)
    parser.add_argument("--pixsize", type=float, default=12.0, help="Arcseconds")
    parser.add_argument(
        "--pad-frac", type=float, default=0.25, help="Map padding, as twores_ex"
    )
    parser.add_argument("--nfilt", type=int, default=8)
    parser.add_argument("--min-hits", type=float, default=20)
    parser.add_argument(
        "--maxiter", type=int, default=50, help="PCG steps for the noise model"
    )
    parser.add_argument(
        "--nref",
        type=int,
        default=50,
        help="Further PCG steps after the noise model ones",
    )
    parser.add_argument(
        "--ridges", type=float, nargs="+", default=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    )
    parser.add_argument(
        "--cg-report", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40]
    )
    parser.add_argument(
        "--debug", action="store_true", help="One TOD, coarse pixels, few iterations"
    )
    args = parser.parse_args()
    if args.debug:
        args.ntods, args.pixsize, args.nfilt = 1, 24.0, 6
        args.maxiter, args.nref, args.cg_report = 5, 20, [1, 3]
        args.cache = args.cache + "/debug"
    os.makedirs(args.cache, exist_ok=True)
    main(args)
