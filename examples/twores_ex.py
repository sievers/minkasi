"""
Two resolution (MUSTANG-2 TODs + ACT maps) needlet mapmaking example.

Steps: read TODs, run PCG in the needlet basis to remove small scales, compute the (sparse)
wavelet space noise matrix of the TODs and of ACT, add them and solve for the combined map.
"""

import argparse
import glob
import os
import time

import astropy.units as u
import minkasi.minkasi_all as minkasi
import numpy as np
from minkasi.maps.mapset import Mapset
from minkasi.maps.skymap import SkyMap
from minkasi.needlet.needlet import (
    Needlet,
    WavSkyMap,
    apply_beam,
    block_average,
    gaussian_beam_ft,
    highly_composite_size,
    map2wav_real,
    wav2map_real,
)
from pixell import enmap, utils
from scipy import sparse


def save_response(prefix, resp):
    """
    Save a list of per filter sparse matrices as {prefix}_{filter}.npz.

    Parameters
    ----------
    prefix : str
        Path prefix of the files.
    resp : NDArray[object]
        Sparse matrix per filter, None for those not computed.
    """
    for i, r in enumerate(resp):
        if r is not None:
            sparse.save_npz(f"{prefix}_{i}.npz", r)


def cached(prefix, compute, redo, nfilt, bands):
    """
    Load the response matrices of the requested bands from the cache at prefix and compute and
    save only those that are missing. The cache does not record how the matrices were made
    (TODs, radius, units, ...): use a new directory or redo when these change.

    Parameters
    ----------
    prefix : str
        Path prefix of the cache.
    compute : Callable
        Function taking a list of bands and returning the response matrices (see
        WavSkyMap.get_response_matrix).
    redo : bool
        Ignore any existing cache.
    nfilt : int
        Number of filters.
    bands : list[int]
        Bands needed.

    Returns
    -------
    resp : NDArray[object]
        Sparse matrix per filter, None for the bands not requested.
    """
    have = [i for i in bands if not redo and os.path.exists(f"{prefix}_{i}.npz")]
    missing = [i for i in bands if i not in have]
    resp = np.full(nfilt, None, dtype=object)
    for i in have:
        resp[i] = sparse.load_npz(f"{prefix}_{i}.npz")
    if missing:
        new = compute(missing)
        for i in missing:
            resp[i] = new[i]
        if minkasi.myrank == 0:
            save_response(prefix, new)
    return resp


def act_conversion(freq=90e9, beam_fwhm=9.0, unit="Jy"):
    """
    Factor converting ACT maps (uK CMB) to the units of the M2 data (unit/beam, RJ).

    Parameters
    ----------
    freq : float, default: 90e9
        Observing frequency in Hz.
    beam_fwhm : float, default: 9.0
        FWHM of the M2 beam in arcsec.
    unit : str, default: "Jy"
        Flux unit of the M2 TODs, "Jy" or "mJy". The median per sample noise of the MOOJ1142
        TODs is about 2 in their units, which is only realistic for Jy/beam.

    Returns
    -------
    float
        unit/beam per uK_CMB.
    """
    omega = (2 * np.pi * (beam_fwhm * u.arcsec / (8 * np.log(2)) ** 0.5) ** 2).to(u.sr)
    equiv = u.thermodynamic_temperature(freq * u.Hz, 2.7255 * u.K)
    return ((1 * u.uK).to(u.Jy / u.sr, equiv) * omega).to_value(u.Unit(unit))


def setup_wmap(todvec, pixsize, nfilt, pad_frac=0.0):
    """
    Build the needlet and the WavSkyMap covering the TODs. The map is square with a highly
    composite size so that filters can be downsampled by many factors; the bounds only need
    to contain the data.

    Parameters
    ----------
    todvec : TodVec
        TODs to cover.
    pixsize : float
        Pixel size in arcsec.
    nfilt : int
        Number of needlet filters.
    pad_frac : float, default: 0.0
        Fraction of the data extent added on every side. The filters are periodic, so
        this moves the artefacts of the map edges away from the data.

    Returns
    -------
    wmap : WavSkyMap
    """
    x0, x1, y0, y1 = todvec.lims()
    px, py = pad_frac * (x1 - x0), pad_frac * (y1 - y0)
    lims = [x0 - px, x1 + px, y0 - py, y1 + py]
    ps = pixsize / 3600 * np.pi / 180
    n0 = SkyMap(list(lims), ps, square=True, multiple=2).map.shape[0]
    n = highly_composite_size(n0)
    box = SkyMap(list(lims), ps, nx=n, ny=n).map
    need = Needlet(
        np.arange(nfilt), lightcone=box, L=n * pixsize * np.sqrt(2), pixsize=pixsize
    )
    need.get_needlet_filters_2d(
        need.lightcone_box.get_grid_dimless_2d(return_grid=True)
    )
    return WavSkyMap(need, lims, ps, nx=n, ny=n)


def read_tods(idir, ntods, name=None, total=None):
    """
    Read, downsample and common mode subtract MUSTANG-2 TODs, splitting files between MPI ranks.

    Parameters
    ----------
    idir : str
        Directory holding Sig*.fits.
    ntods : int
        Maximum number of TODs to read on this rank.
    name : str | None, default: None
        Source name; if given, the TODs blacklisted for it in minkasi.get_bad_tods are skipped.
    total : int | None, default: None
        If given, only the first `total` TODs (after the blacklist) are used, in all ranks together.

    Returns
    -------
    todvec : TodVec
    """
    tod_names = sorted(glob.glob(os.path.join(idir, "Sig*.fits")))
    if name is not None:
        tod_names = minkasi.cut_blacklist(tod_names, minkasi.get_bad_tods(name)[0])
    tod_names = tod_names[:total][minkasi.myrank :: minkasi.nproc][:ntods]
    todvec = minkasi.TodVec()
    for fname in tod_names:
        t1 = time.time()
        dat = minkasi.read_tod_from_fits(fname)
        minkasi.truncate_tod(dat)  # chop samples from the end to make a happy length
        minkasi.downsample_tod(
            dat
        )  # sometimes we have faster sampled data than we need
        minkasi.truncate_tod(dat)
        dat["dat_calib"] = minkasi.fit_cm_plus_poly(dat["dat_calib"])
        todvec.add_tod(minkasi.Tod(dat))
        print(f"read {fname} in {time.time() - t1:.1f} s")
    return todvec


def edge_taper(shape, width):
    """
    Window that is 1 in the interior and falls to 0 at the edges with a cosine ramp.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the map.
    width : int
        Ramp width in pixels. 0 gives no taper.

    Returns
    -------
    NDArray[np.floating]
        2D window of the given shape.
    """
    if width <= 0:
        return np.ones(shape)

    def ramp(n):
        r = np.ones(n)
        edge = 0.5 * (1 - np.cos(np.pi * (np.arange(width) + 0.5) / width))
        r[:width], r[-width:] = edge, edge[::-1]
        return r

    return ramp(shape[0])[:, None] * ramp(shape[1])[None, :]


ACT_OSAMP = 10  # fine pixels per ACT pixel


def read_act(paths, lims, wmap, hits, fac=ACT_OSAMP, conv=1.0):
    """
    Read ACT split maps and inverse variance maps in a box around the field and
    coadd them onto the (fine) wavelet map pixelization.

    Parameters
    ----------
    paths : list[str]
        ACT ivar maps. The corresponding maps are found by replacing _ivar with _map.
    lims : list[float]
        Field limits (radians).
    wmap : WavSkyMap
        Map whose real_map pixelization to use.
    hits : SkyMap
        Hit map, used as the mask of the coarse to fine mapping.
    fac : int, default: 10
        Oversampling factor of the ACT pixels.
    conv : float, default: 1.0
        Factor converting ACT map units to the units of the M2 data. Maps are multiplied by
        it and ivar divided by its square.

    Returns
    -------
    ivar : NDArray[np.floating]
        Total inverse variance on the wmap pixelization, shape (nx, ny).
    wdata : NDArray[np.floating]
        Sum over splits of ivar * map on the wmap pixelization, shape (nx, ny).
    """
    ra, dec = np.mean(lims[:2]), np.mean(lims[2:])
    rad = 0.5 * utils.degree
    box = np.array([np.array([dec, ra]) - rad, np.array([dec, ra]) + rad])
    ivar = np.zeros_like(wmap.real_map.map)
    wdata = np.zeros_like(wmap.real_map.map)
    for path in paths:
        ivar_map = enmap.read_map(path, box=box) / conv**2
        act_map = enmap.read_map(path.replace("_ivar", "_map"), box=box) * conv
        if act_map.ndim == 3:
            act_map = act_map[0]  # temperature
        two_res = minkasi.SkyMapTwoRes(
            np.array(ivar_map), lims, fac, big_wcs=ivar_map.wcs
        )
        two_res.set_beam_gauss(3)
        two_res.set_mask(hits.map)
        ivar += two_res.coarse2fine(wmap.real_map.map, ivar_map)
        wdata += two_res.coarse2fine(wmap.real_map.map, ivar_map * act_map)
    return ivar, wdata


def symmetry_check(m2_dot, act_dot, like, solve=None):
    """
    Print the relative asymmetry |<y, A x> - <A y, x>| / (|y| |A x|) of the M2 operator, the ACT term
    and their sum for random vectors, as a CG on a non symmetric operator stalls. With a
    preconditioner, also the asymmetry of that and the sign of <x, P x> for several vectors.

    Parameters
    ----------
    m2_dot : Callable[[Mapset], Mapset]
        The M2 operator, e.g. TodVec.dot.
    act_dot : Callable[[Mapset], np.ndarray] | None
        The ACT term acting on a Mapset, returning a wavelet array, or None without ACT.
    like : Mapset
        Mapset to take the shape from.
    solve : Callable[[np.ndarray], np.ndarray] | None, default: None
        The preconditioner acting on wavelet arrays.
    """
    rng = np.random.default_rng(0)  # the same vectors on every rank
    x, y = like.copy(), like.copy()
    x.maps[0].map[:] = rng.normal(size=x.maps[0].map.shape)
    y.maps[0].map[:] = rng.normal(size=y.maps[0].map.shape)
    ops = {"M2": lambda v: m2_dot(v).maps[0].map}
    if act_dot is not None:
        ops["ACT"] = act_dot
        ops["M2+ACT"] = lambda v: ops["M2"](v) + act_dot(v)
    if solve is not None:
        ops["precon"] = lambda v: solve(v.maps[0].map)
    for name, op in ops.items():
        ax, ay = op(x), op(y)
        a, b = np.sum(y.maps[0].map * ax), np.sum(ay * x.maps[0].map)
        err = abs(a - b) / (np.linalg.norm(y.maps[0].map) * np.linalg.norm(ax))
        if minkasi.myrank == 0:
            print(f"{name}: <y,Ax> = {a:.6e}, <Ay,x> = {b:.6e}, asymmetry {err:.2e}")
    if solve is not None and minkasi.myrank == 0:
        xs = [x.maps[0].map, y.maps[0].map] + [
            rng.normal(size=x.maps[0].map.shape) for _ in range(3)
        ]
        print(
            "<x,Px> / |x|^2:",
            [f"{np.sum(v * solve(v)) / np.sum(v * v):.2e}" for v in xs],
        )


def pcg(apply, rhs, x0, precon, niter, tol=0.0, callback=None):
    """
    Preconditioned CG on a Mapset with callable operator and preconditioner.

    Parameters
    ----------
    apply : Callable[[Mapset], Mapset]
        The operator, e.g. TodVec.dot.
    rhs : Mapset
        Right hand side.
    x0 : Mapset
        Starting guess.
    precon : Callable[[Mapset], Mapset]
        Preconditioner, a symmetric positive definite approximation of the inverse of apply.
    niter : int
        Maximum number of iterations.
    tol : float, default: 0
        Stop when |r| / |rhs| is below this.
    callback : Callable | None, default: None
        Called as callback(i, x, relres) after every iteration.

    Returns
    -------
    x : Mapset
    """
    bnorm = np.sqrt(rhs.dot(rhs))
    x, r = x0.copy(), rhs.copy()
    r.axpy(apply(x), -1)
    z = precon(r)
    p, zr = z.copy(), r.dot(z)
    for i in range(niter):
        Ap = apply(p)
        alpha = zr / p.dot(Ap)
        x.axpy(p, alpha)
        r.axpy(Ap, -alpha)
        z = precon(r)
        zr_new = r.dot(z)
        p_new = z.copy()
        p_new.axpy(p, zr_new / zr)
        p, zr = p_new, zr_new
        relres = np.sqrt(r.dot(r)) / bnorm
        if callback:
            callback(i, x, relres)
        if relres < tol:
            break
    return x


def main(args):
    """
    Run the whole pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    todvec = read_tods(args.idir, args.ntods, args.name, args.ntods_total)
    wmap = setup_wmap(todvec, args.pixsize, args.nfilt, args.pad_frac)
    need = wmap.needlet

    for tod in todvec.tods:
        tod.info["ipix"] = wmap.get_pix(tod)
        tod.set_noise(minkasi.NoiseSmoothedSVD)
    hits = minkasi.make_hits(todvec, wmap.real_map)
    if minkasi.myrank == 0:
        hits.write(args.outroot + "_hits.fits")
    minkasi.barrier()
    if args.min_hits > 0:
        wmap.set_mask(hits.map, args.min_hits - 1)  # mask pixels with hits < min_hits
    mask = 1.0 if wmap.mask is None else wmap.mask

    # Run a few PCG iterations to get the small scales to converge, then subtract them.
    mapset = Mapset()
    mapset.add_map(wmap)
    rhs = mapset.copy()
    todvec.make_rhs(rhs)
    x0 = rhs.copy()
    x0.clear()
    precon = mapset.copy()
    tmp = hits.map.copy()
    ii = tmp > 0
    tmp[ii] = 1.0 / tmp[ii]
    precon.maps[0].map[:] = tmp
    print("Saving to ", args.outroot)
    mapset_out = minkasi.run_pcg_wprior(
        rhs,
        x0,
        todvec,
        precon=precon,
        maxiter=args.maxiter,
        save_iters=[args.maxiter],
        outroot=args.outroot,
    )
    pcg_map = wav2map_real(mapset_out.maps[0].map, need.filters)[0] * mask
    if minkasi.myrank == 0:
        mapset_out.maps[0].write(args.outroot + "_need.fits")
    minkasi.barrier()

    for tod in todvec.tods:
        tmp = np.zeros(tod.info["dat_calib"].shape)
        mapset_out.maps[0].map2tod(tod, tmp)
        if (
            args.direct
        ):  # the PCG model only cleans the noise estimate, the data keep the signal
            tod.set_noise(minkasi.NoiseSmoothedSVD, dat=tod.info["dat_calib"] - tmp)
        else:
            tod.info["dat_calib"] -= tmp
            tod.set_noise(minkasi.NoiseSmoothedSVD)

    rhs = mapset.copy()
    todvec.make_rhs(rhs)  # wavelet basis rhs, of the residual unless --direct
    wmap.get_downsamps()
    cache = os.path.join(args.cache, args.name)
    # window of the response matrices: a few filter extents, never more than --radius
    radii = [
        min(
            args.radius or np.inf,
            2 * need.get_need_lims(i, real_space=True)[1] / args.pixsize,
        )
        for i in range(need.nfilt)
    ]
    kw = {"max_res": args.max_res, "radius": None if args.radius is None else radii}
    kw_bands = wmap.select_filters(None if args.direct else args.max_res)
    t0 = time.time()
    m2_resp = cached(
        cache + "_m2",
        lambda b: wmap.get_response_matrix(
            todvec,
            nfilts=b,
            radius=kw["radius"],
            exact_below=args.exact_below,
        ),
        args.redo,
        need.nfilt,
        kw_bands,
    )

    print(f"M2 response took {time.time() - t0:.1f} s")
    responses = [m2_resp]
    act_bands = []
    act_rhs = 0
    t0 = time.time()
    paths = sorted(glob.glob(args.act_glob))
    if args.debug:
        paths = paths[:2]
    if paths:
        t1 = time.time()
        # TODO: ivar is treated as a diagonal map space inverse covariance. Replace this with a
        # robust map space noise covariance estimate once we have one (see tools.noise_estimation).
        conv = args.act_conv or act_conversion(args.freq, args.m2_beam, args.m2_unit)
        ivar, wdata = read_act(paths, wmap.lims, wmap, hits, conv=conv)
        if (
            minkasi.myrank == 0
        ):  # ACT only (inverse variance weighted coadd) for comparison
            wmap.real_map.map = np.divide(
                wdata, ivar, out=np.zeros_like(wdata), where=ivar > 0
            )
            wmap.real_map.write(args.outroot + "_act.fits")
        # The filters are periodic but the ACT map is not: taper the weights at the edges
        taper = edge_taper(ivar.shape, args.taper)
        ivar, wdata = ivar * taper, wdata * taper
        # ACT sees the sky through its beam and the PCG model is already removed from the M2
        # TODs, so ACT enters through its residual with respect to the beam smoothed PCG map.
        beam_ft = gaussian_beam_ft(ivar.shape, args.act_beam * 60 / args.pixsize)

        def forward(m):  # beam, then the ACT pixel average
            return block_average(apply_beam(m, beam_ft), ACT_OSAMP)

        def adjoint(m):
            return apply_beam(block_average(m, ACT_OSAMP), beam_ft)

        # ACT only constrains the bands its beam transmits; deconvolving the others amplifies noise
        fwhm_pix = args.act_beam * 60 / args.pixsize
        transfer = wmap.band_transfer(fwhm_pix)
        act_bands = [i for i in kw_bands if transfer[i] >= args.act_min_transfer]
        print("ACT used in bands", act_bands, "beam transfer", transfer.round(3))
        resid = wdata if args.direct else wdata - ivar * forward(pcg_map)
        act_rhs = map2wav_real(adjoint(resid), wmap.needlet.filters)[0]
        act_rhs[[i for i in range(need.nfilt) if i not in act_bands]] = 0
        print(f"reading ACT took {time.time() - t1:.1f} s")
        responses.append(
            cached(
                f"{cache}_act_t{args.taper}_c{conv:.3g}_b{args.act_beam}_pw_tr{args.act_min_transfer}",
                lambda b: wmap.get_response_matrix_map(
                    ivar,
                    forward=forward,
                    adjoint=adjoint,
                    nfilts=b,
                    radius=kw["radius"],
                    exact_below=args.exact_below,
                ),
                args.redo,
                need.nfilt,
                act_bands,
            )
        )

    print(f"ACT response took {time.time() - t0:.1f} s")
    t0 = time.time()
    if args.cg:
        # CG on the full (band coupled) operator, the block solve is only the preconditioner
        filt = wmap.needlet.filters
        off = [i for i in range(need.nfilt) if i not in act_bands]

        def act_dot(x):
            """ACT term of the operator, on its bands only."""
            xa = x.maps[0].map.copy()
            xa[off] = 0
            ya = map2wav_real(adjoint(ivar * forward(wav2map_real(xa, filt)[0])), filt)[
                0
            ]
            ya[off] = 0
            return ya

        def apply(x):
            y = todvec.dot(x)
            if paths:
                y.maps[0].map += act_dot(x)
            return y

        solve = wmap.block_solver(responses, inner=args.inner)
        print(f"factorized the block preconditioner in {time.time() - t0:.1f} s")
        if args.symcheck:
            symmetry_check(todvec.dot, act_dot if paths else None, rhs, solve)
            return

        def precon(r):
            z = r.copy()
            z.maps[0].map[:] = solve(r.maps[0].map)
            return z

        def report(i, x, relres):
            print(f"CG iter {i + 1}: |r|/|b| = {relres:.2e} ({time.time() - t0:.0f} s)")
            if (i + 1) in args.cg_save and minkasi.myrank == 0:
                np.save(
                    f"{args.outroot}_cg{i + 1}.npy",
                    wav2map_real(x.maps[0].map, filt)[0] * mask,
                )

        b = rhs.copy()
        b.maps[0].map += act_rhs
        x0 = b.copy()
        x0.clear()
        if (
            args.consistent
        ):  # a rhs in the range of the operator, to see if the real one is not
            b = apply(pcg(apply, b, x0, precon, args.consistent))
        wsol = pcg(apply, b, x0, precon, args.cg, args.cg_tol, report).maps[0].map
    else:
        wsol = wmap.solve_bands(rhs.maps[0].map + act_rhs, responses)
    mmap = wav2map_real(wsol, wmap.needlet.filters)[0] * mask
    print(f"solve took {time.time() - t0:.1f} s")
    wmap.real_map.map = mmap
    if minkasi.myrank == 0:
        wmap.real_map.write(args.outroot + "_combined.fits")
        # Unless --direct, the combined solution is for the residual after PCG: add the PCG map back.
        wmap.real_map.map = mmap if args.direct else mmap + pcg_map
        wmap.real_map.write(args.outroot + "_final.fits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="MOOJ1142")
    parser.add_argument(
        "--idir", default="/mnt/welch/MUSTANG/M2-TODs/MOOJ1142/mustang2/"
    )
    parser.add_argument("--outroot", default="/mnt/welch/USERS/jorlo/needlets/MOOJ1142")
    parser.add_argument(
        "--cache", default=".", help="Directory of the response matrix caches"
    )
    parser.add_argument(
        "--act-glob",
        default="/mnt/welch/USERS/jorlo/maps/dr6_20230316_splits/cmb_night_*_f090_3pass_4way_set*_ivar.fits",
    )
    parser.add_argument(
        "--act-conv",
        type=float,
        default=None,
        help="ACT map units (uK_CMB) to M2 units; default computed from --freq and --m2-beam",
    )
    parser.add_argument(
        "--freq", type=float, default=90e9, help="Hz, for the unit conversion"
    )
    parser.add_argument(
        "--m2-unit",
        default="Jy",
        choices=["Jy", "mJy"],
        help="Flux unit of the M2 TODs",
    )
    parser.add_argument(
        "--m2-beam", type=float, default=9.0, help="M2 beam FWHM in arcsec"
    )
    parser.add_argument(
        "--taper", type=int, default=0, help="Pixels; edge taper of the ACT weights"
    )
    parser.add_argument(
        "--pad-frac",
        type=float,
        default=0.25,
        help="Map padding, fraction of data extent",
    )
    parser.add_argument(
        "--act-beam", type=float, default=2.1, help="ACT beam FWHM in arcmin"
    )
    parser.add_argument(
        "--act-min-transfer",
        type=float,
        default=0.05,
        help="Minimum ACT beam transfer of a band for ACT to be used in it",
    )
    parser.add_argument("--ntods", type=int, default=2)
    parser.add_argument("--nfilt", type=int, default=10)
    parser.add_argument("--pixsize", type=float, default=3.0, help="Arcseconds")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument(
        "--min-hits",
        type=float,
        default=20,
        help="Mask pixels with fewer hits from mapmaking and the output maps (0 = no mask)",
    )
    parser.add_argument(
        "--max-res", type=float, default=5.0, help="Arcsec; smallest scale for ANA"
    )
    parser.add_argument(
        "--radius", type=float, default=30.0, help="Pixels; ANA window, <0 for exact"
    )
    parser.add_argument(
        "--exact-below",
        type=int,
        default=5000,
        help="Bands with at most this many lattice points are computed exactly (one TOD pass per point)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Final map = ANA^-1 rhs of the full M2 (and ACT) data in all bands, ignoring "
        "--max-res; PCG is only used to clean the noise estimate",
    )
    parser.add_argument(
        "--cg",
        type=int,
        default=0,
        help="Solve with this many CG iterations on the full (M2 + ACT) operator, preconditioned "
        "by the block solve, instead of the block solve alone. Implies --direct",
    )
    parser.add_argument(
        "--symcheck",
        action="store_true",
        help="With --cg: print the symmetry of the operator and exit",
    )
    parser.add_argument(
        "--consistent",
        type=int,
        default=0,
        metavar="N",
        help="With --cg: replace the rhs by A x, x the result of N CG iterations on the real one",
    )
    parser.add_argument(
        "--cg-save",
        type=int,
        nargs="*",
        default=[],
        help="With --cg: iterations at which to save the real space map to <outroot>_cg<N>.npy",
    )
    parser.add_argument(
        "--cg-tol", type=float, default=1e-3, help="Relative residual to stop at"
    )
    parser.add_argument(
        "--inner",
        type=int,
        default=20,
        help="Jacobi CG iterations of the large bands in the --cg preconditioner",
    )
    parser.add_argument(
        "--ntods-total",
        type=int,
        help="Use only this many TODs in total (after the blacklist), whatever the number of ranks",
    )
    parser.add_argument(
        "--redo", action="store_true", help="Recompute cached responses"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Fewer TODs/iterations, coarser pixels"
    )
    args = parser.parse_args()
    args.direct |= args.cg > 0
    if args.radius < 0:
        args.radius = None
    if args.debug:
        args.ntods, args.maxiter, args.pixsize, args.nfilt = 1, 3, 6.0, 6
        args.max_res = None if args.max_res is None else min(args.max_res, 60.0)
        args.cache = os.path.join(args.cache, "debug")
        os.makedirs(args.cache, exist_ok=True)
    main(args)
