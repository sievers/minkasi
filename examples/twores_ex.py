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
import numpy as np
from pixell import enmap, utils
from scipy import sparse

import minkasi.minkasi_all as minkasi
from minkasi.maps.mapset import Mapset
from minkasi.maps.skymap import SkyMap
from minkasi.needlet.needlet import (
    Needlet,
    WavSkyMap,
    highly_composite_size,
    map2wav_real,
    wav2map_real,
)


def load_response(prefix, nfilt):
    """
    Load a list of per filter sparse matrices saved by save_response.

    Parameters
    ----------
    prefix : str
        Path prefix of the files.
    nfilt : int
        Number of filters.

    Returns
    -------
    resp : NDArray[object]
        Sparse matrix per filter, None where nothing was saved.
    """
    resp = np.full(nfilt, None, dtype=object)
    for f in glob.glob(prefix + "_*.npz"):
        resp[int(f.rsplit("_", 1)[1][:-4])] = sparse.load_npz(f)
    return resp


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


def cached(prefix, compute, redo, nfilt):
    """
    Load the response matrices from prefix if present, otherwise compute and save them.

    Parameters
    ----------
    prefix : str
        Path prefix of the cache.
    compute : Callable
        Function with no arguments returning the response matrices.
    redo : bool
        Ignore any existing cache.
    nfilt : int
        Number of filters.
    """
    if not redo and glob.glob(prefix + "_*.npz"):
        return load_response(prefix, nfilt)
    resp = compute()
    if minkasi.myrank == 0:
        save_response(prefix, resp)
    return resp


def act_conversion(freq=90e9, beam_fwhm=9.0):
    """
    Factor converting ACT maps (uK CMB) to the units of the M2 data (mJy/beam, RJ).

    Parameters
    ----------
    freq : float, default: 90e9
        Observing frequency in Hz.
    beam_fwhm : float, default: 9.0
        FWHM of the M2 beam in arcsec.

    Returns
    -------
    float
        mJy/beam per uK_CMB.
    """
    omega = (2 * np.pi * (beam_fwhm * u.arcsec / (8 * np.log(2)) ** 0.5) ** 2).to(u.sr)
    equiv = u.thermodynamic_temperature(freq * u.Hz, 2.7255 * u.K)
    return ((1 * u.uK).to(u.mJy / u.sr, equiv) * omega).to_value(u.mJy)


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


def read_tods(idir, ntods):
    """
    Read, downsample and common mode subtract MUSTANG-2 TODs, splitting files between MPI ranks.

    Parameters
    ----------
    idir : str
        Directory holding Sig*.fits.
    ntods : int
        Maximum number of TODs to read on this rank.

    Returns
    -------
    todvec : TodVec
    """
    tod_names = sorted(glob.glob(os.path.join(idir, "Sig*.fits")))
    tod_names = tod_names[minkasi.myrank :: minkasi.nproc][:ntods]
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


def read_act(paths, lims, wmap, hits, fac=10, conv=1.0):
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


def main(args):
    """
    Run the whole pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    todvec = read_tods(args.idir, args.ntods)
    wmap = setup_wmap(todvec, args.pixsize, args.nfilt, args.pad_frac)
    need = wmap.needlet

    for tod in todvec.tods:
        tod.info["ipix"] = wmap.get_pix(tod)
        tod.set_noise(minkasi.NoiseSmoothedSVD)
    hits = minkasi.make_hits(todvec, wmap.real_map)

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
    if minkasi.myrank == 0:
        mapset_out.maps[0].write(args.outroot + "_need.fits")
    minkasi.barrier()

    for tod in todvec.tods:
        tmp = np.zeros(tod.info["dat_calib"].shape)
        mapset_out.maps[0].map2tod(tod, tmp)
        tod.info["dat_calib"] -= tmp
        tod.set_noise(minkasi.NoiseSmoothedSVD)

    rhs = mapset.copy()
    todvec.make_rhs(rhs)  # residual rhs in the wavelet basis
    wmap.get_downsamps()
    cache = os.path.join(args.cache, args.name)
    kw = dict(max_res=args.max_res, radius=args.radius)
    t0 = time.time()
    m2_resp = cached(
        cache + "_m2",
        lambda: wmap.get_response_matrix(todvec, **kw),
        args.redo,
        need.nfilt,
    )

    print(f"M2 response took {time.time() - t0:.1f} s")
    responses = [m2_resp]
    act_rhs = 0
    t0 = time.time()
    paths = sorted(glob.glob(args.act_glob))
    if args.debug:
        paths = paths[:2]
    if paths:
        t1 = time.time()
        # TODO: ivar is treated as a diagonal map space inverse covariance. Replace this with a
        # robust map space noise covariance estimate once we have one (see tools.noise_estimation).
        conv = args.act_conv or act_conversion(args.freq, args.m2_beam)
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
        act_rhs = map2wav_real(wdata, wmap.needlet.filters)[0]
        print(f"reading ACT took {time.time() - t1:.1f} s")
        responses.append(
            cached(
                f"{cache}_act_t{args.taper}_c{conv:.3g}",
                lambda: wmap.get_response_matrix_map(ivar, **kw),
                args.redo,
                need.nfilt,
            )
        )

    print(f"ACT response took {time.time() - t0:.1f} s")
    t0 = time.time()
    wsol = wmap.solve_bands(rhs.maps[0].map + act_rhs, responses)
    mmap = wav2map_real(wsol, wmap.needlet.filters)[0]
    print(f"solve took {time.time() - t0:.1f} s")
    wmap.real_map.map = mmap
    if minkasi.myrank == 0:
        wmap.real_map.write(args.outroot + "_combined.fits")


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
    parser.add_argument("--ntods", type=int, default=2)
    parser.add_argument("--nfilt", type=int, default=10)
    parser.add_argument("--pixsize", type=float, default=3.0, help="Arcseconds")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument(
        "--max-res", type=float, default=60.0, help="Arcsec; smallest scale for ANA"
    )
    parser.add_argument(
        "--radius", type=float, default=30.0, help="Pixels; ANA window, <0 for exact"
    )
    parser.add_argument(
        "--redo", action="store_true", help="Recompute cached responses"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Fewer TODs/iterations, coarser pixels"
    )
    args = parser.parse_args()
    if args.radius < 0:
        args.radius = None
    if args.debug:
        args.ntods, args.maxiter, args.pixsize, args.nfilt = 1, 3, 6.0, 6
        args.max_res = None if args.max_res is None else min(args.max_res, 60.0)
        args.cache = os.path.join(args.cache, "debug")
        os.makedirs(args.cache, exist_ok=True)
    main(args)
