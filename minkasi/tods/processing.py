from typing import overload, Literal, Iterable
import numpy as np
from numpy.typing import NDArray
from . import Tod, CutsCompact
from .. import mkfftw
from ..utils import find_good_fft_lens


def _linfit_2mat(dat, mat1, mat2):
    np1 = mat1.shape[1]
    np2 = mat2.shape[1]
    mm = np.append(mat1, mat2, axis=1)
    lhs = np.dot(mm.transpose(), mm)
    rhs = np.dot(mm.transpose(), dat)
    lhs_inv = np.linalg.inv(lhs)
    fitp = np.dot(lhs_inv, rhs)
    fitp1 = fitp[0:np1].copy()
    fitp2 = fitp[np1:].copy()
    assert len(fitp2) == np2
    return fitp1, fitp2


def find_spikes(
    dat: NDArray[np.floating], inner: float = 1, outer: float = 10, thresh: float = 8
) -> tuple[list[list[int]], NDArray[np.floating]]:
    """
    Find spikes in a block of timestreams using a difference of gaussians filter.

    Parameters
    ----------
    dat : NDArray[np.floating]
        Data to find spikes in along each row.
        Assumed to be 2d.
    inner : float, default: 1
        Sigma of smaller gaussian for filter in samples.
    outer : float, default: 10
        Sigma of larger gaussian for filter in samples.
    thresh : float, default: 8
        Threshold in units of filtered data median absolute deviation to qualify as a spike.

    Returns
    -------
    spikes : list[list[int]]
        List of lists, each list corresponds to a row of dat with the indices of spikes for that row.
    datfilt : NDArray[np.floating]
        The filtered data with spike locations set to 0.
    """
    ndet, n = dat.shape
    x = np.arange(n)

    # Smaller gaussian
    filt1 = np.exp(-0.5 * x**2 / inner**2)
    filt1 = filt1 + np.exp(-0.5 * (x - n) ** 2 / inner**2)
    filt1 = filt1 / filt1.sum()

    # Larger gaussian
    filt2 = np.exp(-0.5 * x**2 / outer**2)
    filt2 = filt2 + np.exp(-0.5 * (x - n) ** 2 / outer**2)
    filt2 = filt2 / filt2.sum()

    # Apply the difference of gaussian filter
    filt = filt1 - filt2
    filtft = np.fft.rfft(filt)
    datft = np.fft.rfft(dat, axis=1)
    datfilt = np.fft.irfft(filtft * datft, axis=1, n=n)

    spikes = [[]] * ndet
    deviation = np.median(np.abs(datfilt), axis=1)
    for i in range(ndet):
        ind = np.where(np.abs(datfilt[i, :]) > thresh * deviation[i])[0]
        spikes[i] = list(ind)
        datfilt[i, ind] = 0
    return spikes, datfilt


@overload
def find_jumps(
    dat: NDArray[np.floating],
    width: int = 10,
    pad: int = 2,
    thresh: float = 10,
    rat: float = 0.5,
    dejump: Literal[False] = False,
) -> list[list[int]]:
    ...


@overload
def find_jumps(
    dat: NDArray[np.floating],
    width: int = 10,
    pad: int = 2,
    thresh: float = 10,
    rat: float = 0.5,
    dejump: Literal[True] = True,
) -> tuple[list[list[int]], NDArray[np.floating]]:
    ...


@overload
def find_jumps(
    dat: NDArray[np.floating],
    width: int = 10,
    pad: int = 2,
    thresh: float = 10,
    rat: float = 0.5,
    dejump: bool = False,
) -> list[list[int]] | tuple[list[list[int]], NDArray[np.floating]]:
    ...


def find_jumps(
    dat: NDArray[np.floating],
    width: int = 10,
    pad: int = 2,
    thresh: float = 10,
    rat: float = 0.5,
    dejump: bool = False,
) -> list[list[int]] | tuple[list[list[int]], NDArray[np.floating]]:
    """
    Find jumps in a block of timestreams, preferably with the common mode removed.

    Parameters
    ----------
    dat : NDArray[np.floating]
        Data to find jumps in along each row.
        Assumed to be 2d.
    width : int, default: 10
        Width in pixels to average over when looking for a jump.
    pad : int, default: 2
        The length in units of width to mask at beginning/end of timestream.
    thresh : float, default: 10
        Threshold in units of filtered data median absolute deviation to qualify as a jump.
    rat : float, default: .5
        The ratio of largest neighboring opposite-sign jump to the found jump.
        If there is an opposite-sign jump nearby, the jump finder has probably just picked up a spike.
    dejump : bool, default: False
        If True return dejumped data.

    Returns
    -------
    jumps: list[list[int]]
        List of lists, each list corresponds to a row of dat with the indices of jumps for that row.
    dat_dejump : NDArray[np.floating]
        The data with jumps removed.
        Only returned if dejump is True.
    """
    ndet, n = dat.shape

    # make a filter template that is a gaussian with sigma with, sign-flipped in the center
    # so, positive half-gaussian starting from zero, and negative half-gaussian at the end
    x = np.arange(n)
    filt: NDArray[np.floating] = np.exp(-0.5 * x**2 / width**2)
    filt_sub: NDArray[np.floating] = np.exp((-0.5 * (x - n) ** 2 / width**2))
    filt -= filt_sub
    fac = np.abs(filt).sum() / 2.0
    filt /= fac

    dat_filt = np.fft.rfft(dat, axis=1)

    filt_ft = np.fft.rfft(filt)
    dat_filt = dat_filt * np.repeat([filt_ft], ndet, axis=0)
    dat_filt = np.fft.irfft(dat_filt, axis=1, n=n)
    # dat_filt_org=dat_filt.copy()

    print(dat_filt.shape)
    dat_filt[:, 0 : pad * width] = 0
    dat_filt[:, -pad * width :] = 0
    det_thresh = thresh * np.median(np.abs(dat_filt), axis=1)
    dat_dejump = dat
    if dejump:
        dat_dejump = dat.copy()
    jumps = [[]] * ndet
    print("have filtered data, now searching for jumps")
    for i in range(ndet):
        while np.max(np.abs(dat_filt[i, :])) > det_thresh[i]:
            ind = (
                np.argmax(np.abs(dat_filt[i, :])) + 1
            )  # +1 seems to be the right index to use
            imin = max(int(ind - width), 0)
            imax = min(int(ind + width), n)
            val = dat_filt[i, ind]
            if val > 0:
                val2 = np.min(dat_filt[i, imin:imax])
            else:
                val2 = np.max(dat_filt[i, imin:imax])

            print("found jump on detector ", i, " at sample ", ind)
            if np.abs(val2 / val) > rat:
                print("I think this is a spike due to ratio ", np.abs(val2 / val))
            else:
                jumps[i].append(ind)
            # independent of if we think it is a spike or a jump, zap that stretch of the data
            if dejump:
                dat_dejump[i, ind:] = dat_dejump[i, ind:] + dat_filt[i, ind]
            dat_filt[i, ind - pad * width : ind + pad * width] = 0
        jumps[i].sort()
    if dejump:
        return jumps, dat_dejump
    return jumps


def fit_jumps_from_cm(
    dat: NDArray[np.floating],
    jumps: list[list[int]],
    cm: NDArray[np.floating],
    cm_order: int = 1,
    poly_order: int = 1,
) -> NDArray[np.floating]:
    """
    Use common mode to fit jumps out from data.

    Parameters
    ----------
    dat : NDArray[np.floating]
        Data to remove jumps in along each row.
        Assumed to be 2d.
    jumps: list[list[int]]
        List of lists, each list corresponds to a row of dat with the indices of jumps for that row.
    cm : NDArray[np.floating]
        The common mode of the data.
    cm_order : int, default: 1
        Order of the legvander to multiply the common mode by.
    poly_order : int, default: 1
        Order of the other legvander.

    Returns
    -------
    dat_dejump : NDArray[np.floating]
        The data with jumps removed.
        Only returned if dejump is True.
    """
    jump_vals = jumps[:]
    ndet, n = dat.shape
    x = np.linspace(-1, 1, n)
    m1 = np.polynomial.legendre.legvander(x, poly_order)
    m2 = np.polynomial.legendre.legvander(x, cm_order - 1)
    for i in range(cm_order):
        m2[:, i] = m2[:, i] * cm
    mat = np.append(m1, m2, axis=1)
    npp = mat.shape[1]

    dat_dejump = dat.copy()
    for i in range(ndet):
        njump = len(jumps[i])
        segs = np.append(jumps[i], n)
        print(
            "working on detector ",
            i,
            " who has ",
            len(jumps[i]),
            " jumps with segments ",
            segs,
        )
        mm = np.zeros([n, npp + njump])
        mm[:, :npp] = mat
        for j in range(njump):
            mm[segs[j] : segs[j + 1], j + npp] = 1.0
        lhs = np.dot(mm.transpose(), mm)
        rhs = np.dot(mm.transpose(), dat[i, :].transpose())
        lhs_inv = np.linalg.inv(lhs)
        fitp = np.dot(lhs_inv, rhs)
        jump_vals[i] = fitp[npp:]
        jump_pred = np.dot(mm[:, npp:], fitp[npp:])
        dat_dejump[i, :] = dat_dejump[i, :] - jump_pred

    return dat_dejump


def gapfill_eig(
    dat: NDArray[np.floating],
    cuts: CutsCompact,
    tod: Tod | None = None,
    thresh: float = 5.0,
    niter_eig: int = 3,
    niter_inner: int = 3,
    insert_cuts: bool = False,
) -> CutsCompact:
    """
    Use eigenmodes to fill in gaps.

    Parameters
    ----------
    dat : NDArray[np.floating]
        Data to gapfill.
    cuts : CutsCompact
        CutsCompact for this data.
    tod : Tod | None, default: None
        The tod to pass to map2tod and tod2map.
        Not actually used in a meaningful way in the current state so None is fine.
    thresh : float, default: 5
        Threshold in units of the eigenmode median used to make eigenmode mask.
        Gets squared for some reason.
    niter_eig : int, default: 3
        Number of iterations of finding eigenmodes to do.
    niter_inner : int, default: 3
        Number of iterations of fitting the data to do.
        This is per eigenmode iteration.
    insert_cuts : bool, default: False
        If True, add the gapfilled cuts map into dat.

    Returns
    -------
    cuts_cur : CutsCompact
        Cuts with gapfilling applied.
    """
    # use this to clear out cut samples
    cuts_cur = cuts.copy()
    if cuts_cur.map is None:
        cuts.get_imap()
    cuts_cur.clear()
    for _ in range(niter_eig):
        tmp = dat.copy()
        cuts_cur.map2tod(tod, tmp, do_add=False)
        mycov = np.dot(tmp, tmp.T)
        ee, vv = np.linalg.eig(mycov)
        mask = ee > thresh * thresh * np.median(ee)
        neig = np.sum(mask)
        print("working with " + repr(neig) + " eigenvectors.")
        ee = ee[mask]
        vv = vv[:, mask]
        uu = np.dot(vv.T, tmp)
        lhs = np.dot(uu, uu.T)
        lhs_inv = np.linalg.inv(lhs)
        for __ in range(niter_inner):
            # in this inner loop, we fit the data
            rhs = np.dot(tmp, uu.T)
            fitp = np.dot(lhs_inv, rhs.T)
            pred = np.dot(fitp.T, uu)
            cuts_cur.tod2map(tod, pred, do_add=False)
            cuts_cur.map2tod(tod, tmp, do_add=False)
    if insert_cuts:
        cuts_cur.map2tod(dat)
    return cuts_cur


def __gapfill_eig_poly(
    dat, cuts, tod=None, npoly=2, thresh=5.0, niter_eig=3, niter_inner=3
):
    assert (
        1 == 0
    )  # this code is not yet working.  regular gapfill_eig should work since the polys could
    # be described by SVD, so SVD modes should look like polys iff they would have been important
    ndat = dat.shape[1]
    if npoly > 0:
        xvec = np.linspace(-1, 1, ndat)
        polymat = np.polynomial.legendre.legvander(x, npoly - 1)
    old_coeffs = None
    cuts_cur = cuts.copy()
    cuts_cur.clear()
    cuts_empty.cuts.copy()
    cuts_empty.clear()
    for eig_ctr in range(niter_eig):
        tmp = dat.copy()
        cuts_cur.map2tod(
            tod, tmp, do_add=False
        )  # insert current best-guess solution for the cuts
        if (
            npoly > 1
        ):  # if we're fitting polynomials as well as eigenmodes, subtract them off before re-estimating the covariance
            if not (old_coeffs is None):
                tmp = tmp - np.dot(polymat, old_coeffs[neig:, :]).T
        mycov = np.dot(tmp, tmp.T)
        mycov = 0.5 * (mycov + mycov.T)
        ee, vv = np.linalg.eig(mycov)
        mode_map = ee > thresh * thresh * np.median(ee)
        neig = mode_map.sum()
        mat = np.zeros([ndat, neig + npoly])
        eigs = vv[:, mode_map]
        ts_vecs = np.dot(eigs.T, tmp)
        mat[:, :neig] = ts_vecs.T
        if npoly > 0:
            mat[:, neig:] = polymat
        lhs = np.dot(mat.T, mat)
        lhs_inv = np.linalg.inv(lhs)
        # now that we have the vectors we expect to describe our data, do a few rounds
        # of fitting amplitudes to timestream models, subtract that off, assign cuts to zero,
        # and restore the model.
        tmp = dat.copy()
        for inner_ctr in range(niter_inner):
            cuts_cur.map2tod(tod, tmp)
            rhs = np.dot(tmp, mat)
            fitp = np.dot(lhs_inv, rhs.T)
            pred = np.dot(mat, fitp).T


@overload
def fit_cm_plus_poly(
    dat: NDArray[np.floating],
    ord: int = 2,
    cm_ord: int = 1,
    niter: int = 1,
    medsub: bool = False,
    full_out: Literal[False] = False,
) -> NDArray[np.floating]:
    ...


@overload
def fit_cm_plus_poly(
    dat: NDArray[np.floating],
    ord: int = 2,
    cm_ord: int = 1,
    niter: int = 1,
    medsub: bool = False,
    full_out: Literal[True] = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    ...


@overload
def fit_cm_plus_poly(
    dat: NDArray[np.floating],
    ord: int = 2,
    cm_ord: int = 1,
    niter: int = 1,
    medsub: bool = False,
    full_out: bool = False,
) -> NDArray[np.floating] | tuple[
    NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]
]:
    ...


def fit_cm_plus_poly(
    dat: NDArray[np.floating],
    ord: int = 2,
    cm_ord: int = 1,
    niter: int = 1,
    medsub: bool = False,
    full_out: bool = False,
) -> NDArray[np.floating] | tuple[
    NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]
]:
    """
    Fit the common mode with polynomials for drifts across the focal plane.

    Parameters
    ----------
    dat : NDArray[np.floating]
        The data to fit common mode out of.
        Should be (ndet, ndata).
    ord : int, default: 2
        Order of the legvander that is used as the non common mode polynomial.
    cm_ord : int, default: 2
        Order of the legvander used for the common mode.
    niter : int, default: 1
    medsub : bool, default: False
        If True, subtract the median before fitting.
        Shouldn'd really make a difference since the CM should include the median.
    full_out : bool, default: False
        If True also return pred2 and cm

    Returns
    -------
    dd : NDArray[np.floating]
        The data with the polynomial drifts subtracted.
    pred2 : NDArray[np.floating]
        The polynomial common mode.
    cm : NDArray[np.floating]
        The median common mode.
    """
    ndet, n = dat.shape
    if medsub:
        med = np.median(dat, axis=1)
        dat = dat - np.repeat([med], n, axis=0).transpose()

    xx = np.arange(n) + 0.0
    xx = xx - xx.mean()
    xx = xx / xx.max()

    pmat = np.polynomial.legendre.legvander(xx, ord)
    cm_pmat = np.polynomial.legendre.legvander(xx, cm_ord - 1)
    calfacs = np.ones(ndet) * 1.0
    dd = dat.copy()
    pred2 = np.zeros(n)
    cm = np.zeros(n)
    for i in range(niter):
        for j in range(ndet):
            dd[j, :] /= calfacs[j]

        cm = np.median(dd, axis=0)
        cm_mat = np.zeros(cm_pmat.shape)
        for i in range(cm_mat.shape[1]):
            cm_mat[:, i] = cm_pmat[:, i] * cm
        fitp_p, fitp_cm = _linfit_2mat(dat.transpose(), pmat, cm_mat)
        pred1 = np.dot(pmat, fitp_p).transpose()
        pred2 = np.dot(cm_mat, fitp_cm).transpose()
        dd = dat - pred1

    if full_out:
        return dd, pred2, cm  # if requested, return the modelled CM as well
    return dd


def find_bad_skew_kurt(
    dat: NDArray[np.floating], skew_thresh: float = 6.0, kurt_thresh: float = 5.0
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]:
    """
    Find detectors with high skew or kurtosis.

    Parameters
    ----------
    dat : NDArray[np.floating]
        The data to look for bad detectors in.
    skew_thresh : float, default: 6
        The minimum bad skew in units of average skew.
    kurt_thresh : float, default: 5
        The minimum bad kurtosis in units of average kurtosis.

    Returns
    -------
    skew : NDArray[np.floating]
        The skew of each detctor.
    kurt : NDArray[np.floating]
        The kurtosis of each detector.
    isgood : NDArray[np.bool_]
        Boolean mask of size (ndet,).
        True for detectors that are below both the skew and kurtosis threshold.
    """
    ndet = dat.shape[0]
    isgood = np.ones(ndet, dtype="bool")
    skew = np.mean(dat**3, axis=1)
    mystd = np.std(dat, axis=1)
    skew = skew / mystd**1.5
    mykurt = np.mean(dat**4, axis=1)
    kurt = mykurt / mystd**4 - 3

    isgood[np.abs(skew) > skew_thresh * np.median(np.abs(skew))] = False
    isgood[np.abs(kurt) > kurt_thresh * np.median(np.abs(kurt))] = False

    return skew, kurt, isgood


def downsample_array_r2r(
    arr: NDArray[np.floating], fac: int, axis: int = -1
) -> NDArray[np.floating]:
    """
    Downsample array using fourier transform.

    Parameters
    ----------
    arr : NDArray[np.floating]
        Array to downsample.
    fac : int
        Factor to downsample by.
    axis : int, default: -1
        The axis to downsample along.

    Returns
    -------
    downsampled_arr : NDArray[np.floating]
        The downsampled array.
    """
    n = arr.shape[axis]
    nn = int(n / fac)
    arr_ft = mkfftw.fft_r2r(arr)
    arr_ft = np.take(arr_ft, indices=range(0, nn), axis=axis)
    downsampled_arr = mkfftw.fft_r2r(arr_ft) / (2 * (n - 1))
    return downsampled_arr


def downsample_vec_r2r(vec: NDArray[np.floating], fac: int) -> NDArray[np.floating]:
    """
    Just a wrapped around downsample_array_r2r to support legacy code.
    """
    return downsample_array_r2r(vec, fac)


def downsample_tod(dat: dict, fac: int = 10):
    """
    Downsample a TOD using fourier transforms.
    Only fields that have size ndata along their last axis are downsampled,
    where ndata is the number of samples in each row of dat['dat_calib'].
    If dat_calib isn't in dat then all arrays in dat are downsampled.

    Parameters
    ----------
    dat : dict
        The TOD to downsample.
        If you have a Tod object you probably want to pass in tod.info.
    fac : int, default: 10
        The factor to downsample by.
    """
    ndata = dat["dat_calib"].shape[1]
    for key in dat.keys():
        if hasattr(dat[key], "shape"):
            if dat[key].shape[-1] != ndata:
                continue
            dat[key] = downsample_array_r2r(dat[key], fac)


def truncate_tod(dat: dict, primes: Iterable[int] = [2, 3, 5, 7, 11]):
    """
    Truncate TOD to the closet good FFT length.
    Required 'dat_calib' to be in dat, if it isn't nothing is done.

    Parameters
    ----------
    dat : dict
        The TOD to truncate.
        If you have a Tod object you probably want to pass in tod.info.
    primes : Iterable[int]
        Prime number to use to calculate good fft lengths.
    """
    if "dat_calib" not in dat:
        return
    n = dat["dat_calib"].shape[1]
    lens = find_good_fft_lens(n - 1, primes)
    n_new = lens.max() + 1
    if n_new >= n:
        return
    print("truncating from ", n, " to ", n_new)
    for key in dat.keys():
        if not hasattr(dat[key], "shape"):
            continue
        axes = np.where(np.array(dat[key].shape) == n)[0]
        if len(axes) == 0:
            continue
        axis = axes[0]
        dat[key] = np.take(dat[key], indices=range(0, n_new), axis=axis)


def decimate(
    vec: NDArray[np.floating], nrep: int = 1, axis: int = -1
) -> NDArray[np.floating]:
    """
    Decimate an array by a factor of 2^nrep.

    Parameters
    ----------
    vec : NDArray[np.floating]
        The array to decimate.
    nrep : int, default: 1
        The number of times to decimate the array
    axis : int, default: -1
        The axis to decimate along.

    Returns
    -------
    decimated : NDArray[np.floating]
        The decimated array.
    """
    even = [slice(None)] * len(vec.shape)
    odd = [slice(None)] * len(vec.shape)
    for _ in range(nrep):
        end = vec.shape[axis]
        if end % 2:
            end -= 1
        even[axis] = slice(0, end, 2)
        odd[axis] = slice(1, end, 2)
        vec = 0.5 * (vec[tuple(even)] + vec[tuple(odd)])
    return vec


def fit_mat_vecs_poly_nonoise(
    dat: NDArray[np.floating],
    mat: NDArray[np.floating],
    order: int,
    cm_order: int | None = None,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Fit polynomial common mode to a matrix and data.
    This function needs a more descriptive docstring but I don't really understand its purpose.

    Parameters
    ----------
    dat : NDArray[np.floating]
        The data to fit.
    mat : NDArray[np.floating]
        The matrix to fit.
    order : int
        The order of the legvander to fit.
    cm_order : int | None, default: None
        The order of the common mode legvander.
        If None then order is used
    Returns
    -------
    pred : NDArray[np.floating]
        The combined common mode result.
    cm_fitp : NDArray[np.floating]
        The fit common mode parameters.
    mat_fitp : NDArray[np.floating]
        The fit parameters to the matrix.
    polys : NDArray[np.floating]
        The legvander used for for fitting.
    """
    if cm_order is None:
        cm_order = order
    n = dat.shape[1]
    x = np.linspace(-1, 1, n)
    polys = np.polynomial.legendre.legvander(x, order).transpose()
    cm_polys = np.polynomial.legendre.legvander(x, cm_order).transpose()
    v1 = np.sum(dat, axis=0)
    v2 = np.sum(dat * mat, axis=0)
    rhs1 = np.dot(cm_polys, v1)
    rhs2 = np.dot(polys, v2)
    ndet = dat.shape[0]
    A1 = cm_polys * ndet
    vv = np.sum(mat, axis=0)
    A2 = polys * np.repeat([vv], order + 1, axis=0)
    A = np.append(A1, A2, axis=0)
    rhs = np.append(rhs1, rhs2)
    lhs = np.dot(A, A.transpose())
    fitp = np.dot(np.linalg.inv(lhs), rhs)
    cm_fitp = fitp[: cm_order + 1]
    mat_fitp = fitp[cm_order + 1 :]
    assert len(mat_fitp) == (order + 1)
    cm_pred = np.dot(cm_fitp, cm_polys)
    tmp = np.dot(mat_fitp, polys)
    mat_pred = np.repeat([tmp], ndet, axis=0) * mat
    pred = cm_pred + mat_pred
    return pred, cm_fitp, mat_fitp, polys
