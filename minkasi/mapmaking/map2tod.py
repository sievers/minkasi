from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ..lib.minkasi import (
    map2tod_iqu_omp_c,
    map2tod_omp_c,
    map2tod_qu_omp_c,
    map2tod_simple_c,
)

try:
    import numba as nb
except ImportError:
    from ..tools import no_numba as nb


def map2tod(
    dat: NDArray[np.floating],
    map: NDArray[np.floating],
    ipix: NDArray[np.int32],
    do_add: bool = False,
    do_omp: bool = True,
):
    """
    Unroll a map into a a TOD inplace.
    This is for temperature map, for pol maps use polmap2tod.

    Parameters
    ----------
    dat : NDArray[np.floating]
        The TOD, will be modified in place.
        Should be ndet by ndata.
    map : NDArray[np.floating]
        The map to unroll.
    ipix : NDArray[np.int32]
        The pixellization matrix.
    do_add : bool, default: False
        If True then dat will be added to.
        If False dat will be overwriten.
    do_omp : bool, default: True
        If True then OMP us used as a speedup.
    """
    ndet = dat.shape[0]
    ndata = dat.shape[1]
    if do_omp:
        map2tod_omp_c(
            dat.ctypes.data, map.ctypes.data, ndet, ndata, ipix.ctypes.data, do_add
        )
    else:
        map2tod_simple_c(
            dat.ctypes.data, map.ctypes.data, ndet, ndata, ipix.ctypes.data, do_add
        )


def polmap2tod(
    dat: NDArray[np.floating],
    map: NDArray[np.floating],
    poltag: str,
    twogamma: NDArray[np.floating],
    ipix: NDArray[np.int32],
    do_add: bool = False,
    do_omp: bool = True,
):
    """
    Unroll a polmap into a a TOD inplace.
    This is for temperature map, for pol maps use polmap2tod.

    Parameters
    ----------
    dat : NDArray[np.floating]
        The TOD, will be modified in place.
        Should be ndet by ndata.
    map : NDArray[np.floating]
        The map to unroll.
    poltag : str
        The polarization tag.
        Valid options are:
            * QU
            * IQU
            * QU_PRECON (not implemented)
            * IQU_PRECON (not implemented)
    twogamma : NDArray[np.floating]
        Array of twogamma terms.
        This sets the weight between Q and U.
    ipix : NDArray[np.int32]
        The pixellization matrix.
    do_add : bool, default: False
        If True then dat will be added to.
        If False dat will be overwriten.
    do_omp : bool, default: True
        If True then OMP us used as a speedup.
    """
    ndet = dat.shape[0]
    ndata = dat.shape[1]
    fun = None
    if poltag == "QU":
        if not do_omp:
            print("No non OMP method implemented for this poltag, using OMP.")
        fun = map2tod_qu_omp_c
    elif poltag == "IQU":
        if not do_omp:
            print("No non OMP method implemented for this poltag, using OMP.")
        fun = map2tod_iqu_omp_c
    elif poltag == "QU_PRECON":
        raise NotImplementedError(f"Poltag {poltag} is not implemented")
        # fun=map2tod_qu_precon_omp_c
    elif poltag == "IQU_PRECON":
        raise NotImplementedError(f"Poltag {poltag} is not implemented")
        # fun=map2tod_iqu_precon_omp_c
    if fun is None:
        raise ValueError("unknown poltag " + repr(poltag) + " in polmap2tod.")
    fun(
        dat.ctypes.data,
        map.ctypes.data,
        twogamma.ctypes.data,
        ndet,
        ndata,
        ipix.ctypes.data,
        do_add,
    )


@nb.jit(nopython=True)
def map2todbowl(
    vecs: NDArray[np.floating], params: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Converts parameters to tods for the tsBowl class.

    Parameters
    ----------
    vecs: NDArray[np.floating]
        The pseudo-Vandermonde matrix.
        Should have shape (order, ndata, ndet)
    params: NDArray[np.floating]
        Corresponding weights for pseudo-Vandermonde matrix.
        Should have shape (order, ndet)

    Returns
    -------
    bowl_tod : NDArray[np.floating]
        The TOD of the bowl.
    """

    # Return tod should have shape ndet x ndata
    bowl_tod = np.zeros((vecs.shape[0], vecs.shape[-2]))
    for i in range(vecs.shape[0]):
        bowl_tod[i] = np.dot(vecs[i, ...], params[i, ...])

    return bowl_tod


@nb.njit(parallel=True)
def map2tod_destriped(
    mat: NDArray[np.floating],
    pars: NDArray[np.floating],
    lims: NDArray[np.int_],
    do_add: bool = True,
):
    """
    Unroll a destriper map into a TOD inplace.

    Parameters
    ----------
    mat : NDArray[np.floating]
        The array to fill with the TOD, should be (ndet, ndata).
    pars : NDArray[np.floating]
        The stripe parameters, should be (ndet, nseg).
    lims : NDArray[np.int_]
        The edges of the stripe segments, should be (nseg+1,).
    do_add : bool, default: True
        If True then the TOD is added to mat.
        If False the contents of mat are overwriten.
    """
    nseg = len(lims) - 1
    for seg in nb.prange(nseg):
        if do_add:
            mat[:, lims[seg] : lims[seg + 1]] += pars[:, seg][..., np.newaxis]
        else:
            mat[:, lims[seg] : lims[seg + 1]] = pars[:, seg][..., np.newaxis]


@nb.njit(parallel=True)
def __map2tod_binned_det_loop(
    pars: NDArray[np.floating],
    inds: NDArray[np.int_],
    mat: NDArray[np.floating],
    ndet: int,
    ndata: int,
):
    """
    Helper function to add data to a TOD using numba
    to parallelize across the det axis.

    Parameters
    ----------
    pars : NDArray[np.floating]
        Values to add to the TOD, should be (ndet, ?).
        Detector axis needs to be in the same order as mat.
        Mapping to the TOD gets done by inds.
    inds : NDArray[np.int_]
        Array of indices that maps each col of pars into the TOD.
        This mapping should be the same for each row/det.
        This is analogous to P in the mapmaker eq.
    mat : NDArray[np.floating]
        The TOD array, should be (ndet, n).
        Will be added to (not overwriten) in place.
    ndet : int
        The number of detectors.
    ndata : int
        The number of samples per detector.
    """
    for det in nb.prange(ndet):
        mat[det][0:ndata] += pars[det][inds[0:ndata]]


def map2tod_binned_det(
    mat: NDArray[np.floating],
    binned: NDArray[np.floating],
    vec: NDArray[np.floating],
    lims: Tuple[float, float],
    nbin: int,
    do_add: bool = True,
):
    """
    Unroll binned data into a TOD.

    Parameters
    ----------
    mat : NDArray[np.floating]
        The TOD array, should be (ndet, ndata).
    binned : NDArray[np.floating]
        The binned data, should be (ndet, nbin).
    vec : NDArray[np.floating]
        The vector that we are binned by, should be (ndet, ndata).
    lims : tuple[float, float]
        The bounds of the binning, should only have 2 elements.
    nbin : int
       The number of bins.
    do_add : bool, default: True
        If True then the TOD is added to mat.
        If False the contents of mat are overwriten.
    """
    ndet, ndata = mat.shape
    fac = nbin / (lims[1] - lims[0])
    inds = np.asarray((vec - lims[0]) * fac, dtype="int64")
    if do_add == False:
        mat[:] = 0
    __map2tod_binned_det_loop(binned, inds, mat, ndet, ndata)


# @nb.njit(parallel=True)
# def map2tod_binned_det(mat,pars,vec,lims,nbin,do_add=True):
#    n=mat.shape[1]
#    inds=np.empty(n,dtype='int')
#    fac=nbin/(lims[1]-lims[0])
#    for i in nb.prange(n):
#        inds[i]=(vec[i]-lims[0])*fac
#    ndet=mat.shape[0]
#    if do_add==False:
#        mat[:]=0
#    for det in np.arange(ndet):
#        for i in nb.prange(n):
#            mat[det][i]=mat[det][i]+pars[det][inds[i]]
