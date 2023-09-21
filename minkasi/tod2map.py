import numpy as np
from numpy.typing import NDArray
from .parallel import get_nthread, have_mpi
from .minkasi import (
    tod2map_simple_c,
    tod2map_everyone_c,
    tod2map_atomic_c,
    tod2map_omp_c,
    tod2map_cached_c,
    tod2map_qu_simple_c,
    tod2map_iqu_simple_c,
    tod2map_qu_precon_simple_c,
    tod2map_iqu_precon_simple_c,
)
from .tods import TodVec
from .maps import MapType, PolMap

try:
    import numba as nb
except ImportError:
    import no_numba as nb


def tod2map_simple(
    map: NDArray[np.floating], dat: NDArray[np.floating], ipix: NDArray[np.int32]
):
    """
    Bin a TOD into a map in place.
    This is for temperature map, for pol maps use polmap2tod.

    Parameters
    ----------
    map : NDArray[np.floating]
        The map to fill.
    dat : NDArray[np.floating]
        The TOD to bin.
        Should be ndet by ndata.
    ipix : NDArray[np.int32]
        The pixellization matrix.
    """
    ndet = dat.shape[0]
    ndata = dat.shape[1]
    if ipix.dtype != "int32":
        print(
            "Warning - ipix is not int32 in tod2map_simple.  this is likely to produce garbage results."
        )
    tod2map_simple_c(map.ctypes.data, dat.ctypes.data, ndet, ndata, ipix.ctypes.data)


def tod2map_everyone(
    map: NDArray[np.floating],
    dat: NDArray[np.floating],
    ipix: NDArray[np.int32],
    edges: NDArray[np.int32],
):
    """
    Bin a TOD into a map in place,
    all threads will loop over all data, but only assign a region they are responsible for.
    This is for temperature map, for pol maps use polmap2tod.

    Parameters
    ----------
    map : NDArray[np.floating]
        The map to fill.
    dat : NDArray[np.floating]
        The TOD to bin.
        Should be ndet by ndata.
    ipix : NDArray[np.int32]
        The pixellization matrix.
    """
    assert len(edges) == get_nthread() + 1
    if ipix.dtype != "int32":
        print(
            "Warning - ipix is not int32 in tod2map_everyone.  this is likely to produce garbage results."
        )
    if edges.dtype != "int32":
        print(
            "Warning - edges is not int32 in tod2map_everyone.  this is likely to produce garbage results."
        )
    tod2map_everyone_c(
        map.ctypes.data,
        dat.ctypes.data,
        dat.shape[0],
        dat.shape[1],
        ipix.ctypes.data,
        map.size,
        edges.ctypes.data,
        len(edges),
    )


def tod2map_omp(
    map: NDArray[np.floating],
    dat: NDArray[np.floating],
    ipix: NDArray[np.int32],
    atomic: bool = False,
):
    """
    Bin a TOD into a map in place using OMP for parallelization.
    This is for temperature map, for pol maps use polmap2tod.

    Parameters
    ----------
    map : NDArray[np.floating]
        The map to fill.
    dat : NDArray[np.floating]
        The TOD to bin.
        Should be ndet by ndata.
    ipix : NDArray[np.int32]
        The pixellization matrix.
    atomic : bool, default: False
        If True use atomic operations to reduce overhead.
    """
    ndet = dat.shape[0]
    ndata = dat.shape[1]
    if ipix.dtype != "int32":
        print(
            "Warning - ipix is not int32 in tod2map_omp.  this is likely to produce garbage results."
        )
    if atomic:
        tod2map_atomic_c(
            map.ctypes.data, dat.ctypes.data, ndet, ndata, ipix.ctypes.data, map.size
        )
    else:
        tod2map_omp_c(
            map.ctypes.data, dat.ctypes.data, ndet, ndata, ipix.ctypes.data, map.size
        )


def tod2map_cached(
    map: NDArray[np.floating], dat: NDArray[np.floating], ipix: NDArray[np.int32]
):
    """
    Bin a TOD into a map in place using OMP for parallelization
    but with each thread caching a sticky copy of the map.
    This is for temperature map, for pol maps use polmap2tod.

    Parameters
    ----------
    map : NDArray[np.floating]
        The map to fill.
    dat : NDArray[np.floating]
        The TOD to bin.
        Should be ndet by ndata.
    ipix : NDArray[np.int32]
        The pixellization matrix.
    """
    ndet = dat.shape[0]
    ndata = dat.shape[1]
    if ipix.dtype != "int32":
        print(
            "Warning - ipix is not int32 in tod2map_cached.  this is likely to produce garbage results."
        )
    tod2map_cached_c(
        map.ctypes.data, dat.ctypes.data, ndet, ndata, ipix.ctypes.data, map.shape[1]
    )


def tod2polmap(
    map: NDArray[np.floating],
    dat: NDArray[np.floating],
    poltag: str,
    twogamma: NDArray[np.floating],
    ipix: NDArray[np.int32],
):
    """
    Unroll a polmap into a a TOD inplace.
    This is for temperature map, for pol maps use polmap2tod.

    Parameters
    ----------
    map : NDArray[np.floating]
        The map to fill.
    dat : NDArray[np.floating]
        The TOD to bin.
        Should be ndet by ndata.
    poltag : str
        The polarization tag.
        Valid options are:
            * QU
            * IQU
            * QU_PRECON
            * IQU_PRECON
    twogamma : NDArray[np.floating]
        Array of twogamma terms.
        This sets the weight between Q and U.
    ipix : NDArray[np.int32]
        The pixellization matrix.
    """
    ndet = dat.shape[0]
    ndata = dat.shape[1]
    if poltag == "QU":
        fun = tod2map_qu_simple_c
    elif poltag == "IQU":
        fun = tod2map_iqu_simple_c
    elif poltag == "QU_PRECON":
        fun = tod2map_qu_precon_simple_c
    elif poltag == "IQU_PRECON":
        fun = tod2map_iqu_precon_simple_c
    else:
        raise ValueError("unrecognized poltag " + repr(poltag) + " in tod2polmap.")
    fun(
        map.ctypes.data,
        dat.ctypes.data,
        twogamma.ctypes.data,
        ndet,
        ndata,
        ipix.ctypes.data,
    )


@nb.jit(nopython=True)
def tod2mapbowl(vecs: NDArray[np.floating], mat: NDArray[np.floating]):
    """
    Convert bowling TOD to bowling parameters for tsBowl.

    Parameters
    ----------
    vecs: NDArray[np.floating]
        The pseudo-Vandermonde matrix.
        Should have shape (order, ndata, ndet)
    bowl_tod : NDArray[np.floating]
        The TOD of the bowl.


    params: NDArray[np.floating]
        Corresponding weights for pseudo-Vandermonde matrix.
        Should have shape (order, ndet)
    """
    # Return tod should have shape ndet x ndata
    params = np.zeros((vecs.shape[0], vecs.shape[-1]))
    for i in range(vecs.shape[0]):
        params[i] = np.dot(vecs[i, ...].T, mat[i, ...])


@nb.njit(parallel=True)
def tod2map_destriped(mat, pars, lims, do_add=True):
    """
    Bin a TOD into destriper params.

    Parameters
    ----------
    mat : NDArray[np.floating]
        TOD data array, should be (ndet, ndata).
    pars : NDArray[np.floating]
        Array to fill with the stripe parameters, should be (ndet, nseg).
    lims : NDArray[np.int_]
        The edges of the stripe segments, should be (nseg+1,).
    do_add : bool, default: True
        If True then the params are added to pars.
        If False the contents of pars are overwriten.
    """
    if not do_add:
        pars[:] = 0
    nseg = len(lims) - 1
    for seg in nb.prange(nseg):
        pars[:, seg] += np.sum(mat[:, lims[seg] : lims[seg + 1]], axis=1)[
            ..., np.newaxis
        ]


@nb.njit(parallel=True)
def __tod2map_binned_det_loop(
    pars: NDArray[np.floating],
    inds: NDArray[np.int_],
    mat: NDArray[np.floating],
    ndet: int,
    ndata: int,
):
    """
    Helper function to add binned data to a map using numba
    to parallelize across the det axis.

    Parameters
    ----------
    pars : NDArray[np.floating]
        Parameters to add the TOD into.
        Detector axis needs to be in the same order as mat.
        Mapping to the TOD gets done by inds.
        Will be added to in place.
    inds : NDArray[np.int_]
        Array of indices that maps each col of pars into the TOD.
        This mapping should be the same for each row/det.
        This is analogous to P in the mapmaker eq.
    mat : NDArray[np.floating]
        The TOD array, should be (ndet, ndata).
    ndet : int
        The number of detectors.
    ndata : int
        The number of samples per detector.
    """
    for det in nb.prange(ndet):
        pars[det][inds[0:ndata]] += mat[det][0:ndata]


def tod2map_binned_det(mat, binned, vec, lims, nbin, do_add=True):
    """
    Bin a TOD into a map with additional sample binning.

    Parameters
    ----------
    mat : NDArray[np.floating]
        The TOD array, should be (ndet, ndata).
    binned : NDArray[np.floating]
        The binned data, should be (ndet, nbin).
        Will be filled in place
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
        binned[:] = 0
    __tod2map_binned_det_loop(binned, inds, mat, ndet, ndata)


def make_hits(todvec: TodVec, map: MapType, do_weights: bool = False) -> MapType:
    """
    Make the hits map (or a weights map).

    Parameters
    ----------
    todvec : TodVec
        The TODs to make hits for.
    map : MapType
        Map to intialize hits map from.
        Really only used to copy metadata from.
    do_weights : bool, default: False
        If True then detector weights are applied to the hits map.
        In this case the TODs in todvec should have a noise model.

    Returns
    -------
    hits : MapType
        The hits map. Will be the same type as map.
    """
    hits = map.copy()
    if isinstance(hits, PolMap) and isinstance(map, PolMap):
        if map.npol > 1:
            hits.set_polstate(map.poltag + "_PRECON")
    hits.clear()

    for tod in todvec.tods:
        if do_weights:
            weights = tod.get_det_weights()
            if weights is None:
                print(
                    "error in making weight map.  Detector weights requested, but do not appear to be present.  Do you have a noise model?"
                )
                tmp = np.ones(tod.get_data_dims())
            else:
                sz = tod.get_data_dims()
                tmp = np.outer(weights, np.ones(sz[1]))
        else:
            tmp = np.ones(tod.get_data_dims())

        if "mask" in tod.info:
            tmp = tmp * tod.info["mask"]

        hits.tod2map(tod, tmp)

    if have_mpi:
        print("reducing hits")
        tot = hits.map.sum()
        print("starting with total hitcount " + repr(tot))
        hits.mpi_reduce()
        tot = hits.map.sum()
        print("ending with total hitcount " + repr(tot))

    return hits
