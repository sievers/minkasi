import sys
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..lib.minkasi import cuts2tod_c, tod2cuts_c

if TYPE_CHECKING:
    from ..maps import MapType
    from . import Tod

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


def segs_from_vec(
    vec: NDArray[np.bool_], pad: bool = True
) -> Tuple[int, List[int], List[int]]:
    """
    Return the starting/stopping points of regions marked False in vec.  For use in e.g. generating
    cuts from a vector/array.

    Parameters
    ----------
    vec : NDArray[np.bool_]
        Boolean array to make segments from.
    pad : bool, default: True
        Pad vector with a True on each end.
        If pad is False, assume vector is already True-padded.

    Returns
    -------
    nseg : int
        The number of segments.
    istart : list[int]
        The start indices of the segments.
    istop : list[int]
        The stop indices of the segments.
    """
    # insert input vector into a True-padded vector do make reasoning about starting/stopping points
    # of False regions easier.
    if pad:
        vv = np.ones(len(vec) + 2, dtype="bool")
        vv[1:-1] = vec
    else:
        vv = vec.astype(bool)
    if vv.all():
        return 0, [], []
    inds = np.where(np.diff(vv))[0]
    assert len(inds) % 2 == 0
    nseg = len(inds) // 2
    istart = list(inds[::2])
    istop = list(inds[1::2])
    return nseg, istart, istop


class Cuts:
    """
    Class to store cuts information.

    Attributes
    ----------
    map : NDArray[np.floating]
        1D array storing cuts "map".
    inds : NDArray[np.integer]
        Indices with bad samples.
    nsamp : int
        Number of bad indices.
    do_add : bool
        If True then map2tod adds to the TOD.
    """

    def __init__(self, tod: Union["Tod", Self], do_add: bool = True):
        """
        Initialize the Cuts class.

        Parameters
        ----------
        tod : Tod | Cuts
            The TOD to make cuts from, should contain tod.info["bad_samples"].
            Alternatively pass in a Cuts object to initialize a copy.
        do_add : bool, default: True
            If True then map2tod adds to the TOD.

        Raises
        ------
        ValueError
            If tod is an invalid type.
        """
        self.map: NDArray[np.floating]
        self.inds: NDArray[np.integer]
        self.nsamp: int
        self.do_add: bool
        if isinstance(tod, Cuts):
            self.map = tod.map.copy()
            self.inds = tod.inds.copy()
            self.nsamp = tod.nsamp
            self.do_add = tod.do_add
            return
        try:
            bad_inds = np.where(tod.info["bad_samples"])
            dims = tod.get_data_dims()
        except AttributeError as exc:
            raise ValueError("tod must either be a Tod or Cuts object") from exc
        bad_inds = np.ravel_multi_index(bad_inds, dims)
        self.nsamp = len(bad_inds)
        self.inds = bad_inds
        self.map = np.zeros(self.nsamp)
        self.do_add = do_add

    def clear(self):
        """
        Zero out the map.
        """
        self.map[:] = 0

    def axpy(self, cuts: Self, a: float):
        """
        Apply an a*x + y operation to the map.
        self.map is y.

        Parameters
        ----------
        cuts : Cuts
            Cuts object whose map is x.
        a : float
            a from equation above.
        """
        self.map[:] = self.map[:] + a * cuts.map[:]

    def map2tod(self, tod: "Tod", dat: Optional[NDArray[np.floating]] = None):
        """
        Put the cuts map into a TOD.

        Parameters
        ----------
        tod : Tod
            The Tod to use.
            Data is loaded from this unless dat is not None.
        dat : NDArray[np.floating] | None
            It not None this array will be used for the TOD.
        """
        if dat is None:
            dat = tod.get_data()
        dd = np.ravel(dat)
        if self.do_add:
            dd[self.inds] = self.map
        else:
            dd[self.inds] += self.map
        dat = dd.reshape(dat.shape)

    def tod2map(self, tod: "Tod", dat: Optional[NDArray[np.floating]] = None):
        """
        Fill the cuts map from a TOD.

        Parameters
        ----------
        tod : Tod
            The Tod to use.
            Data is loaded from this unless dat is not None.
        dat : NDArray[np.floating] | None
            It not None this array will be used for the TOD.
        """
        if dat is None:
            dat = tod.get_data()
        dd = np.ravel(dat)
        self.map[:] = dd[self.inds]

    def dot(self, cuts: Self) -> float:
        """
        Take the dot product of two cuts maps.

        Parameters
        ----------
        cuts : Cuts
            The Cuts object to dot with this one.

        Returns
        -------
        tot : float
            The dot product.
        """
        tot = np.dot(self.map, cuts.map)
        return tot

    def copy(self) -> Self:
        """
        Make a copy of this Cuts object.

        Returns
        -------
        copy : Cuts
            The copied Cuts object.
        """
        return Cuts(self)


class CutsCompact:
    """
    Class for storing Cuts infromation in a compact way.

    Attributes
    ----------
    ndet : int
        The number of detector.
    nseg : NDArray[np.integer]
        The number of cut segments for each detector.
    imax : int
        The number of samples per detector.
    istart : list[list[int]]
        List of ndet lists, each one with the start indices of cuts.
    istop : list[list[int]]
        List of ndet lists, each one with the stop indices of cuts.
    map : NDArray[np.floating]
        1D array storing cuts "map".
    imap : NDArray[np.int64]
        1D array storing cuts "map" in index space.
    """

    def __init__(self, tod):
        self.ndet: int
        self.nseg: NDArray[np.integer]
        self.imax: int
        self.istart: List[List[int]]
        self.istop: List[List[int]]
        self.map: NDArray[np.floating]
        self.imap: NDArray[np.int64]
        if isinstance(tod, CutsCompact):
            self.ndet = tod.ndet
            self.nseg = tod.nseg
            self.imax = tod.imax
            self.istart = tod.istart
            self.istop = tod.istop
        else:
            ndet = tod.get_ndet()
            self.ndet = ndet
            self.nseg = np.zeros(ndet, dtype="int")
            self.imax = tod.get_ndata()
            self.istart = [[]] * ndet
            self.istop = [[]] * ndet

        self.imap = np.zeros(0, dtype="int64")
        self.map = np.zeros(0)

    def copy(self, deep: bool = True) -> Self:
        """
        Make a copy of this CutsCompact.

        Parameters
        ----------
        deep : bool
            If True then map and imap in the copy will be copies not referances.

        Returns
        -------
        copy : CutsCompact
            The copied object.
        """
        copy = CutsCompact(self)
        if deep:
            copy.imap = self.imap.copy()
            copy.map = self.map.copy()
        else:
            copy.imap = self.imap
            copy.map = self.map
        return copy

    def add_cut(self, det: int, istart: int, istop: int):
        """
        Add a cut to this CutsCompact.

        Parameters
        ----------
        det : int
            The detector to add a cut to.
        istart : int
            The index to start the cut at.
            If this is past the end of the data no cut is added.
        istop : int
            The index to stop the cut at.
            If this is past the end of the data the cut ends with the data.
        """
        if istart >= self.imax:
            # this is asking to add a cut past the end of the data.
            return
        if istop > self.imax:  # don't have a cut run past the end of the timestream
            istop = self.imax

        self.nseg[det] = self.nseg[det] + 1
        self.istart[det].append(istart)
        self.istop[det].append(istop)

    def cuts_from_array(self, cutmat: NDArray[np.bool_]):
        """
        Get cuts from a boolean mask.

        Parameters
        ----------
        cutmat : NDArray[np.bool_]
            (ndet, imax) boolean array.
            False in regions to cut.
        """
        for det in range(cutmat.shape[0]):
            nseg, istart, istop = segs_from_vec(cutmat[det, :])
            self.nseg[det] = nseg
            self.istart[det] = istart
            self.istop[det] = istop

    def merge_cuts(self):
        """
        Merge overlapping cuts.
        """
        tmp = np.ones(self.imax + 2, dtype="bool")
        for det in range(self.ndet):
            if (
                self.nseg[det] < 2
            ):  # if we only have one segment, don't have to worry about strange overlaps
                continue
            tmp[:] = True
            for i in range(self.nseg[det]):
                tmp[(self.istart[det][i] + 1) : (self.istop[det][i] + 1)] = False
            nseg, istart, istop = segs_from_vec(tmp, pad=False)
            self.nseg[det] = nseg
            self.istart[det] = istart
            self.istop[det] = istop

    def get_imap(self):
        """
        Make imap for the current cuts.
        Also initializes an empty map.
        """
        ncut = 0
        for det in range(self.ndet):
            for i in range(self.nseg[det]):
                ncut = ncut + (self.istop[det][i] - self.istart[det][i])
        print("ncut is " + repr(ncut))
        self.imap = np.zeros(ncut, dtype="int64")
        icur = 0
        for det in range(self.ndet):
            for i in range(self.nseg[det]):
                istart = det * self.imax + self.istart[det][i]
                istop = det * self.imax + self.istop[det][i]
                nn = istop - istart
                self.imap[icur : icur + nn] = np.arange(istart, istop)
                icur = icur + nn
        self.map = np.zeros(len(self.imap))

    def tod2map(
        self,
        tod: Optional["Tod"],
        mat: Optional[NDArray[np.floating]] = None,
        do_add: bool = True,
        do_omp: bool = False,
    ):
        """
        Fill the cuts map from a TOD.

        Parameters
        ----------
        tod : Tod | None
            The Tod to use.
            Data is loaded from this unless mat is not None.
        mat : NDArray[np.floating] | None
            It not None this array will be used for the TOD.
        do_add : bool, default: True
            If True then the map is added to.
        do_omp : bool, default: False
            Unused variable. Presumably for compatibility.
        """
        if mat is None:
            if tod is None:
                raise ValueError("Both tod and mat cannot be none")
            mat = tod.get_data()
        tod2cuts_c(
            self.map.ctypes.data,
            mat.ctypes.data,
            self.imap.ctypes.data,
            len(self.imap),
            do_add,
        )

    def map2tod(
        self,
        tod: Optional["Tod"],
        mat: Optional[NDArray[np.floating]] = None,
        do_add: bool = True,
        do_omp: bool = False,
    ):
        """
        Fill a TOD from the cuts map.

        Parameters
        ----------
        tod : Tod | None
            The Tod to use.
            Data is loaded from this unless mat is not None.
        mat : NDArray[np.floating] | None
            It not None this array will be used for the TOD.
        do_add : bool, default: True
            If True then the TOD is added to.
        do_omp : bool, default: False
            Unused variable. Presumably for compatibility.
        """
        if mat is None:
            if tod is None:
                raise ValueError("Both tod and mat cannot be none")
            mat = tod.get_data()
        cuts2tod_c(
            mat.ctypes.data,
            self.map.ctypes.data,
            self.imap.ctypes.data,
            len(self.imap),
            do_add,
        )

    def clear(self):
        """
        Zero out self.map.
        Doesn't clear imap.
        """
        self.map[:] = 0

    def dot(self, other: Optional[Self] = None) -> float:
        """
        Take the dot product of CutsCompact.

        Parameters
        ----------
        other : CutsCompact, default: None
            The CutsCompact to dot with.
            If None, this instance will dot with itself.

        Returns
        -------
        tot : float
            The dot product of self.map and other.map.
        """
        if other is None:
            other = self
        return np.dot(self.map, other.map)

    def axpy(self, common: Self, a: float):
        """
        Apply an a*x + y operation to the map.
        self.map is y.

        Parameters
        ----------
        common : CutsCompact
            Cuts object whose map is x.
        a : float
            a from equation above.
        """
        self.map = self.map + a * common.map

    def apply_prior(self, x: "MapType", Ax: "MapType"):
        """
        Apply prior using this CutsCompact.
        Prior is applied as Ax = Ax + self*x.
        If self.map is None nothing is done.

        Parameters
        ----------
        x : MapType
            The map to multiply by to make a prior
        Ax : MapType
            Map to add prior to.
        """
        if self.map is None:
            return
        Ax.map = Ax.map + self.map * x.map

    def write(self, fname=None):
        """
        Not currently implemented.
        Will presumably write out the CutsCompact to disk.
        """
        pass

    def __mul__(self, to_mul: Self) -> Self:
        """
        Multiply this CutsCompact with another.

        Parameters
        ----------
        to_mul : CutsCompact
            The CutsCompact containing the map to multiply by.
        """
        tt = self.copy()
        tt.map = self.map * to_mul.map
        return tt


# Define a type for typehints
CutsType = Union[Cuts, CutsCompact]
