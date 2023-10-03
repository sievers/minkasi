import sys
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core import Tod

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


def slice_with_copy(arr: NDArray, ind: NDArray[np.bool_], axis: int = 0) -> NDArray:
    """
    Make a copy of any array with a mask applied on a given index.

    Parameters
    ----------
    arr : NDArray
        Array to copy and mask.
    ind : NDArray[np.bool_]
        Boolean mask to apply along the provided axis.
    axis : int, default: 0
        Axis to mask along.

    Returns
    -------
    ans : NDArray
        Copy with mask applied.

    Raises
    ------
    ValueError
        If arr isn't a numpy array.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input arr should be an array")
    indices = np.nonzero(ind)[0]
    ans = np.take(arr, indices, axis).copy()
    return ans


def mask_dict(mydict: dict, mask: NDArray[np.bool_]) -> dict:
    """
    Mask all items in a dictionary in place.
    nd arrays are split along the first axis that has the same length as vec.
    Things that aren't slicible are left alone.

    Parameters
    ----------
    mydict : dict
        Dict whose items will be masked.
    mask : NDArray[np.bool_]
        Mask to apply.

    Returns
    -------
    masked_dict : dict
        The dictionary with the mask applied.
        Note that this function also works in place.
    """
    for key in mydict.keys():
        tmp = mydict[key]
        if isinstance(tmp, np.ndarray):
            axes = np.where(np.array(tmp.shape) == len(mask))[0]
            if len(axes):
                mydict[key] = slice_with_copy(tmp, mask, axes[0])
    return mydict


def split_dict(mydict: dict, vec: NDArray, thresh: float) -> List[dict]:
    """
    Split a dictionary into sub-dictionaries wherever a gap in vec is larger than thresh.
    Useful for e.g. splitting TODs where there's a large time gap due to cuts.
    nd arrays are split along the first axis that has the same length as vec.
    Things that aren't slicible are copied directly.

    Parameters
    ----------
    mydict : dict
        Dict whose items will be split up.
    vec : NDArray
        Vector to get splits from.
    thresh : float
        Minimum difference between samples of vec that cause a split.

    Returns
    -------
    split_dicts : list[dict]
        List of split up dictionaries.
        If no splits occured this will just be [mydict]
    """
    inds = np.where(np.diff(vec) > thresh)[0]
    if len(inds) == 0:
        return [mydict]
    ndict = len(inds) + 1
    inds = np.hstack([[0], inds + 1, [len(vec)]])
    masks = [np.zeros_like(vec, dtype=bool)] * ndict
    for i in range(ndict):
        masks[i][inds[i] : inds[i + 1]] = True

    out = [mask_dict(mydict.copy(), mask) for mask in masks]

    return out


class detOffset:
    """
    Class to store detector offsets for a TOD.

    Attributes
    ----------
    sz : int
        The number of detectors.
    params : NDArray[np.floating]
        The detector offsets.
    fname : str
        The TOD filename.
    """

    def __init__(self, tod: Optional["Tod"] = None):
        """
        Initialize the detector offsets.
        params will be all zeros.

        Parameters
        ----------
        tod : Tod | None, default: None
            The TOD to initialize from.
            If None then the detOffset will only have 1 detector.
        """
        self.sz: int
        self.params: NDArray[np.floating]
        self.fname: str
        if tod is None:
            self.sz = 1
            self.params = np.zeros(1)
            self.fname = ""
        else:
            self.sz = tod.get_ndet()
            self.params = np.zeros(self.sz)
            self.fname = tod.info["fname"]

    def copy(self) -> Self:
        """
        Make a copy of this instance.
        """
        cp = detOffset()
        cp.sz = self.sz
        cp.params = self.params.copy()
        cp.fname = self.fname
        return cp

    def clear(self):
        """
        Zero out params.
        """
        self.params[:] = 0

    def dot(self, other: Optional[Self] = None) -> float:
        """
        Take the dot product of this detOffset with another.

        Parameters
        ----------
        other : detOffset | None
            The detOffset to dot with.
            If None, this instance will be dotted with itself.
        """
        if other is None:
            return np.dot(self.params, self.params)
        else:
            return np.dot(self.params, other.params)

    def axpy(self, common: Self, a: float):
        """
        Apply an ax + y operation to this detOffset.
        self.params will be modified in place.

        Parameters
        ----------
        common : detOffset
            The x in the axpy operation.
        a : float
            The value to multipy common by.
        """
        self.params = self.params + a * common.params

    def tod2map(
        self,
        tod: "Tod",
        dat: Optional[NDArray[np.floating]] = None,
        do_add: bool = True,
        do_omp: bool = False,
    ):
        """
        Project a TOD into params.

        Parameters
        ----------
        tod : Tod
            TOD to use, tod.info['dat_calib] is used if dat is None.
        dat : NDArray[np.array] | None, default: None
            The TOD data to use.
            If None, tod.info['dat_calib'] is used.
        do_add : bool, default: True
            If True add to the current params value.
        do_omp : bool, default: False
            Doesn't do anything.
            Presumably just here to keep consistant function signatures.
        """
        if dat is None:
            dat = tod.get_data()
        if do_add == False:
            self.clear()
        self.params[:] = self.params[:] + np.sum(dat, axis=1)

    def map2tod(
        self,
        tod: "Tod",
        dat: Optional[NDArray[np.floating]] = None,
        do_add: bool = True,
        do_omp: bool = False,
    ):
        """
        Project params into a TOD.

        Parameters
        ----------
        tod : Tod
            TOD to use, tod.info['dat_calib] is used if dat is None.
        dat : NDArray[np.array] | None, default: None
            The TOD data to use.
            If None, tod.info['dat_calib'] is used.
        do_add : bool, default: True
            If True add to the current params value.
        do_omp : bool, default: False
            Doesn't do anything.
            Presumably just here to keep consistant function signatures.
        """
        if dat is None:
            dat = tod.get_data()
        if do_add == False:
            dat[:] = 0
        dat[:] = dat[:] + np.repeat([self.params], dat.shape[1], axis=0).transpose()

    def write(self, fname: Optional[str] = None):
        """
        Doesn't currently do anything.
        Presumably will write the det offsets to disk when implemented.
        """
        pass

    def __mul__(self, to_mul: Self) -> Self:
        """
        Multiply this detOffset with another.

        Parameters
        ----------
        to_mul : detOffset
            The detOffset to multiply by.

        Returns
        -------
        multiplied : detOffset
            The result of the multiplication.
        """
        multiplied = self.copy()
        multiplied.params = self.params * to_mul.params
        return multiplied
