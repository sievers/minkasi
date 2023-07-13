import numpy as np
from numpy.typing import NDArray


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


def split_dict(mydict: dict, vec: NDArray, thresh: float) -> list[dict]:
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
