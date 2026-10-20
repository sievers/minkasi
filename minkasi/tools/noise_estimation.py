from math import factorial
from numba import prange

import numpy as np

def get_index(nmaps, idx):
    """
    Unrolls flat index for diff number into map idicies.
    
    Parameters
    ----------
    nmaps : int
        Number of maps
    idx : int
        Flat index
    Returns
    -------
    idx0 : int
        Index of first map
    idx1 : int
        Index of second map
    """
    idx0 = 0
    cur_idx = idx
    
    nmaps -= 1
    assert cur_idx < factorial(nmaps), "Index too large"
    
    while cur_idx >= nmaps-idx0:
        cur_idx -= (nmaps - idx0) #Each loop, the number of maps is reduced by one.
        idx0 += 1
    idx1 = cur_idx + idx0 + 1
    
    return idx0, idx1

def cov_from_maps(maps, ivar):
    """
    Returns an estimate of the mapspace noise covariance from a set of maps.

    Parameters
    ----------
    maps : list[NDArray]
        List of maps from which to compute mapspace noise cov
    ivar : NDArray
        ivar map for the coadded set of maps
    Returns
    -------
    cov : NDArray
        Full pixel-pixel mapspace noise cov
    """

    assert len(maps) != 1, "More than one map required"

    ndiffs = factorial(len(maps)-1)
    shape_diffs = list(maps[0].shape)
    shape_diffs[:0] = [ndiffs]

    diffs = np.empty(shape_diffs)

    for i in prange(ndiffs): #Note this isn't super efficient since i=0 is doing a lot more steps than i=ndiffs-1
        idx0, idx1 = get_index(len(maps), i)

        diffs[i] = maps[idx0] - maps[idx1]



