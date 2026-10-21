import itertools

import numpy as np


def cov_from_maps(maps):
    """
    Estimate the map space noise covariance from a set of independent splits of the same sky.
    The signal cancels in the difference of two splits, so the covariance of each difference
    (divided by 2, as two noise realizations are summed) estimates the noise covariance.
    With n splits the estimate has rank at most n-1, so it is only useful with regularization
    or restricted to a band (e.g. in the wavelet basis).

    Parameters
    ----------
    maps : Sequence[NDArray[np.floating]]
        Split maps, all with the same shape.

    Returns
    -------
    cov : NDArray[np.floating]
        Full pixel-pixel noise covariance of the flattened maps, shape (npix, npix).
    """
    assert len(maps) > 1, "More than one map required"
    maps = np.array([m.ravel() for m in maps])
    diffs = np.array(
        [maps[i] - maps[j] for i, j in itertools.combinations(range(len(maps)), 2)]
    )
    return diffs.T @ diffs / (2 * len(diffs))
