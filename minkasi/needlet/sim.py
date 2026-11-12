"""
Simulations for the needlet mapmaker: a known sky injected into real MUSTANG-2 TODs.
"""

import numpy as np
from numpy.typing import NDArray

from ..mapmaking.map2tod import map2tod
from ..parallel import myrank, nproc
from ..tods import TodVec
from ..tods.processing import fit_cm_plus_poly
from .needlet import WavSkyMap


def inject_sim(
    todvec: TodVec, wmap: WavSkyMap, sky: NDArray[np.floating], noise: str, seed: int
) -> None:
    """
    Replace the TOD data by a simulation, following WITCH: the noise is either white with each detector's level
    estimated from the data ("white") or the data with the sign flipped on every other TOD ("flip"), and the sky is
    added on top. Then the polynomial is subtracted as for real data (fit_cm_plus_poly).

    Parameters
    ----------
    todvec : TodVec
        TODs read with read_tods(..., raw=True).
    wmap : WavSkyMap
        Map giving the pixelization.
    sky : NDArray[np.floating]
        Sky in the units of the TODs on the pixels of wmap.real_map.
    noise : str
        "white" or "flip".
    seed : int
        Random seed (white noise).
    """
    for i, tod in enumerate(todvec.tods):
        dat = tod.info["dat_calib"]
        k = myrank + nproc * i  # global TOD index, as in WITCH
        if noise == "white":
            lo, hi = np.percentile(np.diff(dat, axis=-1), [33.0, 68.0], axis=-1)
            scale = (hi - lo) / np.sqrt(8)
            dat = (
                np.random.default_rng([seed, k]).normal(size=dat.shape) * scale[:, None]
            )
        else:
            dat = dat * (-1) ** (k % 2)
        sig = np.zeros_like(dat)
        map2tod(sig, sky, wmap.real_map.get_pix(tod), do_add=True)
        tod.info["dat_calib"] = fit_cm_plus_poly(np.ascontiguousarray(dat + sig))
