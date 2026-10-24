"""
CMB as a noise source for maps that contain it, e.g. ACT, when the signal of interest is not the CMB.
"""

import os

import numpy as np
from numpy.typing import NDArray

from ..parallel import comm, have_mpi, myrank


def lensed_cmb_cl(
    lmax: int = 10000, cache: str | None = None
) -> NDArray[np.floating]:
    """
    Lensed LCDM TT spectrum from CAMB with Planck 2018 parameters.

    With MPI only rank 0 runs CAMB (or reads the cache) and broadcasts the result.

    Parameters
    ----------
    lmax : int, default: 10000
        Maximum multipole.
    cache : str | None, default: None
        .npy file to read the spectrum from if it exists and to save it to otherwise.

    Returns
    -------
    cl : NDArray[np.floating]
        C_l in uK_CMB^2 sr, indexed by l; zero for l < 2.
    """
    cl = None
    if myrank == 0:
        if cache and os.path.exists(cache):
            cl = np.load(cache)
        else:
            import camb

            pars = camb.set_params(
                H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544, As=2.1e-9, ns=0.9649,
                lmax=lmax, lens_potential_accuracy=1,
            )  # fmt: skip
            dl = camb.get_results(pars).get_lensed_scalar_cls(
                lmax=lmax, CMB_unit="muK", raw_cl=False
            )[:, 0]  # D_l = l (l + 1) C_l / 2 pi
            ell = np.arange(len(dl))
            cl = np.zeros(len(dl))
            cl[2:] = dl[2:] * 2 * np.pi / (ell[2:] * (ell[2:] + 1))
            if cache:
                np.save(cache, cl)
    return comm.bcast(cl) if have_mpi else cl


def cmb_fourier_weight(
    shape: tuple[int, int],
    pixel: float,
    cl: NDArray[np.floating],
    beam_fwhm: float,
    ivar: float,
    area: float,
    scale: float = 1.0,
) -> NDArray[np.floating]:
    """
    Weight N_l / (N_l + scale * C_l B_l^2) of each Fourier mode of a map with white noise plus the
    (beam smoothed) CMB: the factor by which the precision of the white noise model is reduced
    when the CMB is counted as noise too. It tends to 1 where the CMB is negligible.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape (nx, ny) of the map.
    pixel : float
        Pixel size of the map grid, radians.
    cl : NDArray[np.floating]
        CMB C_l, indexed by l, in the (squared) units of the map times sr; zero beyond its end.
    beam_fwhm : float
        FWHM of the beam the CMB is smoothed with in the map, radians.
    ivar : float
        Typical inverse variance of one independent pixel of the map.
    area : float
        Solid angle of that pixel, sr, so the white noise power is N_l = area / ivar.
    scale : float, default: 1.0
        Factor on the CMB power, 0 gives no change of the white noise model.

    Returns
    -------
    weight : NDArray[np.floating]
        Multiplier of shape (nx, ny // 2 + 1), in the layout of np.fft.rfft2.
    """
    f2 = (
        np.fft.fftfreq(shape[0])[:, None] ** 2 + np.fft.rfftfreq(shape[1])[None, :] ** 2
    )
    ell = 2 * np.pi * np.sqrt(f2) / pixel
    sigma = beam_fwhm / np.sqrt(8 * np.log(2))
    cmb = np.interp(ell, np.arange(len(cl)), cl, right=0.0) * np.exp(
        -((ell * sigma) ** 2)
    )
    return 1 / (1 + scale * ivar * cmb / area)
