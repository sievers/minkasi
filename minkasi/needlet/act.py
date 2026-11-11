"""
ACT maps in the needlet mapmaker: reading them onto the fine (M2) grid, units, simulation, and the ACT term of the
operator. ACT pixels are modelled as ACT_OSAMP x ACT_OSAMP blocks of the fine grid.
"""

from collections.abc import Callable, Sequence

import astropy.constants as const
import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from pixell import enmap, utils

from .needlet import (
    WavSkyMap,
    apply_beam,
    block_average,
    gaussian_beam_ft,
    map2wav_real,
    wav2map_real,
)

ACT_OSAMP = 10  # fine pixels per ACT pixel


def act_conversion(
    freq: float = 90e9, beam_fwhm: float = 9.0, unit: str = "K_RJ"
) -> float:
    """
    Factor converting ACT maps (uK CMB) to the units of the M2 data.

    Parameters
    ----------
    freq : float, default: 90e9
        Observing frequency in Hz.
    beam_fwhm : float, default: 9.0
        FWHM of the M2 beam in arcsec, only used for the flux units.
    unit : str, default: "K_RJ"
        Unit of the M2 TODs: "K_RJ" (Rayleigh-Jeans temperature, the calibration of the MUSTANG-2 TODs, as in WITCH)
        or the flux units "Jy" and "mJy" (per M2 beam).

    Returns
    -------
    float
        M2 units per uK_CMB.
    """
    if unit == "K_RJ":  # dB/dT of the CMB blackbody relative to Rayleigh-Jeans
        x = (const.h * freq * u.Hz / (const.k_B * 2.7255 * u.K)).to_value(u.one)
        return 1e-6 * x**2 * np.exp(x) / np.expm1(x) ** 2
    omega = (2 * np.pi * (beam_fwhm * u.arcsec / (8 * np.log(2)) ** 0.5) ** 2).to(u.sr)
    equiv = u.thermodynamic_temperature(freq * u.Hz, 2.7255 * u.K)
    return ((1 * u.uK).to(u.Jy / u.sr, equiv) * omega).to_value(u.Unit(unit))


def read_act(
    paths: Sequence[str],
    lims: Sequence[float],
    wmap: WavSkyMap,
    fac: int = ACT_OSAMP,
    conv: float = 1.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Read ACT split maps and inverse variance maps in a box around the field and coadd them onto the (fine) wavelet
    map pixelization.

    The ACT term models ACT pixels as fac x fac blocks of the fine grid (block_average). Each block takes the ACT
    pixel at its centre's sky position, found through the two WCSs. (Before 2026-11-10 the ACT pixels were copied
    onto the blocks by array index, which mirrored ACT in RA: pixell returns the box with RA increasing along the
    array, the fine grid has RA decreasing.)

    Parameters
    ----------
    paths : Sequence[str]
        ACT ivar maps. The corresponding maps are found by replacing _ivar with _map.
    lims : Sequence[float]
        Field limits (radians).
    wmap : WavSkyMap
        Map whose real_map pixelization to use.
    fac : int, default: ACT_OSAMP
        Fine pixels per ACT pixel along each axis.
    conv : float, default: 1.0
        Factor converting ACT map units to the units of the M2 data. Maps are multiplied by it and ivar divided by
        its square.

    Returns
    -------
    ivar : NDArray[np.floating]
        Total inverse variance on the wmap pixelization, shape (nx, ny), constant on each block.
    wdata : NDArray[np.floating]
        Sum over splits of ivar * map on the wmap pixelization, shape (nx, ny), constant on each block.
    """
    # the field limits plus a margin of a few ACT pixels (the grid of the full map is kept, so the pixels picked do
    # not depend on the box)
    pad = 0.05 * utils.degree
    box = np.array(
        [
            [min(lims[2:]) - pad, min(lims[:2]) - pad],
            [max(lims[2:]) + pad, max(lims[:2]) + pad],
        ]
    )
    nx, ny = wmap.real_map.map.shape
    # block centres on the fine grid; map[i, j] is FITS pixel (i, j), i along RA
    bi = np.arange(-(-nx // fac)) * fac + (fac - 1) / 2
    bj = np.arange(-(-ny // fac)) * fac + (fac - 1) / 2
    ii, jj = np.meshgrid(bi, bj, indexing="ij")
    bra, bdec = np.deg2rad(wmap.real_map.wcs.wcs_pix2world(ii, jj, 0))
    ivar = np.zeros((nx, ny))
    wdata = np.zeros((nx, ny))
    for path in paths:
        ivar_map = enmap.read_map(path, box=box) / conv**2
        act_map = enmap.read_map(path.replace("_ivar", "_map"), box=box) * conv
        if act_map.ndim == 3:
            act_map = act_map[0]  # temperature
        while ivar_map.ndim > 2:  # (T, Q, U) or (3, 3) covariance diagonal: keep TT
            ivar_map = ivar_map[0]
        py, px = enmap.sky2pix(ivar_map.shape, ivar_map.wcs, [bdec, bra])
        pick = [np.round(py).astype(int), np.round(px).astype(int)]  # nearest ACT pixel
        inside = (
            (pick[0] >= 0)
            & (pick[0] < ivar_map.shape[-2])
            & (pick[1] >= 0)
            & (pick[1] < ivar_map.shape[-1])
        )
        pick = [np.clip(p, 0, n - 1) for p, n in zip(pick, ivar_map.shape[-2:])]
        iv = np.where(inside, np.asarray(ivar_map)[pick[0], pick[1]], 0.0)
        ivar += np.repeat(np.repeat(iv, fac, 0), fac, 1)[:nx, :ny]
        wdata += np.repeat(
            np.repeat(iv * np.asarray(act_map)[pick[0], pick[1]], fac, 0), fac, 1
        )[:nx, :ny]
    return ivar, wdata


def sim_act(
    sky: NDArray[np.floating],
    ivar: NDArray[np.floating],
    beam_fwhm_pix: float,
    seed: int,
) -> NDArray[np.floating]:
    """
    Simulated ACT map on the fine pixels: the sky smoothed with the ACT beam and averaged over ACT pixels, plus white
    noise of inverse variance ivar per ACT pixel (constant over each block of fine pixels).

    Parameters
    ----------
    sky : NDArray[np.floating]
        Sky in the M2 data units on the fine pixels.
    ivar : NDArray[np.floating]
        ACT inverse variance per ACT pixel repeated on its fine pixels, in the M2 data units.
    beam_fwhm_pix : float
        ACT beam FWHM in fine pixels.
    seed : int
        Random seed.

    Returns
    -------
    act : NDArray[np.floating]
        The simulated ACT map on the fine pixels.
    """
    signal = block_average(
        apply_beam(sky, gaussian_beam_ft(sky.shape, beam_fwhm_pix)), ACT_OSAMP
    )
    white = np.random.default_rng([seed, 1]).normal(size=sky.shape)
    unit = (
        block_average(white, ACT_OSAMP) * ACT_OSAMP
    )  # unit variance, constant per ACT pixel
    return signal + np.divide(
        unit, np.sqrt(ivar), out=np.zeros_like(unit), where=ivar > 0
    )


def edge_taper(shape: tuple[int, int], width: int) -> NDArray[np.floating]:
    """
    Window that is 1 in the interior and falls to 0 at the edges with a cosine ramp.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the map.
    width : int
        Ramp width in pixels. 0 gives no taper.

    Returns
    -------
    NDArray[np.floating]
        2D window of the given shape.
    """
    if width <= 0:
        return np.ones(shape)

    def ramp(n):
        r = np.ones(n)
        edge = 0.5 * (1 - np.cos(np.pi * (np.arange(width) + 0.5) / width))
        r[:width], r[-width:] = edge, edge[::-1]
        return r

    return ramp(shape[0])[:, None] * ramp(shape[1])[None, :]


class ActTerm:
    """
    The ACT term of the needlet operator, F_i B^T s W s B (sum_j F_j x_j) on the ACT bands, with B the ACT beam and
    pixel (block average), s = sqrt(ivar) and W an optional Fourier weight (e.g. counting the CMB as noise).

    Parameters
    ----------
    ivar : NDArray[np.floating]
        ACT inverse variance per fine pixel, as used in the operator.
    wdata : NDArray[np.floating]
        ivar times the ACT map, on the fine pixels.
    wmap : WavSkyMap
        The needlet map.
    beam_fwhm_pix : float
        ACT beam FWHM in fine pixels.
    bands : Sequence[int]
        Bands in which ACT is used.
    weight_ft : NDArray[np.floating] | None, default: None
        Fourier weight W (rfft2 layout, as gaussian_beam_ft), None for white noise.
    """

    def __init__(
        self,
        ivar: NDArray[np.floating],
        wdata: NDArray[np.floating],
        wmap: WavSkyMap,
        beam_fwhm_pix: float,
        bands: Sequence[int],
        weight_ft: NDArray[np.floating] | None = None,
    ):
        self.ivar = ivar
        self.act_map = np.divide(wdata, ivar, out=np.zeros_like(wdata), where=ivar > 0)
        self.sqrt_ivar = np.sqrt(ivar)
        self.beam_ft = gaussian_beam_ft(ivar.shape, beam_fwhm_pix)
        self.sqrt_weight = None if weight_ft is None else np.sqrt(weight_ft)
        self.wmap = wmap
        self.bands = list(bands)
        self.off = [i for i in range(wmap.needlet.nfilt) if i not in self.bands]

    def smooth(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Apply W^(1/2).

        Parameters
        ----------
        y : NDArray[np.floating]
            Map.

        Returns
        -------
        NDArray[np.floating]
            Weighted map (the map itself without a weight).
        """
        return y if self.sqrt_weight is None else apply_beam(y, self.sqrt_weight)

    def forward(self, m: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        The sky as ACT sees it: beam, then the ACT pixel average.

        Parameters
        ----------
        m : NDArray[np.floating]
            Map on the fine grid.

        Returns
        -------
        NDArray[np.floating]
            Map on the fine grid, constant on each ACT pixel.
        """
        return block_average(apply_beam(m, self.beam_ft), ACT_OSAMP)

    def adjoint(self, m: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Adjoint of forward.

        Parameters
        ----------
        m : NDArray[np.floating]
            Map on the fine grid.

        Returns
        -------
        NDArray[np.floating]
            Map on the fine grid.
        """
        return apply_beam(block_average(m, ACT_OSAMP), self.beam_ft)

    def forward_w(self, m: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        The whitened forward operator W^(1/2) s B.

        Parameters
        ----------
        m : NDArray[np.floating]
            Map on the fine grid.

        Returns
        -------
        NDArray[np.floating]
            Whitened ACT-space map.
        """
        return self.smooth(self.sqrt_ivar * self.forward(m))

    def adjoint_w(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Adjoint of forward_w.

        Parameters
        ----------
        y : NDArray[np.floating]
            Whitened ACT-space map.

        Returns
        -------
        NDArray[np.floating]
            Map on the fine grid.
        """
        return self.adjoint(self.sqrt_ivar * self.smooth(y))

    def rhs(self, resid: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Right hand side F_i B^T s W s (ACT map residual) on the ACT bands.

        Parameters
        ----------
        resid : NDArray[np.floating]
            The ACT data the term should fit (the ACT map, or its residual after a model).

        Returns
        -------
        NDArray[np.floating]
            Wavelet array, zero outside the ACT bands.
        """
        out = map2wav_real(
            self.adjoint_w(self.smooth(self.sqrt_ivar * resid)),
            self.wmap.needlet.filters,
        )[0]
        out[self.off] = 0
        return out

    def dot(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        The term applied to wavelet coefficients, on the ACT bands only.

        Parameters
        ----------
        x : NDArray[np.floating]
            Wavelet array.

        Returns
        -------
        NDArray[np.floating]
            Wavelet array, zero outside the ACT bands.
        """
        filt = self.wmap.needlet.filters
        xa = x.copy()
        xa[self.off] = 0
        ya = map2wav_real(
            self.adjoint_w(self.forward_w(wav2map_real(xa, filt)[0])), filt
        )[0]
        ya[self.off] = 0
        return ya

    def responses(self, radius, exact_below: int) -> Callable[[list[int]], NDArray]:
        """
        Function computing the response matrices of the term for a list of bands (for cached_responses).

        Parameters
        ----------
        radius : Sequence[float] | None
            Window of the response matrices per band (see WavSkyMap.get_response_matrix_map).
        exact_below : int
            Bands with at most this many lattice points are computed exactly.

        Returns
        -------
        Callable[[list[int]], NDArray]
            bands -> response matrices.
        """
        return lambda bands: self.wmap.get_response_matrix_map(
            np.ones(self.ivar.shape),
            forward=self.forward_w,
            adjoint=self.adjoint_w,
            nfilts=bands,
            radius=radius,
            exact_below=exact_below,
        )
