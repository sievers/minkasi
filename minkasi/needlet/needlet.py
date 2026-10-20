import itertools
import sys
import time
from functools import reduce
from typing import (
    Callable,
    Optional,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from astropy import wcs
from astropy.io import fits
from numpy.typing import NDArray
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.linalg import norm
from scipy.sparse.linalg import spsolve

from ..mapmaking.map2tod import map2tod
from ..mapmaking.tod2map import (
    tod2map_cached,
    tod2map_simple,
)
from ..maps.mapset import Mapset
from ..maps.skymap import SkyMap
from ..tods import Tod, TodVec
from ..tools.py_fft import irfftn, rfftn

if sys.version_info >= (3, 11):
    pass
else:
    pass


# TODO: This function should probably go elsewhere
def factors(n):
    return np.array(
        sorted(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
        )
    )


def color_groups(n, width):
    """
    Split a periodic lattice of n points into groups (colors) such that any two points of
    one group are at least width apart, including across the wrap, using as few groups as possible.
    The lattice is cut into floor(n / width) blocks of near equal length and the
    k-th group takes the k-th point of every block that has one.

    Parameters
    ----------
    n : int
        Number of lattice points.
    width : int
        Minimum separation within a group.

    Returns
    -------
    groups : list[NDArray[int]]
        Indices of each group. Every point is in exactly one group.
    """
    nblock = max(n // width, 1)
    edges = np.linspace(0, n, nblock + 1).astype(int)
    lens = np.diff(edges)
    return [edges[:-1][lens > k] + k for k in range(lens.max())]


def band2map(band, filt):
    """
    Single filter version of wav2map_real: real space map of the wavelet coefficients of one filter.

    Parameters
    ----------
    band : NDArray[np.floating]
        Wavelet coefficients of shape (nx, ny).
    filt : NDArray[np.floating]
        The filter window for this band, shape (nx, ny).

    Returns
    -------
    NDArray[np.floating]
        Real space map, shape (nx, ny).
    """
    ft = np.fft.rfft2(np.fft.fftshift(band))
    ft = ft * np.fft.fftshift(filt)[:, : band.shape[1] // 2 + 1]
    return np.fft.fftshift(np.fft.irfft2(ft, s=band.shape))


def map2band(imap, filt):
    """
    Single filter version of map2wav_real. The filter is symmetric so this equals band2map.

    Parameters
    ----------
    imap : NDArray[np.floating]
        Real space map, shape (nx, ny).
    filt : NDArray[np.floating]
        The filter window for this band, shape (nx, ny).

    Returns
    -------
    NDArray[np.floating]
        Wavelet coefficients, shape (nx, ny).
    """
    return band2map(imap, filt)


################################################################
#                       Basis Functions                        #
################################################################


def __Standard(
    k_arr: NDArray[np.floating], j: int, B: np.floating
) -> NDArray[np.floating]:
    """
    Driver function for generating standard needlets

    Parameters
    ----------
    k_arr : NDArray[np.floating]
        Array of k at which to evaluate needlet.
    j : int
        The index of the needlet.
    B : np.floating
        Parameter which sets the needlet width

    Returns
    -------
    to_ret : NDArray[np.floating]
        1D standard needlet response function for index j
    """

    def __f_need(t):
        """Auxiliar function f to define the standard needlet"""
        if t <= -1.0:
            return 0.0
        elif t >= 1.0:
            return 0.0
        else:
            return np.exp(1.0 / (t**2.0 - 1.0))

    def __psi(u):
        """Auxiliar function psi to define the standard needlet"""
        return integrate.quad(__f_need, -1, u)[0] / integrate.quad(__f_need, -1, 1)[0]

    def __phi(q, B):
        """Auxiliar function phi to define the standard needlet"""
        B = float(B)
        if q < 0.0:
            raise ValueError("The multipole should be a non-negative value")
        elif q <= 1.0 / B:
            return 1.0
        elif q >= 1.0:
            return 0
        else:
            return __psi(1.0 - (2.0 * B / (B - 1.0) * (q - 1.0 / B)))

    xi = k_arr / B**j
    b2 = __phi(xi / B, B) - __phi(xi, B)
    return np.max([0.0, b2])


def Standard(
    k_arr: NDArray[np.floating], js: NDArray[int], B: np.floating
) -> NDArray[np.floating]:
    """
    Helper function for constructing standard needlet basis.
    TODO: this might be cleaner with np.vectorize.

    Parameters
    ----------
    k_arr : NDArray[np.floating]
        Array of k at which to evaluate needlets.
    js : NDArray[int]
        The indicies of the needlets.
    B : np.floating
        Parameter which sets the needlet width

    Returns
    -------
    to_ret : NDArray[np.floating]
        1D standard needlet response functions
    """

    to_ret = []
    bl2 = np.vectorize(__Standard)
    for j in js:
        to_ret.append(np.sqrt(bl2(k_arr, j, B)))

    return np.array(to_ret)


def Mexican(
    k_arr: NDArray[np.floating],
    js: NDArray[int],
    jmin: int = 1,
    B: np.floating = 1.5,
    p: np.floating = 1,
) -> NDArray[np.floating]:
    """
    Function for constructing Mexican at needlets.

    Parameters
    ----------
    k_arr : NDArray[np.floating]
        Array of k at which to evaluate needlets.
    js : NDArray[int]
        The index of the needlets.
    jmin : int, default: 1
        Combine all needlets up to jmin into the first needlet.
        The first needlets can have very thin support so it
        sometimes helps to add up the first couple needlets.
    B : np.floating, default: 1.5
        Parameter which sets the needlet width
    p : np.floating, default: 1
        Another parameter which sets needlet shape.

    Returns
    -------
    to_ret : NDArray[np.floating]
        1D Mexican hat needlet response functions
    """

    bs = np.zeros((len(js), len(k_arr)))

    for j in range(len(js)):
        u = k_arr / B**j
        bs[j] = u**p * np.exp(-1 / 2 * u**2)

    bs[0][0] = 1
    to_ret = np.zeros((len(js) - jmin, len(k_arr)))
    to_ret[0] = np.sqrt(np.sum(bs[:jmin] ** 2, axis=0))

    for j in range(1, len(js) - jmin):
        to_ret[j] = bs[jmin + j]

    norm = np.sqrt(
        np.sum(to_ret**2, axis=0)
    )  # For Mexican hat filters you have to just force the normalization
    to_ret /= norm

    return to_ret


def __CosNeed(
    k_arr: NDArray[np.floating], j: int, cs: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Driver function for cosine needlet basis.
    Parameters
    ----------
    k_arr : NDArray[np.floating]
        Array of k at which to evaluate needlets.
    j : NDArray[int]
        The current filter being considered.
    cs : NDArray[np.floating]
        Centers of the cosine filters.
    Returns
    -------
    to_ret : NDArray[np.floating]
        One cosine needlet response function
    """

    to_ret = np.zeros(len(k_arr))
    if j == 0:
        to_ret[k_arr <= cs[1]] = np.cos(
            (np.pi * k_arr[k_arr <= cs[1]] / (2 * cs[1] - cs[0]))
        )
    else:
        flag = np.where((cs[j - 1] < k_arr) & (k_arr < cs[j]))
        to_ret[flag] = np.cos(
            (np.pi * (cs[j] - k_arr[flag])) / (2 * (cs[j] - cs[j - 1]))
        )
        flag = np.where((cs[j] < k_arr) & (k_arr < cs[j + 1]))
        to_ret[flag] = np.cos(
            (np.pi * (k_arr[flag] - cs[j])) / (2 * (cs[j + 1] - cs[j]))
        )

    return to_ret


def CosNeed(
    k_arr: NDArray[np.floating], js: NDArray[int], cs: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Helper function for constructing Cosine needlet basis.
    TODO: This is probably cleaner with a np.vectorize.

    Parameters
    ----------
    k_arr : NDArray[np.floating]
        Array of k at which to evaluate needlets.
    js : NDArray[int]
        The index of the needlets. Unused except
        that for Cosine needlets the 0th needlet
        is treated differently.
    cs : NDArray[np.floating]
        Centers of the cosine filters.

    Returns
    -------
    to_ret : NDArray[np.floating]
        1D cosine needlet response functions
    """

    to_ret = []
    for j in js:
        to_ret.append(__CosNeed(k_arr, j, cs))

    return np.array(to_ret)


################################################################
#                         Core Classes                         #
################################################################


class Needlet:
    """
    Class for making needlet frame
    Code from  Joelle Begin, modifications by JackOS

    js : NDArray[int]
        Resolution parameter; effectively sets central location of bandpass
        filter.
    lightcone : NDArray(np.floating)
        Real space map to make needlets for. First axis can be either frequency or z,
        hence the name.
    kmax_dimless : int
        Maximum k to evaluate needlets at.
    lightcone_box : cosmo_box
        Helper class used to compute k-space grid for needlets.
    nfilt : int
        Number of needlet filters.
    k_arr : NDArray[np.floating]
        Array of k at which to evaluate needlets.
    B : np.floating
        Standard Needlet scaling factor. See https://iopscience.iop.org/article/10.1088/0004-637X/723/1/1
    bands : NDArray[np.floating]
        Needlet bands in 1D.
    filters : NDArray[np.floating]
        Needlet bands in 2D.
    delta_k : np.floating
        Dimensions of k spacing
    """

    def __init__(
        self,
        js: NDArray[int],
        L: np.floating,
        lightcone: NDArray[np.floating],
        pixsize: np.floating,
        basisKwargs: Optional[dict] = None,
        basis: Callable[..., NDArray[np.floating]] = Standard,
        B: Union[None, np.floating] = None,
        kmax_dimless: Union[None, int] = None,
    ):
        """
        Init function.

        Parameters:
        -----------
        js : NDArray[int]
            Resolution parameter; effectively sets central location of bandpass
            filter.
        L : np.floating
            Realspace side length of lightcone. Must be same units as pixsize.
        lightcone : NDArray[np.floating]
            Real space map to make needlets for.
        pixsize : np.floating
            pixsize corresponding to the pixels of lightcone. Must be same units as L.
        basisKwargs : dict
            Dictionary of keywords to be passed to basis. Different bases requrie different parameters.
        B :  None | np.floating
            The parameter B of the needlet which controls the width of the filters.
            Should be larger that 1. If none a suitible B is calculated
        kmax_dimless : None | int
            The maximum k mode for which the filters will be constructed. Dimensionless units!
            If None it is calculated.
        basis : Callable[..., NDArray[np.floating]]
            Basis needlet functions b to use. Several are defined in this file but your own can be used.
            Note this class does not check that your basis function is valid, i.e. that it satisfies
            b^2 has support in [1/B, B]
            b is infinitiely differentiable in [0, inf]
            sum(b^2) = 1
        """

        ####### init attributes
        self.js = js
        self.lightcone = lightcone
        self.kmax_dimless = kmax_dimless
        self.L = L
        self.basis = basis
        self.pixsize = pixsize

        if self.lightcone is not None:
            self.lightcone_box = cosmo_box(lightcone, self.L)
            self.kmax_dimless = self.lightcone_box.kmax_dimless

        self.k_arr = np.append(
            np.array([0]),
            np.logspace(0, np.log10(self.kmax_dimless), int(10 * self.kmax_dimless)),
        )

        self.delta_k = 2 * np.pi / self.L

        if B is not None:
            self.B = B
        else:
            self.B = (
                self.k_arr[-1] ** (1 / self.js[-1]) * 1.01
            )  # Set needlet width to just cover k_arr

        self.basisKwargs = {} if basisKwargs is None else dict(basisKwargs)

        if self.basis.__name__ == "Standard" and "B" not in self.basisKwargs.keys():
            self.basisKwargs["B"] = self.B

        if (
            self.basis.__name__ == "CosNeed" and "cs" not in self.basisKwargs.keys()
        ):  # strangely self.basis == CosNeed does not evaluate correctly
            self.basisKwargs["cs"] = np.linspace(
                0, self.kmax_dimless * 1.15, len(js) + 1
            )

        self.bands = self.get_needlet_bands_1d(self.basisKwargs)
        if self.basis.__name__ == "Mexican":
            # Shrink js for Mexican
            self.js = np.arange(self.bands.shape[0])

        self.nfilt = len(self.js)

    def get_needlet_bands_1d(self, basisKwargs):
        """
        Get 1D needlet response given parameters in basisKwargs.

        Parameters
        ---------
        basisKwargs : dict
            Dictionary that defines the parameters used by he needlet basis.

        Returns
        -------
        needs : NDArray[np.floating]
            The 1D needlet response.
        """
        needs = self.basis(self.k_arr, self.js, **basisKwargs)
        needs[0][0] = 1  # Want the k=0 mode to get the map average
        return needs

    def get_needlet_filters_2d(
        self,
        fourier_radii: NDArray[np.floating],
        return_filt: Optional[bool] = False,
        plot: Optional[bool] = False,
    ) -> Optional[NDArray[np.floating]]:
        """
        Turns 1D needlet response into 2D response by revolving in k space

        Parameters
        ---------
        Fourier_radii : NDArray[np.floating]
            1D needlet responses
        return_filt : bool, default: False
            If true, return the 2D filters in addition to setting self.filters
        plot : bool, default: False
            If true, plot the 2D filter functions

        Returns
        ------
        self.filters : NDArray[np.floating | None
            The 2D filter functions. None if return_filt is false.
        """
        filters = []
        for j in self.js:
            interp_func = interp1d(
                self.k_arr, self.bands[j], fill_value="extrapolate"
            )  # interpolating is faster than computing bands for every row.
            # We should not extrapolate but occasionally the last bin will be very slightly outside range
            filter_2d = []
            for row in fourier_radii:
                filter_2d.append(interp_func(row))
            filters.append(np.array(filter_2d))

        self.filters = np.array(filters)

        if plot:
            n = self.nfilt // 2
            fig, ax = plt.subplots(nrows=2, ncols=n)

            for i in range(n):
                ax[0, i].imshow(self.filters[i])
                ax[0, i].set_title(f"2D filter for j={i}")
                ax[0, i].set_yticklabels([])
                ax[0, i].set_xticklabels([])

                ax[1, i].imshow(self.filters[i + n])
                ax[1, i].set_title(f"2D filter for j={i + n}")
                ax[1, i].set_yticklabels([])
                ax[1, i].set_xticklabels([])

            plt.show()

        if return_filt:
            return self.filters

    def get_need_lims(self, N: int, real_space: bool = False):
        """
        Returns the limits of needlet

        Parameters
        ---------
        N : int
            Needlet of interest.
        real_space : bool
            Whether to return the limits in k or real space units

        Ouputs
        ------
        lims : NDArray[np.floating]
            Lower and upper limit, respectively, of the needlet band
        """

        lim_idx = np.where((self.bands[N] != 0))[0]
        lims = np.array([self.k_arr[lim_idx[0]], self.k_arr[lim_idx[-1]]])
        if real_space:
            lims *= self.delta_k
            lims = np.flip(lims)
            if N == 0:
                lims[1] = 1e-12  # The first need has lower k limit = 0
            lims = np.pi / (lims)  # Unsure about this factor of pi
        return lims

    # ==============================================================================================#
    # ====================================== plotting functions ====================================#
    # ==============================================================================================#

    def plot_bands(self, scale: Optional[np.floating] = None):
        """
        Function that plots the needlets in k space, sometimes called the windows.

        Parameters
        ----------
        scale : Optional[np.floating] | None, default: None
            Inverse physical scale corresponding to k
        """
        fig, ax = plt.subplots()

        self.sum_sq = np.zeros_like(self.k_arr)
        x_arr = self.k_arr
        if scale:
            x_arr = x_arr * scale
        for j, b in enumerate(self.bands):
            ax.plot(x_arr, b, label=f"j={j}")
            self.sum_sq += b**2

        ax.plot(x_arr, self.sum_sq, label=r"$\sum b^2$", color="k")
        ax.set_xscale("log")
        ax.legend(loc="lower right", ncols=self.nfilt // 2)
        ax.set_xlabel("k [dimless]")
        ax.set_ylabel(r"$b_j$")
        plt.show()


class WavSkyMap(SkyMap):
    """
    Wavelet based SkyMap.
    Subclass of SkyMap. See Skymap for full documentation.


    Attributes
    ----------
    filters : NDArray[np.floating]
        An array specifying the needlet filter response from needlet class
    nfilt : int
        Number of needlet filters
    map : NDArray[np.floating]
        Array of wavelet coefficients. Can also be thought of as the wavelet space map.
    real_map : ~minkasi.maps.SkyMap
        SkyMap object containing the real sky counterpart to map.
    isglobal_prior : bool
       True if this map is a prior to be applied to other maps.
    """

    def __init__(
        self,
        needlet: Needlet,
        lims: Union[list[float], NDArray[np.floating]],
        pixsize: float,
        proj: str = "CAR",
        pad: int = 2,
        square: bool = False,
        multiple: Union[int, bool] = False,
        primes: Union[list[int], None] = None,
        cosdec: Union[float, None] = None,
        nx: Union[int, None] = None,
        ny: Union[int, None] = None,
        mywcs: Union[wcs.WCS, None] = None,
        tag: str = "ipix",
        purge_pixellization: bool = False,
        ref_equ: bool = False,
        isglobal_prior: bool = False,
    ):
        """
        Initialize the SkyMap.

        Parameters
        ----------
        needlet : Needlet
            A needlet class object containing the needlets associated with the WavSkyMap.
            Currently you must separately initialized the needlet object and make sure it
            has the same bounds as this WavSkyMap instance but I should change this to move
            the needlet construction inside of WavSkyMap.
        isglobal_prior: bool = False
            Specifies if this map is a prior to be applied to other maps.
        See SkyMap docstring for remainder
        """
        self.needlet = needlet
        self.isglobal_prior = isglobal_prior
        self.downsamps = None
        super().__init__(
            lims,
            pixsize,
            proj,
            pad,
            square,
            multiple,
            primes,
            cosdec,
            nx,
            ny,
            mywcs,
            tag,
            purge_pixellization,
            ref_equ,
        )
        self.nfilt = len(self.needlet.filters)
        self.map = np.zeros([self.nfilt, self.nx, self.ny])
        self.real_map = SkyMap(
            lims,
            pixsize,
            proj,
            pad,
            square,
            multiple,
            primes,
            cosdec,
            nx,
            ny,
            mywcs,
            tag,
            purge_pixellization,
            ref_equ,
        )  # This is very slightly inefficient as it redoes the ssquaring but is safer in avoiding getting a real_map and wave_map with different shapes

    def map2tod(
        self,
        tod: "Tod",
        dat: NDArray[np.floating],
        do_add: bool = True,
        do_omp: bool = True,
        n_filt: Optional[list[int]] = None,
    ):
        """
        Project a wavelet map into a tod, adding or replacing the map contents.
        First converts from wavelet map to realspace map, then from realspace map to tod.

        Parameters
        ----------
        tod : Tod
            Tod object, used to get pixellization.
        dat : NDArray[np.floating]
            Array to put tod data into.
            Shape should be (ndet, ndata).
        do_add : bool, default: True
            If True add the projected map to dat.
            If False replace dat with it.
        do_omp : bool, default: False
            Use omp to parallelize
        n_filt :  None | list[int], default: None
            Filters on which to perform map2tod. If none, then done over all
        """
        ipix = self.get_pix(tod)
        self.real_map.map = np.squeeze(
            wav2map_real(self.map, self.needlet.filters, n_filt=n_filt), axis=0
        )  # Right now let's restrict ourself to 1 freqency input maps, so we need to squeeze down the dummy axis added by map2wav_real
        map2tod(dat, self.real_map.map, ipix, do_add, do_omp)

    def tod2map(
        self,
        tod: "Tod",
        dat: NDArray[np.floating],
        do_add: bool = True,
        do_omp: bool = True,
        n_filt: Optional[list[int]] = None,
    ):
        """
        Project a tod into the wmap. Frist projects tod onto real space map, then converts that to wmap.

        Parameters
        ----------
        tod : Tod
            Tod object, used to get pixellization.
        dat : NDArray[np.floating]
            Array to pull tod data from.
            Shape should be (ndet, ndata).
        do_add : bool, default: True.
            If True add the projected map to this map.
            If False replace this map with it.
        do_omp : bool, default: True
            Use omp to parallelize.
        n_filt : None | list[int], default: None
            Filters on which to perform map2tod. If none, then done over all

        """
        if dat is None:
            dat = tod.get_data()
        if do_add == False:
            self.clear()
        ipix = self.get_pix(tod)

        if not (self.caches is None):
            tod2map_cached(self.caches, dat, ipix)

        self.real_map.clear()

        tod2map_simple(self.real_map.map, dat, ipix)
        if not do_add:
            self.map = np.squeeze(
                map2wav_real(self.real_map.map, self.needlet.filters, n_filt=n_filt),
                axis=0,
            )  # Right now let's restrict ourself to 1 freqency input maps, so we need to squeeze down the dummy axis added by map2wav_real

        else:
            self.map += np.squeeze(
                map2wav_real(self.real_map.map, self.needlet.filters, n_filt=n_filt),
                axis=0,
            )

        if self.purge_pixellization:
            tod.clear_saved_pix(self.tag)

    def write(self, fname: str = "map.fits"):
        """
        Write map to a FITs file.

        Parameters
        ----------
        fname : str, default: 'map.fits'
            The path to save the map to.
        """
        self.real_map.map = np.squeeze(
            wav2map_real(self.map, self.needlet.filters), axis=0
        )
        header = self.wcs.to_header()
        if True:  # try a patch to fix the wcs xxx
            tmp = self.real_map.map.transpose().copy()
            hdu = fits.PrimaryHDU(tmp, header=header)
        else:
            hdu = fits.PrimaryHDU(self.real_map.map, header=header)
        hdu.writeto(fname, overwrite=True)

    def apply_prior(self, p: "WavSkyMap", Ap: "WavSkyMap"):
        """
        Apply prior to the wavelet map. In the ML framework, adding a prior equates to:
        chi^2 -> chi^2 + m^TQ^-1m
        for m the map, Q the prior map. Per Jon, the modified the map equation to:
        (A^T N^-1 A - Q^-1)m = A^T N^-1d
        Note (A^T N^-1 A)m is the output of tods.dot(p). This function then simply performs
        (A^T N^-1 A)m - Q^-1m, although the signage is confusing.

        Parameters
        ----------
        p : WavSkyMap
            Wavelet SkyMap of the conjugate vector p.
        Ap : WavSkyMap
            The result of tods.dot(p)

        Returns
        -------
        Modifies Ap in place by Q^-1 m
        """
        Ap.map = Ap.map + self.map * p.map

    def get_lattice(self, downsamps):
        """
        Get the evenly spaced sub-lattice of pixels kept at each filter after downsampling.

        Parameters
        ----------
        downsamps : NDArray[int]
            Downsampling factor per filter. Each must divide both nx and ny.

        Returns
        -------
        nx_space, ny_space : NDArray[object]
            Per filter, the (reduced) arrays of retained x and y pixel indices.
        """
        nx_space = np.empty(self.needlet.nfilt, dtype=object)
        ny_space = np.empty(self.needlet.nfilt, dtype=object)
        for i, down in enumerate(downsamps):
            down = int(down)
            if self.nx % down or self.ny % down:
                raise ValueError(
                    f"downsamp {down} must divide the map shape {self.nx, self.ny}"
                )
            nx_space[i] = np.arange(down // 2, self.nx, down)
            ny_space[i] = np.arange(down // 2, self.ny, down)
        return nx_space, ny_space

    def get_downsamps(self):
        """
        Resizes maps so that the pixelization matches the smallest scale of the needlet,
        and sets downsamps, nxs_red, nys_red, nx_space, ny_space accordingly.
        The downsamp of each filter is the largest common factor of nx and ny that
        does not undersample the filter.
        """
        map_size = (
            np.rad2deg(max(self.lims[1] - self.lims[0], self.lims[3] - self.lims[2]))
            * 3600
        )
        size_factors = factors(np.gcd(self.nx, self.ny))
        downsamps = np.empty(self.needlet.nfilt, dtype=int)
        for i in range(self.needlet.nfilt):
            lims = self.needlet.get_need_lims(i, real_space=True)
            n_samp = map_size / lims[0]
            downsamp = max(np.floor(self.nx / n_samp), 1)
            downsamps[i] = size_factors[size_factors <= downsamp][-1]
        self.set_downsamps(downsamps)

    def set_downsamps(self, downsamps):
        """
        Manually set the downsampling factor of each filter and the resulting lattices.

        Parameters
        ----------
        downsamps : NDArray[int]
            Downsampling factor per filter.
        """
        self.downsamps = np.asarray(downsamps, dtype=int)
        self.nx_space, self.ny_space = self.get_lattice(self.downsamps)
        self.nxs_red = np.array([len(n) for n in self.nx_space])
        self.nys_red = np.array([len(n) for n in self.ny_space])

    def select_filters(self, max_res=None, nfilts=None):
        """
        Get the filters to compute from max_res or nfilts (mutually exclusive).

        Parameters
        ----------
        max_res : np.floating | None
            Maximum resolution, in arcseconds. Wavelets with maximum extent below it are skipped.
        nfilts : list[int] | None
            Explicit list of filters.

        Returns
        -------
        nfilts : list[int]
            Filters to compute.
        """
        if max_res is not None:
            if nfilts is not None:
                raise ValueError("Cannot specify max_res and nfilts")
            i = 0
            while self.needlet.get_need_lims(i, real_space=True)[0] > max_res:
                i += 1  # not the fastest way to do this but it's only done once
            return list(range(i + 1))
        return list(range(self.needlet.nfilt)) if nfilts is None else list(nfilts)

    def check_response_matrix(self, filt_num: int, down_samp=None):
        """
        Get the dense response matrix of needlet filt_num: row i is the band map of a unit
        impulse at retained pixel i, flattened. Used to check we haven't over downsampled;
        the SVD of this should not fall off a cliff in the last couple of modes.

        Parameters
        ---------
        filt_num : int
            The needlet to compute the repsponse matrix for.
        down_samp : None | NDArray[int]
            Downsample factor per filter. If None the most aggressive (from get_downsamps) is used.

        Returns
        -------
        to_ret : NDArray[np.floating]
            Response matrix of shape (nx_red * ny_red, nx * ny).
        """
        if down_samp is None:
            self.get_downsamps()
        else:
            self.set_downsamps(down_samp)
        nx_space, ny_space = self.nx_space[filt_num], self.ny_space[filt_num]
        filt = self.needlet.filters[filt_num]
        to_ret = np.zeros((len(nx_space) * len(ny_space), self.nx * self.ny))
        for idx, (x, y) in enumerate(itertools.product(nx_space, ny_space)):
            temp = np.zeros((self.nx, self.ny))
            temp[x, y] = 1
            to_ret[idx] = map2band(band2map(temp, filt), filt).ravel()
        return to_ret

    def probe_band(self, apply, nfilt, radius=None, threshold=0.0):
        """
        Compute G = K^T O K for the lattice of band nfilt, where K maps lattice coefficients
        to the band-limited real space map and O is the operator `apply`.
        Rows are computed by probing: impulses spaced further apart than the window
        are summed into one map and passed through `apply` together, so only
        a few applications are needed instead of one per pixel. With radius=None every
        pixel is probed separately and the result is exact.

        Parameters
        ----------
        apply : Callable
            Operator taking and returning a real space map of shape (nx, ny).
        nfilt : int
            Filter to compute.
        radius : float | None, default: None
            Half width of kept window, in full resolution pixels. None keeps everything (exact).
        threshold : float, default: 0
            Entries below threshold * max|G| are dropped.

        Returns
        -------
        G : scipy.sparse.csr_matrix
            Symmetric matrix of shape (nx_red * ny_red, nx_red * ny_red).
        """
        filt = self.needlet.filters[nfilt]
        xs, ys = self.nx_space[nfilt], self.ny_space[nfilt]
        nxr, nyr = len(xs), len(ys)
        down = self.downsamps[nfilt]
        if radius is None:
            rx, ry = nxr, nyr
        else:
            rx = ry = int(np.ceil(radius / down))
        full_x, full_y = 2 * rx + 1 >= nxr, 2 * ry + 1 >= nyr
        offx = np.arange(nxr) if full_x else np.arange(-rx, rx + 1)
        offy = np.arange(nyr) if full_y else np.arange(-ry, ry + 1)
        rows, cols, vals = [], [], []
        for ax, ay in itertools.product(
            color_groups(nxr, 2 * rx + 1), color_groups(nyr, 2 * ry + 1)
        ):
            delta = np.zeros((self.nx, self.ny))
            delta[np.ix_(xs[ax], ys[ay])] = 1
            z = map2band(apply(band2map(delta, filt)), filt)
            for a, b in itertools.product(ax, ay):
                ix = offx if full_x else (a + offx) % nxr
                iy = offy if full_y else (b + offy) % nyr
                vals.append(z[np.ix_(xs[ix], ys[iy])].ravel())
                cols.append((ix[:, None] * nyr + iy[None, :]).ravel())
                rows.append(np.full(vals[-1].size, a * nyr + b))
        vals = np.concatenate(vals)
        keep = np.abs(vals) > threshold * np.abs(vals).max()
        G = sparse.coo_matrix(
            (vals[keep], (np.concatenate(rows)[keep], np.concatenate(cols)[keep])),
            shape=(nxr * nyr, nxr * nyr),
        ).tocsr()
        return ((G + G.T) / 2).tocsr()

    def get_response_matrix(
        self,
        todvec: "TodVec",
        max_res: Optional[np.floating] = None,
        nfilts: Optional[list[int]] = None,
        downsamps: Optional[list[int]] = None,
        radius: Optional[float] = None,
        threshold: float = 0.0,
    ):
        """
        Computes A^T N^-1 A (ANA) for a set of tods in the WavSkyMap basis, one filter
        (diagonal block) at a time, on the downsampled lattice of each filter.
        ANA can be inverted to directly solve for the map, however is expensive to compute.
        In general ANA only needs to be computed for the overlap scales, and other methods
        (e.g. pcg) can be used for the very high/very low resolution scales.

        Parameters:
        -----------
        todvec : TodVec
            TODs over which to compute ANA
        max_res : np.floating, default : None
            Maximum resolution, in arcseconds, at which to compute ANA.
            Wavelets with maximum extent less than max_res are skipped.
        nfilts : list[int], default : None
            Explicitly specify which wavelets to compute.
            Exclusive with max_res.
        downsamps : list[int], default : None
            Explicitly set downsamp rates. If None, good downsamps are computed.
        radius : float, default : None
            Half width in pixels beyond which ANA is neglected. This makes the computation much
            faster (one TOD pass per ~(2*radius/downsamp)^2 pixels rather than one per pixel)
            and the output sparse. None computes everything exactly.
        threshold : float, default : 0
            Relative size below which entries are dropped.

        Returns
        ------
        to_ret : NDArray[object]
            Sparse ANA of each computed filter, None for filters not computed.
        """
        if downsamps is None:
            self.get_downsamps()
        else:
            self.set_downsamps(downsamps)
        mapset = Mapset()
        mapset.add_map(self.real_map.copy())

        def apply(imap):
            mapset.maps[0].map[:] = imap
            return todvec.dot(mapset, cache_maps=False).maps[0].map

        to_ret = np.full(self.needlet.nfilt, None, dtype=object)
        for nfilt in self.select_filters(max_res, nfilts):
            to_ret[nfilt] = self.probe_band(apply, nfilt, radius, threshold)
        return to_ret

    def get_response_matrix_map(
        self,
        imap,
        max_res: Optional[np.floating] = None,
        nfilts: Optional[list[int]] = None,
        radius: Optional[float] = None,
        threshold: float = 0.0,
    ):
        """
        Computes K^T O K, the map space analogue of get_response_matrix, where O is a map space
        inverse covariance (precision) matrix, using the downsamps currently set on this map
        (computed if not set).

        Parameters:
        -----------
        imap : NDArray[np.floating] | scipy.sparse matrix
            Inverse covariance of the map. Either a (nx*ny, nx*ny) matrix or
            an array with nx*ny/(nx, ny) entries, taken to be the diagonal.
        max_res, nfilts, radius, threshold :
            See get_response_matrix.

        Returns
        ------
        to_ret : NDArray[object]
            Sparse response of each computed filter, None for filters not computed.
        """
        if self.downsamps is None:
            self.get_downsamps()
        if getattr(imap, "ndim", 2) == 2 and imap.shape == (self.nx, self.ny):
            imap = imap.ravel()
        if getattr(imap, "ndim", 2) == 1:
            diag = imap.reshape(self.nx, self.ny)
            apply = lambda m: diag * m
        else:
            apply = lambda m: (imap @ m.ravel()).reshape(self.nx, self.ny)
        to_ret = np.full(self.needlet.nfilt, None, dtype=object)
        for nfilt in self.select_filters(max_res, nfilts):
            to_ret[nfilt] = self.probe_band(apply, nfilt, radius, threshold)
        return to_ret

    def solve_bands(self, rhs, responses, ridge=1e-8):
        """
        Solve the per filter normal equations G c = rhs for the lattice coefficients, with
        G the sum of the supplied response matrices, and return the wavelet map
        (coefficients zero filled onto the full resolution grid).
        Filters for which no response is available are left as zero.

        Parameters
        ----------
        rhs : NDArray[np.floating]
            Wavelet space A^T N^-1 d of shape (nfilt, nx, ny).
        responses : Sequence[NDArray[object]]
            Response matrices (e.g. from get_response_matrix and get_response_matrix_map)
            to sum, each an object array with sparse matrices or None for each filter.
        ridge : float, default: 1e-8
            Relative diagonal regularization added to G, to keep unhit pixels invertible.

        Returns
        -------
        wmap : NDArray[np.floating]
            Wavelet map of shape (nfilt, nx, ny). Use wav2map_real to get the real space map.
        """
        wmap = np.zeros((self.needlet.nfilt, self.nx, self.ny))
        for i in range(self.needlet.nfilt):
            mats = [r[i] for r in responses if r[i] is not None]
            if not mats:
                continue
            G = reduce(lambda a, b: a + b, mats).tocsc()
            G = G + ridge * G.diagonal().max() * sparse.identity(
                G.shape[0], format="csc"
            )
            xs, ys = self.nx_space[i], self.ny_space[i]
            c = spsolve(G, rhs[i][np.ix_(xs, ys)].ravel())
            wmap[i][np.ix_(xs, ys)] = c.reshape(len(xs), len(ys))
        return wmap

    def get_wav_beam(self, fwhm_pix: float):
        """
        Get wavelet basis version of beam.

        Parameters:
        -----------
        fwhm_pix : float
           FWHM of beam in pixels.
        """

        xvec = np.arange(-self.map.shape[1] / 2, self.map.shape[1] / 2, 1)
        yvec = np.arange(-self.map.shape[2] / 2, self.map.shape[2] / 2, 1)
        xx, yy = np.meshgrid(yvec, xvec)
        rsqr = xx**2 + yy**2
        sig_pix = fwhm_pix / np.sqrt(8 * np.log(2))
        beam = np.exp(-0.5 * rsqr / (sig_pix**2))
        beam = beam / np.sum(beam)
        self.wavbeam = map2wav_real(np.fft.fftshift(beam), self.needlet.filters)[0]


###############################################################################################


class cosmo_box:
    """
    Class that defines the fourier box our wavelets live in.
    First dim is frequency, so redundant for M2

    Attributes
    ----------
    box : NDArray(np.floating)
        Realspace box with dim [nfreq, nx, ny]
    L : np.floating
        Physical side length of the box. Units arbitrary
    dims : int
        Dimensions of box
    N : int
        Box size in pixels
    origin : int
        FFT convention origin of the box in pixels.
    delta_k : np.floating
        kspace resolution of 1 pixel
    kmax_dimless : np.floating
        Maximum k of the cosmo box corresponding to box
    kmax : np.floating
        Dim-ful version of kmax
    """

    def __init__(self, box, L: np.floating):
        """
        Initialize the cosmo box

        Parameters
        ----------
        box : NDArray(np.floating)
            Realspace box with dim [nfreq, nx, ny]
        L : np.floating
            Physical side length of the box. Units arbitrary
        """

        self.box = box
        self.L = L

        # ----------------------------- box specs ------------------------------#
        self.dims = len(self.box.shape)  # dimensions of box
        self.N = self.box.shape[1]  # number of pixels along one axis of 2D slice
        self.origin = self.N // 2  # origin by fft conventions

        self.delta_k = 2 * np.pi / self.L  # kspace resolution of 1 pixel

        self.kmax_dimless = (
            self.get_kmax_dimless()
        )  # maximum |k| in fourier grid, dimensionless (i.e. pixel space)
        self.kmax = (
            self.kmax_dimless * self.delta_k
        )  # same as above, but with dimensions

    # ======================= fourier functions ====================#
    def get_kmax_dimless(self):
        """
        Gets the maximum k in dimensionless fourier space specified

        Parameters
        ----------

        Returns
        -------
        max_grid_dimless : np.floating
            Maximum k of self.grid_dimless
        """

        self.get_grid_dimless_2d()
        return np.max(self.grid_dimless)

    def get_grid_dimless_2d(self, return_grid: Optional[bool] = False):
        """
        Generates a fourier space dimensionless grid, finds
        radial distance of each pixel from origin.

        Parameters
        ----------
        return_grid : bool, default: False
            If true, then return the grid in addition to setting self.grid_dimless

        Returns
        -------
        self.grid_dimless : NDArray[np.floating]
            Grid of dimensionless 2D k space
        """

        self.indices = np.indices((self.N, self.N)) - self.origin
        self.grid_dimless = norm(
            self.indices, axis=0
        )  # dimensionless kspace radius of each pix

        if return_grid:
            return self.grid_dimless


def map2wav_real(
    imaps: NDArray[np.floating],
    filters: NDArray[np.floating],
    n_filt: Optional[NDArray[np.floating]] = None,
    print_time: Optional[bool] = False,
) -> NDArray[np.floating]:
    """
    Transform from a regular map to a multimap of wavelet coefficients. Adapted from Joelles code + enmap.wavelets

    Parameters
    ----------
    imap : NDArray[np.floating]
        Input map of the sky
    filters : NDArray[np.floating]
        Feedlet basis filters, also called the filter windows
    n_filt : NDArray[np.floating] | None, default: None
        Which filters to perform the coeficient evaluation for.
        By default does all filters
    print_time : bool, default: False
        If true, prints the time taken to perform map2wav
    Returns
    -------
    wmap : NDArray[np.floating]
        Multimap of wavelet coefficients with shape [nfilt, nx, ny]
    """
    toc = time.time()
    if len(imaps.shape) == 2:
        imaps = np.expand_dims(imaps, axis=0)

    elif len(imaps.shape) != 3:
        print("Error: input map must have dim = 2 or 3")
        return

    wmap = []

    if n_filt is None:
        n_filt = range(len(imaps))

    for i in n_filt:
        lightcone_ft = np.fft.fftn(np.fft.fftshift(imaps[i]))

        filtered_slice_real = []
        for filt in filters:
            fourier_filtered = np.fft.fftshift(filt) * lightcone_ft
            filtered_slice_real.append(
                np.fft.fftshift(np.real(np.fft.ifftn(fourier_filtered)))
            )

        wmap.append(np.array(filtered_slice_real))
    tic = time.time()
    if print_time:
        print("map2wav takes ", tic - toc)
    return np.array(wmap)


def wav2map_real(
    wav_mapset: NDArray[np.floating],
    filters: NDArray[np.floating],
    n_filt: Optional[NDArray[np.floating]] = None,
    print_time: Optional[bool] = False,
) -> NDArray[np.floating]:
    """
    Transform from a regular map to a multimap of wavelet coefficients. Adapted from Joelles code + enmap.wavelets

    Parameters
    ----------
    wav_mapset : NDArray[np.floating]
        Input multimap of wavelet coefficients with shape [nfilt, nx, ny]
    filters : NDArray[np.floating]
        Feedlet basis filters, also called the filter windows
    n_filt : NDArray[np.floating] | None, default: None
        Which filters to perform the coeficient evaluation for.
        By default does all filters
    print_time : bool, default: False
        If true, prints the time taken to perform map2wav
    Returns
    -------
    skymap : NDArray[np.floating]
        Skymap corresponding to input filters and wavelet coefficients
    """
    toc = time.time()
    if len(wav_mapset.shape) == 3:
        wav_mapset = np.expand_dims(wav_mapset, axis=0)

    elif len(wav_mapset.shape) != 4:
        print("Error: input wave mapset must have dim = 3 or 4")
        return

    sky_map = []
    if n_filt is None:
        n_filt = range(len(wav_mapset))  # If not provided with filts to do, then do all
    for nu in n_filt:
        fourier_boxes = []
        for b in wav_mapset[nu]:
            fourier_boxes.append(np.fft.fftn(np.fft.fftshift(b)))

        back_transform = np.zeros_like(fourier_boxes[0])
        for i in range(wav_mapset.shape[1]):
            back_transform += np.fft.fftshift(fourier_boxes[i]) * filters[i]
        back_transform = np.fft.fftshift(
            np.real(np.fft.ifftn(np.fft.fftshift(back_transform)))
        )
        sky_map.append(back_transform)
    tic = time.time()
    if print_time:
        print("wav2map took ", tic - toc)
    return np.array(sky_map)


def wav2map(
    wav_mapset: NDArray[np.floating],
    filters: NDArray[np.floating],
    n_filt: Optional[NDArray[np.floating]] = None,
    print_time: Optional[bool] = False,
) -> NDArray[np.floating]:
    # Reimplementation of wav2map_real using Jons ffts. Weirdly doesn't seem faster?
    toc = time.time()
    if len(wav_mapset.shape) == 3:
        wav_mapset = np.expand_dims(wav_mapset, axis=0)

    elif len(wav_mapset.shape) != 4:
        print("Error: input wave mapset must have dim = 3 or 4")
        return

    sky_map = []
    if n_filt is None:
        n_filt = range(len(wav_mapset))  # If not provided with filts to do, then do all
    for nu in n_filt:
        fourier_boxes = []
        for b in wav_mapset[nu]:
            b = np.hstack([b, np.fliplr(b[:, 1:-1])])
            fourier_boxes.append(rfftn(np.fft.fftshift(b)))
        back_transform = np.zeros_like(fourier_boxes[0])
        for i in range(wav_mapset.shape[1]):
            back_transform += np.fft.fftshift(fourier_boxes[i]) * filters[i]
        back_transform = np.fft.fftshift(irfftn(np.fft.fftshift(back_transform)))
        sky_map.append(back_transform)
    tic = time.time()
    if print_time:
        print("wav2map took ", tic - toc)
    return np.array(sky_map)
