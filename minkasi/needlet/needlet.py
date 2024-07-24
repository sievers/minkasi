import time

from numba import jit, prange

from astropy.io import fits
from astropy import wcs

import numpy as np
from numpy.typing import NDArray

from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    NamedTuple,
)

from scipy.linalg import norm
import scipy.integrate as integrate
from scipy.interpolate import interp1d

from ..maps.mapset import Mapset
from ..maps.utils import get_wcs
from ..mapmaking.map2tod import map2tod
from ..mapmaking.tod2map import (
    tod2map_cached,
    tod2map_everyone,
    tod2map_omp,
    tod2map_simple,
)
from ..parallel import comm, get_nthread, have_mpi, nproc, myrank, MPI
from ..tods import Tod, TodVec
from ..tools.fft import find_good_fft_lens, rfftn, irfftn
from ..maps.skymap import SkyMap

import matplotlib.pyplot as plt

import copy
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

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
        basisKwargs: dict = {},
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

        self.basisKwargs = basisKwargs

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
                ax[1, i].set_title(f"2D filter for j={i+n}")
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
            x_arr *= scale
        for j, b in enumerate(self.bands):
            ax.plot(x_arr, b, label=f"j={j}")
            self.sum_sq += b**2

        ax.plot(x_arr, self.sum_sq, label=f"$\sum b^2$", color="k")
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
        try:
            hdu.writeto(fname, overwrite=True)
        except:
            hdu.writeto(fname, clobber=True)

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

    def get_downsamps(self):
        """
        Resizes maps so that the pixelization matches the smallest scale of the needlet, given a needlet basis.
        That needlet basis should probably be the one associated with need.filt but for right now it doesn't need to be.
        I should enfore this. Currently not used, downsampling is handled directly in get_response_matrix.
        Keeping as this mostly works and may one day need it.

        """

        map_size = (
            np.rad2deg(max(self.lims[1] - self.lims[0], self.lims[3] - self.lims[2]))
            * 3600
        )
        downsamps = np.empty(self.needlet.nfilt, dtype=int)
        self.nxs_red = np.empty(self.needlet.nfilt, dtype=int)
        self.nys_red = np.empty(self.needlet.nfilt, dtype=int)
        self.nx_space = np.empty(self.needlet.nfilt, dtype=object)
        self.ny_space = np.empty(self.needlet.nfilt, dtype=object)
        for i in range(self.needlet.nfilt):
            lims = self.needlet.get_need_lims(i, real_space=True)
            n_samp = map_size / lims[0]
            downsamp = max(np.floor(self.nx / n_samp), 1)
            downsamps[i] = downsamp
            nx_red, ny_red = int(self.nx / downsamp), int(
                self.ny / downsamp
            )  # TODO: make this a good FFT number
            self.nxs_red[i] = nx_red
            self.nys_red[i] = ny_red

            # Evenly sample the downsampled nx
            x_step = int(self.nx / nx_red)
            y_step = int(self.ny / ny_red)
            nx_space = np.arange(int(x_step / 2), self.nx, x_step)
            ny_space = np.arange(int(y_step / 2), self.ny, y_step)

            self.nx_space[i] = np.array([int(n) for n in nx_space])
            self.ny_space[i] = np.array([int(n) for n in ny_space])

        self.downsamps = downsamps

    def check_response_matrix(
        self,
        filt_num: int,
        down_samp: Union[None, NDArray[int]] = None,
    ) -> Union[NDArray[np.floating], NamedTuple]:
        """
        Get the response matrix for needlet filt_num.
        Each entry of the response matrix is the map that results from passing a map
        that is 1 at uniquely one pixel thru map2wav with one needlet.
        The map is flattened for convenience.
        This function is used to check that we haven't over downsampled the
        needlet response matrix. Most common usage is to take the SVD of this
        and look at S, it should fall off a cliff in the last couple modes.

        Parameters
        ---------
        filt_num : int
            The needlet to compute the repsponse matrix for.
        down_samp : None | NDArray[int]
            The amount to downsample the input map by.
            You can downsample the map down to ~the needlet scale without loosing information
            If not specified then it computes and uses the most aggresive downsampling.
        Returns
        -------
        to_ret : NDArray[np.floating] | NamedTuple
            If do_svd is False, then returns the response matrix.
            Otherwise returns the SVD of the response matrix.
            See numpy documentation for SVD documentation.
        """
        if down_samp is None:
            self.get_downsamps()
            down_samp = self.downsamps[filt_num]
        else:
            down_samp = down_samp[filt_num]
        nxs_red, nys_red = int(self.nx / down_samp), int(
            self.ny / down_samp
        )  # TODO: make this a good FFT number
        to_ret = np.zeros((nxs_red * nys_red, self.nx * self.ny))

        nx_space = np.linspace(
            0, self.nx - 1, nxs_red
        )  # Evenly sample the downsampled nx
        ny_space = np.linspace(0, self.ny - 1, nys_red)

        nx_space = np.array([int(n) for n in nx_space])
        ny_space = np.array([int(n) for n in ny_space])
        if have_mpi:
            if myrank == 0:
                flags = np.zeros(
                    nproc - 1, dtype=bool
                )  # Flags to track which process are done
                while not np.all(flags):
                    status = MPI.Status()
                    temp = comm.recv(
                        source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
                    )
                    sender = status.Get_source()
                    tag = status.Get_tag()
                    if type(temp) == str:
                        print("Task ", sender, " is done")
                        flags[sender - 1] = temp
                    else:
                        to_ret[tag] = temp

            else:
                for nx in range(myrank - 1, nxs_red, nproc - 1):
                    for ny in range(nys_red):
                        idx = nys_red * nx + ny
                        temp = np.zeros((self.nx, self.ny))
                        temp[nx_space[nx], ny_space[ny]] = 1
                        temp = np.ravel(
                            np.squeeze(
                                map2wav_real(
                                    temp, self.needlet.filters[filt_num : filt_num + 1]
                                )
                            )
                        )
                        comm.send(temp, dest=0, tag=idx)
                comm.send("Done", dest=0, tag=0)

            comm.barrier()

        else:
            for nx in range(nxs_red):
                for ny in range(nys_red):
                    idx = nys_red * nx + ny
                    temp = np.zeros((self.nx, self.ny))
                    temp[nx_space[nx], ny_space[ny]] = 1
                    to_ret[idx, :] = np.ravel(
                        np.squeeze(
                            map2wav_real(
                                temp, self.needlet.filters[filt_num : filt_num + 1]
                            )
                        )
                    )

        return to_ret

    # @jit(parallel=True, forceobj=True, nopython=False)
    # @profile
    def get_response_matrix(
        self,
        todvec: "TodVec",
        max_res: Optional[np.floating] = None,
        nfilts: Optional[list[int]] = None,
    ):
        """
        Computes A^-1N^TA (ANA) for a set of tods in the WavSkyMap basis.
        ANA can be inverted to directly solve for the map, however is very
        expensive to compute. In general ANA only needs to be computed for
        the overlap scales, and other methods (e.g. pcg) can be used for
        the very high/very low resolution scales.

        Parameters:
        -----------
        todvec : TodVec
            TODs over which to comput ANA
        max_res : np.floating, default : None
            Maximum resolution, in arcseconds, at which to compute ANA.
            Wavelets with maximum extent less than max_res are skipped.
        nfilts : list[int], default : None
            Explicitly specify which wavelets to compute.
            Exclusive with max_res.

        Returns
        ------
        to_ret : NDArray[np.floating]
            Reduced ANA at the specified wavelet scales
        """
        if not self.downsamps:
            self.get_downsamps()
            down_samps = self.downsamps

        # We want to return 0 at all scales we don't compute
        to_ret = np.zeros(self.needlet.nfilt, dtype=object)

        if max_res is not None:
            if nfilts is not None:
                raise ArgError("Cannot specify max_res and nfilts")
            i = 0
            while self.needlet.get_need_lims(i, real_space=True)[0] > max_res:
                i += 1  # not the fastest way to do this but it's only done once
            nfilts = range(i + 1)

        for nfilt, filt in enumerate(self.needlet.filters):
            if nfilts is not None:
                if nfilt not in nfilts:
                    continue
            down_samp = down_samps[nfilt]
            # nxs_red = self.nxs_red[nfilt]
            # nys_red = self.nys_red[nfilt]
            nx_space = self.nx_space[nfilt]
            ny_space = self.ny_space[nfilt]
            nxs_red = len(nx_space)
            nys_red = len(ny_space)  # TODO: why isn't this already true?
            # to_ret_cur = np.empty((nxs_red * nys_red, self.needlet.nfilt, self.nx , self.ny), dtype=np.float32, order="C")
            to_ret_cur = np.empty(
                (nxs_red * nys_red, nxs_red * nys_red), dtype=np.float32, order="C"
            )

            real_map = self.real_map.copy()
            mapset = Mapset()
            mapset.add_map(real_map)
            for nx in prange(nxs_red):
                for ny in range(nys_red):
                    idx = nys_red * nx + ny

                    toc = time.time()
                    unit_impulse = np.zeros((self.needlet.nfilt, self.nx, self.ny))
                    unit_impulse[nfilt, nx_space[nx], ny_space[ny]] = (
                        1  # Make a map in wavelet space that is one at only one place
                    )
                    unit_impulse = unit_impulse[None,]  # Dummy axis

                    m_wav2map = wav2map_real(
                        unit_impulse, self.needlet.filters
                    )  # What is the mapspace response of that wavelet

                    mapset.maps[0].map[:] = m_wav2map

                    map_accum = todvec.dot(
                        mapset, cache_maps=False
                    )  # This gets the noise response of the tods to
                    # the realspace map generated by the unit impulse
                    # wavelet
                    temp = map2wav_real(map_accum.maps[0].map, self.needlet.filters)[
                        0, nfilt
                    ]

                    temp = np.array(
                        [temp[idx, idy] for idx in nx_space for idy in ny_space]
                    )

                    to_ret_cur[idx] = temp
                    tic = time.time()
                    print(nfilt, nx, ny, end="\r")
                    # print(tic-toc, end='\r')
                    # print("Whole will take ", (tic-toc)*np.sum(self.nxs_red**2))
            to_ret[nfilt] = to_ret_cur
        return to_ret

    def get_response_matrix_map(
        self,
        imap: NDArray[np.floating],
        max_res: Optional[np.floating] = None,
        nfilts: Optional[list[int]] = None,
    ):
        if not self.downsamps:
            self.get_downsamps()
            down_samps = self.downsamps

        # We want to return 0 at all scales we don't compute
        to_ret = np.zeros(self.needlet.nfilt, dtype=object)

        if max_res is not None:
            if nfilts is not None:
                raise ArgError("Cannot specify max_res and nfilts")
            i = 0
            while self.needlet.get_need_lims(i, real_space=True)[0] > max_res:
                i += 1  # not the fastest way to do this but it's only done once
            nfilts = range(i + 1)

        for nfilt, filt in enumerate(self.needlet.filters):
            if nfilts is not None:
                if nfilt not in nfilts:
                    continue
            down_samp = down_samps[nfilt]
            nx_space = self.nx_space[nfilt]
            ny_space = self.ny_space[nfilt]
            nxs_red = len(nx_space)
            nys_red = len(ny_space)  # TODO: why isn't this already true?
            to_ret_cur = np.empty(
                (nxs_red * nys_red, nxs_red * nys_red), dtype=np.float32, order="C"
            )

            real_map = self.real_map.copy()
            mapset = Mapset()
            mapset.add_map(real_map)
            for nx in prange(nxs_red):
                for ny in range(nys_red):
                    idx = nys_red * nx + ny
                    temp = map2wav_real(imap, self.needlet.filters)[0, nfilt]
                    temp = np.array(
                        [temp[idx, idy] for idx in nx_space for idy in ny_space]
                    )
                    to_ret_cur[idx] = temp
            to_ret[nfilt] = to_ret_cur
        return to_ret


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
