from numba import jit

from astropy.io import fits
from astropy import wcs

import numpy as np
from numpy.typing import NDArray

from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple, Union

from scipy.linalg import norm
import scipy.integrate as integrate
from scipy.interpolate import interp1d

from ..maps.utils import get_wcs
from ..mapmaking.map2tod import map2tod
from ..mapmaking.tod2map import (
    tod2map_cached,
    tod2map_everyone,
    tod2map_omp,
    tod2map_simple,
)
from ..parallel import comm, get_nthread, have_mpi, nproc
from ..tods import Tod, TodVec
from ..tools.fft import find_good_fft_lens
from ..maps.skymap import SkyMap

#from pixell import enmap

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

def Standard(k_arr, B, j):

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
        return (
            integrate.quad(__f_need, -1, u)[0]
            / integrate.quad(__f_need, -1, 1)[0]
        )

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

def Mexican(k_arr, B, j, p = 1):
    
    def __phi(q, B, p = 1):
        B = float(B)
        if q < 0.0:
            raise ValueError("The multipole should be a non-negative value")
        if q <= 1.0 / B:
            return 1.0
        elif q >= 1.0
            return 0
        else
            return (q / B**j)**p * np.exp(-1/2*(q/B**j)**2)
   
    xi = k_arr / B**j 
    b2 = __phi(xi / B, B, p) - __phi(xi, B, p)
    return b2



################################################################
#                         Core Classes                         #
################################################################

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
        filters: NDArray[np.floating],
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
        isglobal_prior : bool = False,
    ):
        """
        Initialize the SkyMap.

        Parameters
        ----------
        filters: NDArray[np.floating]
            An array specifying the needlet filter response from needlet class
        isglobal_prior: bool = False
            Specifies if this map is a prior to be applied to other maps.
        See SkyMap docstring for remainder
        """
        self.filters = filters
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
        self.nfilt = len(self.filters)
        self.map = np.zeros([self.nfilt, self.nx, self.ny])
        self.real_map = SkyMap(lims, pixsize, proj, pad, square, multiple, primes, cosdec, nx, ny, mywcs, tag, purge_pixellization, ref_equ,)#This is very slightly inefficient as it redoes the ssquaring but is safer in avoiding getting a real_map and wave_map with different shapes
    
    def map2tod(
        self,
        tod: "Tod",
        dat: NDArray[np.floating],
        do_add: bool = True,
        do_omp: bool = True,
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
        """
        ipix = self.get_pix(tod)
        self.real_map.map = np.squeeze(
            wav2map_real(self.map, self.filters), axis=0
        )  # Right now let's restrict ourself to 1 freqency input maps, so we need to squeeze down the dummy axis added by map2wav_real
        map2tod(dat, self.real_map.map, ipix, do_add, do_omp)

    def tod2map(
        self,
        tod: "Tod",
        dat: NDArray[np.floating],
        do_add: bool = True,
        do_omp: bool = True,
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
                map2wav_real(self.real_map.map, self.filters), axis=0
            )  # Right now let's restrict ourself to 1 freqency input maps, so we need to squeeze down the dummy axis added by map2wav_real

        else:
            self.map += np.squeeze(
                map2wav_real(self.real_map.map, self.filters), axis=0
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
        self.real_map.map = np.squeeze(wav2map_real(self.map, self.filters), axis=0)
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

    def apply_prior(self, p, Ap):
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
        
        Outputs
        -------
        Modifies Ap in place by Q^-1 m
        """
        Ap.map=Ap.map+self.map*p.map

class needlet:
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
        Needlet scaling factor. See https://iopscience.iop.org/article/10.1088/0004-637X/723/1/1
    bands : NDArray[np.floating]
        Needlet bands in 1D.
    filters : NDArray[np.floating]
        Needlet bands in 2D.

    """

    def __init__(
        self,
        js: NDArray[int],
        L: np.floating,
        lightcone: NDArray[np.floating],
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
            Realspace side length of lightcone.
        lightcone : NDArray[np.floating]
            Real space map to make needlets for.
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

        self.basis = basis

        if self.lightcone is not None:
            self.lightcone_box = cosmo_box(lightcone, L)
            self.kmax_dimless = self.lightcone_box.kmax_dimless
        self.nfilt = len(self.js)
       
        self.k_arr = np.append(
            np.array([0]),
            np.logspace(0, np.log10(self.kmax_dimless), int(10 * self.kmax_dimless)),
        )
        if B is not None:
            self.B = B
        else:
            self.B = (
                self.k_arr[-1] ** (1 / self.js[-1]) * 1.01
            )  # Set needlet width to just cover k_arr
        self.bands = self.get_needlet_bands_1d()
        
    
    def get_needlet_bands_1d(self):
        """
        Get 1D needlet response given parameters
        """
        needs = []
        bl2 = np.vectorize(self.basis)

        for j in self.js:
            bl = np.sqrt(bl2(self.k_arr, self.B, j)) #This will need to be fixed for Sigurd's basis
            needs.append(bl)
        needs = np.squeeze(needs)
        needs[0][0] = 1  # Want the k=0 mode to get the map average
        return needs

    def get_needlet_filters_2d(self, fourier_radii, return_filt=False, plot=False):
        """
        Turns 1D needlet response into 2D response.
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



    # ==============================================================================================#
    # ====================================== plotting functions ====================================#
    # ==============================================================================================#

    def plot_bands(self, scale = None):
        fig, ax = plt.subplots()

        self.sum_sq = np.zeros_like(self.k_arr)
        x_arr = self.k_arr
        if scale: x_arr *= scale
        for j, b in enumerate(self.bands):
            ax.plot(x_arr, b, label=f"j={j}")
            self.sum_sq += b**2
        
        ax.plot(x_arr, self.sum_sq, label=f"$\sum b^2$", color="k")
        ax.set_xscale("log")
        ax.legend(loc="lower right", ncols=self.nfilt // 2)
        ax.set_xlabel("k [dimless]")
        ax.set_ylabel(r"$b_j$")
        plt.show()


###############################################################################################


class cosmo_box:
    def __init__(self, box, L):
        # initializing some attributes
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
        self.get_grid_dimless_2d()
        return np.max(self.grid_dimless)

    def get_grid_dimless_2d(self, return_grid=False):
        """
        Generates a fourier space dimensionless grid, finds
        radial distance of each pixel from origin.
        """

        self.indices = np.indices((self.N, self.N)) - self.origin
        self.grid_dimless = norm(
            self.indices, axis=0
        )  # dimensionless kspace radius of each pix

        if return_grid:
            return self.grid_dimless


def new_map2wav(imaps, filters):
    npix = imaps.shape[-2] * imaps.shape[-1]
    weights = np.array([np.sum(f**2) / npix for f in filters])

    fmap = np.fft.fft(
        imaps, norm="backward"
    )  # We're gonna do a hack-y thing where we force numpy to un normalize the ffts by calling norm=backwards going forwards and norm=forwards going backwards. We are imposing our own norm
    fmap_npix = 1
    for dim in np.shape(imaps):
        fmap_npix *= dim
    to_return = np.zeros(
        (imaps.shape[0], filters.shape[0], imaps.shape[-2], imaps.shape[-1])
    )
    for i in range(len(imaps)):
        for j in range(len(filters)):
            fsmall = fmap[i]
            fsmall *= filters[j] / (weights[i] ** 0.5 * fmap_npix)

            to_return[i][j] = np.fft.ifft(fsmall, norm="forward").real
    return to_return


def map2wav_real(imaps, filters):
    """
    Transform from a regular map to a multimap of wavelet coefficients. Adapted from Joelles code + enmap.wavelets

    Parameters
    ----------
    imap: np.array()
        input map
    basis:
        needlet basis

    Returns
    -------
    wmap: np.array
        multimap of wavelet coefficients
    """

    if len(imaps.shape) == 2:
        imaps = np.expand_dims(imaps, axis=0)

    elif len(imaps.shape) != 3:
        print("Error: input map must have dim = 2 or 3")
        return
    #print(imaps.shape)
    filtered_slices_real = []

    #npix = imaps.shape[-2] * imaps.shape[-1]
    #weights = np.array([np.sum(f**2) / npix for f in filters])

    for i in range(len(imaps)):
        lightcone_ft = np.fft.fftn(np.fft.fftshift(imaps[i]))

        filtered_slice_real = []
        for filt in filters:
            fourier_filtered = (
                np.fft.fftshift(filt) * lightcone_ft
            )  # / (weights[i]**2 * lightcone_ft.shape[-1] * lightcone_ft.shape[-2])
            filtered_slice_real.append(
                np.fft.fftshift(np.real(np.fft.ifftn(fourier_filtered)))
            )  # should maybe catch if imag(fourier_filtered)!=0 here

        filtered_slices_real.append(np.array(filtered_slice_real))

    return np.array(filtered_slices_real)


def wav2map_real(wav_mapset, filters):
    if len(wav_mapset.shape) == 3:
        wav_mapset = np.expand_dims(wav_mapset, axis=0)

    elif len(wav_mapset.shape) != 4:
        print("Error: input wave mapset must have dim = 3 or 4")
        return

    #npix = wav_mapset.shape[-2] * wav_mapset.shape[-1]
    #weights = np.array([np.sum(f**2) / npix for f in filters])

    back_transformed = []
    for nu in range(len(wav_mapset)):
        fourier_boxes = []
        for b in wav_mapset[nu]:
            fourier_boxes.append(np.fft.fftn(np.fft.fftshift(b)))
        #npix_f = fourier_boxes[0].shape[-1] * fourier_boxes[0].shape[-2]

        back_transform = np.zeros_like(fourier_boxes[0])
        for i in range(wav_mapset.shape[1]):
            back_transform += (
                np.fft.fftshift(fourier_boxes[i]) * filters[i]
            )  # * (weights[i]**2 / npix_f)
        back_transform = np.fft.fftshift(
            np.real(np.fft.ifftn(np.fft.fftshift(back_transform)))
        )
        back_transformed.append(back_transform)
    return np.array(back_transformed)
