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

from pixell import enmap

import matplotlib.pyplot as plt

import copy
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


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
    wmap : NDArray[np.floating]
        Array of wavelet coefficients. Can also be thought of as the wavelet space map.
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
    ):
        """
        Initialize the SkyMap.

        Parameters
        ----------
        filters: NDArray[np.floating]
            An array specifying the needlet filter response from needlet class
        See SkyMap docstring for remainer
        """
        self.filters = filters
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
        self.wmap = np.zeros([self.nfilt, self.nx, self.ny])

    def axpy(self, map, a):
        """
        Apply a*x + y to wavelet map.

        Parameters
        ----------
        map : WavSkyMap
             The map to act as y

        a : float
            Number to mutiply the map by.
        """
        self.wmap[:] = self.wmap[:] + a * map.wmap[:]

    def clear(self):
        """
        Zero out the wmap.
        """
        self.wmap[:] = 0
        self.map[:] = 0
    def clear_real_map(self):
        """
        Zero out the realspace map.
        """
        self.map[:] = 0

    def assign(self, arr: NDArray[np.floating]):
        """
        Assign new values to wmap.

        Parameters
        ----------
        arr : NDArray[np.floating]
            Array of values to assign to wmap.
            Should be the same shape as self.wmap.
        """
        assert arr.shape[0] == self.nfilt
        assert arr.shape[1] == self.nx
        assert arr.shape[2] == self.ny

        self.map[:] = arr

    def assign_real_map(self, arr: NDArray[np.floating]):
        """
        Assign new values to realspace map.

        Parameters
        ----------
        arr : NDArray[np.floating]
            Array of values to assign to map.
            Should be the same shape as self.map.
        """
        assert arr.shape[0] == self.nx
        assert arr.shape[1] == self.ny

        self.map[:] = arr

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
        self.map = np.squeeze(
            wav2map_real(self.wmap, self.filters), axis=0
        )  # Right now let's restrict ourself to 1 freqency input maps, so we need to squeeze down the dummy axis added by map2wav_real
        map2tod(dat, self.map, ipix, do_add, do_omp)

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

        self.map[:] = 0 #clear real map before conversion
        tod2map_simple(self.map, dat, ipix)
        self.wmap = np.squeeze(
            map2wav_real(self.map, self.filters), axis=0
        )  # Right now let's restrict ourself to 1 freqency input maps, so we need to squeeze down the dummy axis added by map2wav_real

        if self.purge_pixellization:
            tod.clear_saved_pix(self.tag)

    def dot(self, map: Self) -> float:
        """
        Take dot product of this wmap with another.

        Parameters
        ----------
        map : WavSkyMap
            Map to take dot product with.

        Returns
        -------
        tot : float
            The dot product
        """
        tot = np.sum(self.wmap * map.wmap)
        return tot

    def write(self, fname: str = "map.fits"):
        """
        Write map to a FITs file.

        Parameters
        ----------
        fname : str, default: 'map.fits'
            The path to save the map to.
        """
        self.map = np.squeeze(wav2map_real(self.wmap, self.filters), axis=0)
        header = self.wcs.to_header()
        if True:  # try a patch to fix the wcs xxx
            tmp = self.map.transpose().copy()
            hdu = fits.PrimaryHDU(tmp, header=header)
        else:
            hdu = fits.PrimaryHDU(self.map, header=header)
        try:
            hdu.writeto(fname, overwrite=True)
        except:
            hdu.writeto(fname, clobber=True)

    def __mul__(self, map: Self) -> Self:
        """
        Multiply this map with another.

        Parameters
        ----------
        map : WavSkyMap
            Map to multiply by.

        Returns
        -------
        new_map : WavSkylMap
            The result of the multiplication.
        """
        new_map = map.copy()
        new_map.wmap[:] = self.wmap[:] * map.wmap[:]
        return new_map

    def invert(self):
        """
        Invert the map. Note that the inverted map is stored in self.map.
        """
        mask = np.abs(self.wmap) > 0
        self.wmap[mask] = 1.0 / self.wmap[mask]

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
        """

        ####### init attributes
        self.js = js
        self.lightcone = lightcone
        self.kmax_dimless = kmax_dimless

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
        bl2 = np.vectorize(self.__b2_need) #b2_need is a function

        for j in self.js:
            xi = self.k_arr / self.B**j
            bl = np.sqrt(bl2(xi))
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

    def filter_lightcone(
        self, return_fourier=False, plot=False, plot_norm="lin", n_nu_plot=None
    ):
        filtered_slices_real = []
        filtered_slices_fourier = []
        fourier_radii = self.lightcone_box.get_grid_dimless_2d(return_grid=True)
        self.get_needlet_filters_2d(fourier_radii)

        for i in range(len(self.lightcone)):
            lightcone_ft = np.fft.fftn(np.fft.fftshift(self.lightcone[i]))

            filtered_slice_real = []
            filtered_slice_fourier = []
            for filt in self.filters:
                fourier_filtered = np.fft.fftshift(filt) * lightcone_ft
                filtered_slice_fourier.append(fourier_filtered)
                filtered_slice_real.append(
                    np.fft.fftshift(np.real(np.fft.ifftn(fourier_filtered)))
                )  # should maybe catch if imag(fourier_filtered)!=0 here

            filtered_slices_real.append(np.array(filtered_slice_real))
            filtered_slices_fourier.append(np.array(filtered_slice_fourier))

        if return_fourier:
            return np.array(filtered_slices_real), np.array(filtered_slices_fourier)
        else:
            return np.array(filtered_slices_real)

    def back_transform(self, filtered_boxes):
        back_transformed = []
        for nu in range(len(self.lightcone)):
            fourier_boxes = []
            for b in filtered_boxes[nu]:
                fourier_boxes.append(np.fft.fftn(np.fft.fftshift(b)))

            back_transform = np.zeros_like(fourier_boxes[0])
            for i in range(self.nfilt):
                back_transform += np.fft.fftshift(fourier_boxes[i]) * self.filters[i]
            back_transform = np.fft.fftshift(
                np.real(np.fft.ifftn(np.fft.fftshift(back_transform)))
            )
            back_transformed.append(back_transform)
        return np.array(back_transformed)

    def __f_need(self, t):
        """Auxiliar function f to define the standard needlet"""
        if t <= -1.0:
            return 0.0
        elif t >= 1.0:
            return 0.0
        else:
            return np.exp(1.0 / (t**2.0 - 1.0))

    def __psi(self, u):
        """Auxiliar function psi to define the standard needlet"""
        return (
            integrate.quad(self.__f_need, -1, u)[0]
            / integrate.quad(self.__f_need, -1, 1)[0]
        )

    def __phi(self, q):
        """Auxiliar function phi to define the standard needlet"""
        B = float(self.B)
        if q < 0.0:
            raise ValueError("The multipole should be a non-negative value")
        elif q <= 1.0 / B:
            return 1.0
        elif q >= 1.0:
            return 0
        else:
            return self.__psi(1.0 - (2.0 * B / (B - 1.0) * (q - 1.0 / B)))

    def __b2_need(self, xi):
        """Auxiliar function b^2 to define the standard needlet"""
        b2 = self.__phi(xi / self.B) - self.__phi(xi)
        return np.max([0.0, b2])
        ## np.max in order to avoid negative roots due to precision errors

    # ==============================================================================================#
    # ====================================== plotting functions ====================================#
    # ==============================================================================================#

    def plot_bands(self):
        fig, ax = plt.subplots()

        self.sum_sq = np.zeros_like(self.k_arr)
        for j, b in enumerate(self.bands):
            ax.plot(self.k_arr, b, label=f"j={j}")
            self.sum_sq += b**2

        ax.plot(self.k_arr, self.sum_sq, label=f"$\sum b^2$", color="k")
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
    filtered_slices_real = []

    npix = imaps.shape[-2] * imaps.shape[-1]
    weights = np.array([np.sum(f**2) / npix for f in filters])

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

    npix = wav_mapset.shape[-2] * wav_mapset.shape[-1]
    weights = np.array([np.sum(f**2) / npix for f in filters])

    back_transformed = []
    for nu in range(len(wav_mapset)):
        fourier_boxes = []
        for b in wav_mapset[nu]:
            fourier_boxes.append(np.fft.fftn(np.fft.fftshift(b)))

        npix_f = fourier_boxes[0].shape[-1] * fourier_boxes[0].shape[-2]

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
