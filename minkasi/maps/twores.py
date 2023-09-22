import copy
import sys

import numpy as np
from astropy import wcs
from numpy.typing import NDArray

from ..mapmaking.noise import MapNoiseWhite
from .mapset import MapsetTwoRes
from .skymap import SkyMap, SkyMapCoarse
from .utils import get_aligned_map_subregion_car, get_ft_vec, read_fits_map

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class SkyMapTwoRes:
    """
    A pair of maps to serve as a prior for multi-experiment mapping.
    This would e.g. be the ACT map that e.g. Mustang should agree with on large scales.

    Attributes
    ----------
    small_lims : NDArray[np.floating]
        The limits of ra/dec (ra_low, ra_high, dec_low, dec_high) for the used subregion.
    small_wcs : wcs.WCS
        The WCS for the used subregion.
    map : NDArray[nf.floating]
        Array representing loaded low res map.
    osamp : int
        Factor to divide pixsize of larger WCS to get pixsize of subregion.
    map_corner : NDArray[np.integer]
        The corner pixel of the subregion.
    beamft : NDArray[np.complex128] | None
        The beam in fourier space.
    mask : NDArray[np.bool_] | None
        Mask computed from hits map.
    map_deconvolved : NDArray[np.floating] | None
        The deconvolved map.
    noise : MapNoiseWhite | None
        Noise object for the map.
    fine_prior : NDArray[np.floating] | None
        Prior on the fine map.
    nx_coarse : int | None
        Size of coarse map along the x axis.
    ny_coarse : int | None
        Size of coarse map along the y axis.
    grid facs : NDArray[np.floating] | None
        Average mask value for each coarse pixel.
    isglobal_prior : bool
       True if the prior to apply is global and takes in a mapset.
    smooth_fac : float
       Factor to smooth prior by.
       Not used when it is 0.
    """

    def __init__(
        self,
        map_lowres: str,
        lims: tuple[float, ...] | list[float] | NDArray[np.floating],
        osamp: int = 1,
        smooth_fac: float = 0.0,
    ):
        """
        Initialize SkyMapTwoRes.

        Parameters
        ----------
        map_lowres : str
            Path to the FITS file containing the low res map.
        lims : tuple[float, ...] | list[float] | NDArray[np.floating]
            The limits of ra/dec (ra_low, ra_high, dec_low, dec_high) for the requested subregion.
        int : float, default: 1
            Factor to divide pixsize of larger WCS to get pixsize of subregion.
        smooth_fac : float, default: 0
            Factor to smooth prior by.
            Not used when it is 0.
        """
        small_wcs, lims_use, map_corner = get_aligned_map_subregion_car(
            lims, map_lowres, osamp=osamp
        )
        self.small_lims: NDArray[np.floating] = lims_use
        self.small_wcs: wcs.WCS = small_wcs
        self.map: NDArray[np.floating] = read_fits_map(map_lowres)
        self.osamp: int = osamp
        self.map_corner: NDArray[np.integer] = map_corner
        self.beamft: NDArray[np.complex128] | None = None
        self.mask: NDArray[np.bool_] | None = None
        self.map_deconvolved: NDArray[np.floating] | None = None
        self.noise: MapNoiseWhite | None = None
        self.fine_prior: NDArray[np.floating] | None = None
        self.nx_coarse: int | None = None
        self.ny_coarse: int | None = None
        self.grid_facs: NDArray[np.floating] | None = None
        self.isglobal_prior: bool = True
        self.smooth_fac: float = smooth_fac

    def copy(self) -> Self:
        """
        Return a copy of this map
        """
        return copy.copy(self)

    def get_map_deconvolved(self, map_deconvolved: str):
        """
        Load the deconvolved map.

        Parameters
        ----------
        map_deconvolved : str
            Path to FITs file with map.
        """
        self.map_deconvolved = read_fits_map(map_deconvolved)

    def set_beam_gauss(self, fwhm_pix: float):
        """
        Set beamft with a gaussian beam.

        Parameters
        ---------
        fwhm_pix : float
           FWHM of beam in pixels.
        """
        tmp = 0 * self.map
        xvec = get_ft_vec(tmp.shape[0])
        yvec = get_ft_vec(tmp.shape[1])
        xx, yy = np.meshgrid(yvec, xvec)
        rsqr = xx**2 + yy**2
        sig_pix = fwhm_pix / np.sqrt(8 * np.log(2))
        beam = np.exp(-0.5 * rsqr / (sig_pix**2))
        beam = beam / np.sum(beam)
        self.beamft = np.fft.rfft2(beam)

    def set_beam_1d(self, prof: NDArray[np.floating], pixsize: float):
        """
        Set beamft with a 1d profile.

        Parameters
        ---------
        prof : NDArray[np.floating]
            Beam profile, should be a (2, n) array.
            First row is the radius values.
            Second row is the beam value at the given radius.
        pixsize : float
           Pixel size.
        """
        tmp = 0 * self.map
        xvec = get_ft_vec(tmp.shape[0])
        yvec = get_ft_vec(tmp.shape[1])
        xx, yy = np.meshgrid(yvec, xvec)
        rsqr = xx**2 + yy**2
        rr = np.sqrt(rsqr) * pixsize
        beam = np.interp(rr, prof[:, 0], prof[:, 1])
        beam = beam / np.sum(beam)
        self.beamft = np.fft.rfft2(beam)

    def beam_convolve(self, map: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Convolve with the stored beam.
        Requires beamft to be set.

        Parameters
        ----------
        map : NDArray[np.floating]
            Map to beam convolve.

        Returns
        -------
        convolved : NDArray[np.floating]
            The beam convolved map.

        Raises
        ------
        ValueError
            If beamft is not set.
        """
        if self.beamft is None:
            raise ValueError("beamft not set")
        mapft = np.fft.rfft2(map)
        mapft = mapft * self.beamft
        return np.fft.irfft2(mapft)

    def set_noise_white(
        self, ivar_map: NDArray[np.floating], isinv: bool = True, nfac: float = 1.0
    ):
        """
        Create a MapNoiseWhite object and set it as the noise.

        Parameters
        ----------
        ivar_map : NDArray[np.floating]
            Inverse variance map.
        isinv : bool, default: True
            Set to false if ivar_map is actually a variance map.
        nfac : float, default: 1.0
            Noise factor that is multiplied with the inverse variance.
        """
        self.noise = MapNoiseWhite(ivar_map, isinv, nfac)

    def maps2fine(
        self, fine: NDArray[np.floating], coarse: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Combine fine and coarse map to make one fine map.
        You need to have run set_mask before running this.

        Parameters
        ----------
        fine : NDArray[np.floating]
            Input fine map.
            Should be the same size as self.mask.

        coarse : NDArray[np.floating]
            Input coarse map.

        Returns
        -------
        out : NDArray[np.floating]
            Output fine map.
            Same shape as the input fine map.

        Raises
        ------
        ValueError
            If set_mask hasn't been run.
        """
        if self.mask is None or self.nx_coarse is None or self.ny_coarse is None:
            raise ValueError("Doesn't look like set_mask has been run")
        out = fine.copy()
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                out[
                    (i * self.osamp) : ((i + 1) * self.osamp),
                    (j * self.osamp) : ((j + 1) * self.osamp),
                ] = coarse[i + self.map_corner[0], j + self.map_corner[1]]
        out[self.mask] = fine[self.mask]
        return out

    def maps2coarse(
        self, fine: NDArray[np.floating], coarse: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Combine fine and course map to make one coarse map.
        You need to have run set_mask before running this.

        Parameters
        ----------
        fine : NDArray[np.floating]
            Input fine map.
            Should be the same size as self.mask.

        coarse : NDArray[np.floating]
            Input coarse map.

        Returns
        -------
        out : NDArray[np.floating]
            Output coarse map.
            Same shape as the input coarse map.

        Raises
        ------
        ValueError
            If set_mask hasn't been run.
        """
        if self.nx_coarse is None or self.ny_coarse is None or self.grid_facs is None:
            raise ValueError("Doesn't look like set_mask has been run")
        out = coarse.copy()
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                out[i + self.map_corner[0], j + self.map_corner[1]] = (
                    1 - self.grid_facs[i, j]
                ) * coarse[i + self.map_corner[0], j + self.map_corner[1]] + np.sum(
                    fine[
                        (i * self.osamp) : ((i + 1) * self.osamp),
                        (j * self.osamp) : ((j + 1) * self.osamp),
                    ]
                ) / self.osamp**2
        return out

    def coarse2maps(
        self, inmap: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Make a fine and a coarse map from a coarse map.
        You need to have run set_mask before running this.
        Parameters
        ----------
        inmap: NDArray[np.floating]
            Input coarse map.

        Returns
        -------
        coarse : NDArray[np.floating]
            Output coarse map.
            Same shape as the input coarse map.

        fine : NDArray[np.floating]
            Output fine map.
            Same shape as self.mask.

        Raises
        ------
        ValueError
            If set_mask hasn't been run.
        """
        if (
            self.mask is None
            or self.nx_coarse is None
            or self.ny_coarse is None
            or self.grid_facs is None
        ):
            raise ValueError("Doesn't look like set_mask has been run")
        coarse = 1.0 * inmap
        fine = np.zeros((self.nx_coarse * self.osamp, self.ny_coarse * self.osamp))
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                coarse[i + self.map_corner[0], j + self.map_corner[1]] = (
                    1 - self.grid_facs[i, j]
                ) * inmap[i + self.map_corner[0], j + self.map_corner[1]]
                fine[
                    (i * self.osamp) : ((i + 1) * self.osamp),
                    (j * self.osamp) : ((j + 1) * self.osamp),
                ] = (
                    inmap[i + self.map_corner[0], j + self.map_corner[1]]
                    / self.osamp**2
                )
        fine = fine * self.mask
        return coarse, fine

    def set_mask(self, hits, thresh=0):
        """
        Set mask from a hits map.
        Also computes and sets nx_coarse, ny_coarse, fine_prior, and grid_facs.
        If self.map_deconvolved is not set then fine_prior will not be set.

        Parameters
        ----------
        hits : NDArray[np.floating]
            The hits map.
        thresh : float, default: 0
            Hits threshold to mask at.
        """
        self.mask = hits > thresh
        make_prior = self.map_deconvolved is not None
        if make_prior:
            self.fine_prior = np.zeros_like(hits)
        self.nx_coarse = int(np.round(hits.shape[0] / self.osamp))
        self.ny_coarse = int(np.round(hits.shape[1] / self.osamp))
        self.grid_facs = np.zeros([self.nx_coarse, self.ny_coarse])
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                self.grid_facs[i, j] = np.mean(
                    self.mask[
                        (i * self.osamp) : ((i + 1) * self.osamp),
                        (j * self.osamp) : ((j + 1) * self.osamp),
                    ]
                )
        if self.map_deconvolved is not None:
            self.fine_prior = np.zeros_like(hits)
            for i in range(self.nx_coarse):
                for j in range(self.ny_coarse):
                    self.fine_prior[
                        (i * self.osamp) : ((i + 1) * self.osamp),
                        (j * self.osamp) : ((j + 1) * self.osamp),
                    ] = self.map_deconvolved[
                        self.map_corner[0] + i, self.map_corner[1] + j
                    ]

    def apply_Qinv(self, map: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Make a coarse map, beam convolve it, apply noise, and then beam convolve again.
        Then make an fine copy with a factor of 1/osamp**2 applied.
        set_mask needs to have been run and beamft and noise must have been set for this function to run.

        Parameters
        ----------
        map : NDArray[np.floating]
            Input map, will be used to make the coarse map where mask is True.
            fine_prior is used elsewhere.

        Returns
        -------
        ans : NDArray[np.floating]
            The output map, set to zero where mask is False.

        Raises
        ------
        ValueError
            If set_mask hasn't been run.
            Or is noise is not set.
        """
        if (
            self.mask is None
            or self.fine_prior is None
            or self.nx_coarse is None
            or self.ny_coarse is None
            or self.map_deconvolved is None
        ):
            raise ValueError("Doesn't look like set_mask has been run")
        if self.noise is None:
            raise ValueError("Noise not set")
        tmp = self.fine_prior.copy()
        tmp[self.mask] = map[self.mask]
        tmp2 = 0 * self.map_deconvolved.copy()
        for i in range(self.nx_coarse):
            for j in range(self.nx_coarse):
                tmp2[self.map_corner[0] + i, self.map_corner[1] + j] = np.mean(
                    tmp[
                        (i * self.osamp) : ((i + 1) * self.osamp),
                        (j * self.osamp) : ((j + 1) * self.osamp),
                    ]
                )
        tmp2_conv = self.beam_convolve(tmp2)
        tmp2_conv_filt = self.noise.apply_noise(tmp2_conv)
        tmp2_reconv = self.beam_convolve(tmp2_conv_filt)
        # tmp2_reconv=np.fft.irfft2(np.fft.rfft2(tmp2_conv)*self.beamft)
        # tmp2_reconv=tmp2.copy()
        fac = 1.0 / self.osamp**2
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                tmp[
                    (i * self.osamp) : ((i + 1) * self.osamp),
                    (j * self.osamp) : ((j + 1) * self.osamp),
                ] = (
                    fac * tmp2_reconv[i + self.map_corner[0], j + self.map_corner[1]]
                )
        ans = 0.0 * tmp
        ans[self.mask] = tmp[self.mask]
        return ans

    def apply_H(
        self, coarse: NDArray[np.floating], fine: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Make a coarse map from a coarse and fine map and then beam convolve.

        Parameters
        ----------
        fine : NDArray[np.floating]
            Input fine map.
            Should be the same size as self.mask.

        coarse : NDArray[np.floating]
            Input coarse map.

        Returns
        -------
        mm : NDArray[np.floating]
            Beam convolved coarse map.
        """
        mm = self.maps2coarse(coarse, fine)
        mm = self.beam_convolve(mm)
        return mm

    def apply_HT(
        self, mm: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Beam convolve a coarse map and then make a coarse and fine map from it.

        Parameters
        ----------
        mm : NDArray[np.floating]
            Beam convolved coarse map.

        Returns
        -------
        coarse : NDArray[np.floating]
            Output coarse map.
            Same shape as the input coarse map.

        fine : NDArray[np.floating]
            Output fine map.
            Same shape as self.mask.
        """
        mm = self.beam_convolve(mm)
        coarse, fine = self.coarse2maps(mm)
        return coarse, fine

    def get_rhs(self, mapset: MapsetTwoRes):
        """
        Solve for right hand side from mapset.
        I don't really understand what this function is doing.

        Parameters
        ----------
        mapset : MapsetTwoRes
            Mapset containing a fine and coarse map.

        Raises
        ------
        ValueError
            If noise hasn't been set.
            If both a fine and coarse map aren't in the mapset.
        """
        # if map is None:
        #    map=self.map
        # map_filt=self.noise.apply_noise(map)
        # map_filt_conv=np.fft.irfft2(np.fft.rfft2(map_filt)*self.beamft)
        # tmp=0.0*self.mask
        # fac=1.0/self.osamp**2
        # for i in range(self.nx_coarse):
        #    for j in range(self.ny_coarse):
        #        tmp[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)]=fac*map_filt_conv[i+self.map_corner[0],j+self.map_corner[1]]

        # ans=0*tmp
        # ans[self.mask]=tmp[self.mask]
        # return ans
        if self.noise is None:
            raise ValueError("Noise isn't set")

        coarse_ind = None
        fine_ind = None
        for i in range(mapset.nmap):
            if isinstance(mapset.maps[i], SkyMapCoarse):
                coarse_ind = i
            else:
                if isinstance(mapset.maps[i], SkyMap):
                    fine_ind = i
        if (coarse_ind is None) or (fine_ind is None):
            raise ValueError(
                "Errror in twolevel prior:  either fine or coarse skymap not found."
            )

        mm = self.noise.apply_noise(self.map)
        if True:
            coarse, fine = self.apply_HT(mm)
            mapset.maps[coarse_ind].map[:] = mapset.maps[coarse_ind].map[:] + coarse
            mapset.maps[fine_ind].map[:] = mapset.maps[fine_ind].map[:] + fine
        else:
            mm = self.beam_convolve(mm)
            coarse, fine = self.coarse2maps(mm)
            i1 = self.map_corner[0]
            i2 = i1 + self.nx_coarse
            j1 = self.map_corner[1]
            j2 = j1 + self.ny_coarse
            coarse[i1:i2, j1:j2] = coarse[i1:i2, j1:j2] * (1 - self.grid_facs)
            mapset.maps[coarse_ind].map[:] = mapset.maps[coarse_ind].map[:] + coarse
            mapset.maps[fine_ind].map[self.mask] = (
                mapset.maps[fine_ind].map[self.mask] + fine[self.mask] / self.osamp**2
            )

    def apply_prior(self, mapset, outmapset):
        """
        Apply priors to outmapset.
        I don't really understand what this function is doing.

        Parameters
        ----------
        mapset : MapsetTwoRes
            Mapset containing a fine and coarse map.
        outmapset : MapsetTwoRes
            Mpaset storing maps with priors applied.

        Raises
        ------
        ValueError
            If noise hasn't been set.
            If both a fine and coarse map aren't in the mapset.
        """
        if self.noise is None:
            raise ValueError("Noise isn't set")
        coarse_ind = None
        fine_ind = None
        for i in range(mapset.nmap):
            if isinstance(mapset.maps[i], SkyMapCoarse):
                coarse_ind = i
            else:
                if isinstance(mapset.maps[i], SkyMap):
                    fine_ind = i
        if (coarse_ind is None) or (fine_ind is None):
            raise ValueError(
                "Errror in twolevel prior:  either fine or coarse skymap not found."
            )
        if True:
            mm = self.apply_H(mapset.maps[fine_ind].map, mapset.maps[coarse_ind].map)
            mm_filt = self.noise.apply_noise(mm)
            coarse, fine = self.apply_HT(mm_filt)

        else:
            summed = self.maps2coarse(
                mapset.maps[fine_ind].map, mapset.maps[coarse_ind].map
            )
            summed = self.beam_convolve(summed)
            summed = self.noise.apply_noise(summed)
            summed = self.beam_convolve(summed)
            coarse, fine = self.coarse2maps(summed)

        outmapset.maps[fine_ind].map[self.mask] = (
            outmapset.maps[fine_ind].map[self.mask] + fine[self.mask]
        )
        outmapset.maps[coarse_ind].map[:] = outmapset.maps[coarse_ind].map[:] + coarse

        if self.smooth_fac > 0:
            summed = self.maps2coarse(
                mapset.maps[fine_ind].map, mapset.maps[coarse_ind].map
            )
            summed_smooth = self.beam_convolve(summed)
            delt = summed - summed_smooth
            delt_filt = self.noise.apply_noise(delt) * self.smooth_fac
            delt_filt = delt_filt - self.beam_convolve(delt_filt)
            coarse, fine = self.coarse2maps(delt_filt)
            outmapset.maps[fine_ind].map[self.mask] = (
                outmapset.maps[fine_ind].map[self.mask] + fine[self.mask]
            )
            outmapset.maps[coarse_ind].map[:] = (
                outmapset.maps[coarse_ind].map[:] + coarse
            )

    def __bust_apply_prior(self, map, outmap):
        outmap.map[:] = outmap.map[:] + self.apply_Qinv(map.map)
