import copy
import sys
from astropy import wcs
from astropy.io import fits
import numpy as np
from numpy.typing import NDArray
from .utils import get_wcs
from ..parallel import have_mpi, comm, get_nthread
from ..utils import find_good_fft_lens
from ..tod2map import tod2polmap, tod2map_cached, tod2map_omp, tod2map_simple
from ..map2tod import polmap2tod, map2tod
from ..tods import Tod

try:
    have_healpy = True
    import healpy
except ImportError:
    have_healpy = False

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


def _poltag2pols(poltag):
    if poltag == "I":
        return ["I"]
    if poltag == "IQU":
        return ["I", "Q", "U"]
    if poltag == "QU":
        return ["Q", "U"]
    if poltag == "IQU_PRECON":
        return ["I", "Q", "U", "QQ", "QU", "UU"]
    if poltag == "QU_PRECON":
        return ["QQ", "UU", "QU"]

    return None


class PolMap:
    """
    Class to store and operate on a polarization map.

    Attributes
    ----------
    wcs : wcs.WCS
        The WCS for this map.
    primes : list[int]
        Prime numbers to use when calculating good fft lengths.
    nx : int
        The number of pixels in the x-axis.
    ny : int
        The number of pixels in the y-axis.
    npol : int
        The number of polarization states.
    poltag : str
        String that lists the polarization states.
    pols : list[str]
        List of polarizations in this map.
    lims : tuple[float, ...] | list[float] | NDArray[np.floating]
        The limits of ra/dec (ra_low, ra_high, dec_low, dec_high).
    tag : str
        Name to store pixellization under.
    purge_pixellization : bool
        Clear the pixellization when calling tod2map.
    pixsize : float
        The size of a pixel in radians.
    map : NDArray[np.floating]
        The actual map.
        Shape is (nx, ny, npol) unless npol is 1 in which case it is (nx, ny).
    proj : str
        The projection of the map.
    pad : int
        Amount of pixels to pad the map outside of lims.
    caches : NDArray[np.floating] | None
        Cached maps for tod2map_cached.
    cosdec : float
        The cosine declination correction.
    """

    def __init__(
        self,
        lims: tuple[float, ...] | list[float] | NDArray[np.floating],
        pixsize: float,
        poltag: str = "I",
        proj: str = "CAR",
        pad: int = 2,
        primes: list[int] | None = None,
        cosdec: float | None = None,
        nx: int | None = None,
        ny: int | None = None,
        mywcs: wcs.WCS | None = None,
        tag: str = "ipix",
        purge_pixellization: bool = False,
        ref_equ: bool = False,
    ):
        """
        Initialize the PolMap.

        Parameters
        ----------
        lims : tuple[float, ...] | list[float] | NDArray[np.floating]
            The limits of ra/dec (ra_low, ra_high, dec_low, dec_high).
        pixsize : float
            The size of a pixel in radians.
        poltag : str, default: 'I'
            String that lists the polarization states.
        proj : str, default: 'CAR'
            The projection of the map.
        pad : int, default: 2
            Amount of pixels to pad the map outside of lims.
        primes : list[int] | None, default: None
            Prime numbers to use when calculating good fft lengths (modifies nx and ny).
            If None this is not performed.
        cosdec : float | None, default: None
            The cosine declination correction.
            Set to None to have this function calculate it.
        nx : int | None, default: None
            The number of pixels in the x-axis.
            Set to None to have this function calculate it.
        ny : int | None, default: None
            The number of pixels in the y-axis.
            Set to None to have this function calculate it.
        mywcs : wcs.WCS | None, default: None
            WCS for the map.
            If None then lims, pixsize, proj, cosdec, and ref_equ are used to make one.
        tag : str, default: 'ipix'
            Name to store pixellization under.
        purge_pixellization : bool, default: False
            Clear the pixellization when calling tod2map.
        ref_equ : bool, default: False
            Use equtorial reference.
        """
        pols: list[str] | None = _poltag2pols(poltag)
        if pols is None:
            print("Unrecognized polarization state " + poltag + " in PolMap.__init__")
            return
        npol: int = len(pols)
        self.wcs: wcs.WCS
        if mywcs is None:
            self.wcs = get_wcs(lims, pixsize, proj, cosdec, ref_equ)
        else:
            self.wcs = mywcs
        corners: NDArray[np.floating] = np.zeros([4, 2])
        corners[0, :] = [lims[0], lims[2]]
        corners[1, :] = [lims[0], lims[3]]
        corners[2, :] = [lims[1], lims[2]]
        corners[3, :] = [lims[1], lims[3]]
        pix_corners = self.wcs.wcs_world2pix(corners * 180 / np.pi, 1)
        pix_corners = np.round(pix_corners)
        if pix_corners.min() < -0.5:
            print(
                "corners seem to have gone negative in SkyMap projection.  not good, you may want to check this."
            )
        if True:  # try a patch to fix the wcs xxx
            if nx is None:
                nx = int(pix_corners[:, 0].max() + pad)
            self.nx: int = nx
            if ny is None:
                ny = int(pix_corners[:, 1].max() + pad)
            self.ny: int = ny
        else:
            self.nx = int(pix_corners[:, 0].max() + pad)
            self.ny = int(pix_corners[:, 1].max() + pad)
        self.primes: None | list[int]
        if primes is None:
            self.primes = primes
        else:
            lens = find_good_fft_lens(2 * (self.nx + self.ny), primes)
            self.nx = lens[lens >= nx].min()
            self.ny = lens[lens >= ny].min()
            self.primes = primes[:]
        self.npol: int = npol
        self.poltag: str = poltag
        self.pols: list[str] = pols
        self.lims: tuple[float, ...] | list[float] | NDArray[np.floating] = lims
        self.tag: str = tag
        self.purge_pixellization: bool = purge_pixellization
        self.pixsize: float = pixsize
        self.map: NDArray[np.floating]
        if npol > 1:
            self.map = np.zeros([self.nx, self.ny, self.npol])
        else:
            self.map = np.zeros([self.nx, self.ny])
        self.proj: str = proj
        self.pad: int = pad
        self.caches: NDArray[np.floating] | None = None
        self.cosdec: float | None = cosdec

    def get_caches(self):
        """
        Setup caches for this map.
        """
        npix: int = self.nx * self.ny * self.npol
        nthread: int = get_nthread()
        self.caches = np.zeros([nthread, npix])

    def clear_caches(self):
        """
        Set map value to cache and clear caches for this map.
        If cache is None does nothing.
        """
        if self.caches is None:
            return
        self.map[:] = np.reshape(np.sum(self.caches, axis=0), self.map.shape)
        self.caches = None

    def copy(self) -> Self:
        """
        Return a copy of this map
        """
        if False:
            newmap = PolMap(
                self.lims,
                self.pixsize,
                self.poltag,
                self.proj,
                self.pad,
                self.primes,
                cosdec=self.cosdec,
                nx=self.nx,
                ny=self.ny,
                mywcs=self.wcs,
            )
            newmap.map[:] = self.map[:]
            return newmap
        else:
            return copy.deepcopy(self)

    def clear(self):
        """
        Zero out the map.
        """
        self.map[:] = 0

    def axpy(self, map: Self, a: float):
        """
        Apply a*x + y to map.

        Parameters
        ----------
        map : PolMap
            The map to act as y.

        a : float
            Number to multiply the map  by.
        """
        self.map[:] = self.map[:] + a * map.map[:]

    def assign(self, arr: NDArray[np.floating]):
        """
        Assign new values to map.

        Parameters
        ----------
        arr : NDArray[np.floating]
            Array of values to assign to map.
            Should be the same shape as self.map.
        """
        assert arr.shape[0] == self.nx
        assert arr.shape[1] == self.ny
        if self.npol > 1:
            assert arr.shape[2] == self.npol
        self.map[:] = arr

    def set_polstate(self, poltag: str):
        """
        Set a new polarization state.
        Updates self.poltag, self.npol, and self.pols.
        Also reinitialize self.map.

        Parameters
        ----------
        poltag : str
            The new poltag.
            If it is an invalid tag no change is applied.
        """
        pols = _poltag2pols(poltag)
        if pols is None:
            print(
                "Unrecognized polarization state " + poltag + " in PolMap.set_polstate."
            )
            return
        npol = len(pols)
        self.npol = npol
        self.poltag = poltag
        self.pols = pols
        if npol > 1:
            self.map = np.zeros([self.nx, self.ny, npol])
        else:
            self.map = np.zeros([self.nx, self.ny])

    def invert(self, thresh: float = 1e-6):
        """
        Invert the map. Note that the inverted map is stored in self.map.

        We can use np.linalg.pinv to reasonably efficiently invert a bunch of tiny matrices with an
        eigenvalue cut.  It's more efficient to do this in C, but it only has to be done once per run

        Parameters
        ----------
        thresh : float, default:1e-6
            Cutoff for small singular values.
        """
        if self.npol > 1:
            if self.poltag == "QU_PRECON":
                tmp = np.zeros([self.nx * self.ny, 2, 2])
                tmp[:, 0, 0] = np.ravel(self.map[:, :, 0])
                tmp[:, 1, 1] = np.ravel(self.map[:, :, 1])
                tmp[:, 0, 1] = np.ravel(self.map[:, :, 2])
                tmp[:, 1, 0] = np.ravel(self.map[:, :, 2])
                tmp = np.linalg.pinv(tmp, thresh)
                self.map[:, :, 0] = np.reshape(
                    tmp[:, 0, 0], [self.map.shape[0], self.map.shape[1]]
                )
                self.map[:, :, 1] = np.reshape(
                    tmp[:, 1, 1], [self.map.shape[0], self.map.shape[1]]
                )
                self.map[:, :, 2] = np.reshape(
                    tmp[:, 0, 1], [self.map.shape[0], self.map.shape[1]]
                )
            if self.poltag == "IQU_PRECON":
                # the mapping may seem a bit abstruse here.  The preconditioner matrix has entries
                #  [I   Q   U ]
                #  [Q  QQ  QU ]
                #  [U  QU  UU ]
                # so the unpacking needs to match the ordering in the C file before we can use pinv
                n = self.nx * self.ny
                nx = self.nx
                ny = self.ny
                tmp = np.zeros([self.nx * self.ny, 3, 3])
                tmp[:, 0, 0] = np.reshape(self.map[:, :, 0], n)
                tmp[:, 0, 1] = np.reshape(self.map[:, :, 1], n)
                tmp[:, 1, 0] = tmp[:, 0, 1]
                tmp[:, 0, 2] = np.reshape(self.map[:, :, 2], n)
                tmp[:, 2, 0] = tmp[:, 0, 2]
                tmp[:, 1, 1] = np.reshape(self.map[:, :, 3], n)
                tmp[:, 1, 2] = np.reshape(self.map[:, :, 4], n)
                tmp[:, 2, 1] = tmp[:, 1, 2]
                tmp[:, 2, 2] = np.reshape(self.map[:, :, 5], n)
                alldets = np.linalg.det(tmp)
                isbad = alldets < thresh * alldets.max()
                ispos = tmp[:, 0, 0] > 0
                inds = isbad & ispos
                vec = tmp[inds, 0, 0]
                print(
                    "determinant range is "
                    + repr(alldets.max())
                    + "  "
                    + repr(alldets.min())
                )
                tmp = np.linalg.pinv(tmp, thresh)
                if True:
                    print(
                        "Warning!  zeroing out bits like this is super janky.  Be warned..."
                    )
                    tmp[isbad, :, :] = 0
                    inds = isbad & ispos
                    tmp[inds, 0, 0] = 1.0 / vec
                alldets = np.linalg.det(tmp)
                print(
                    "determinant range is now "
                    + repr(alldets.max())
                    + "  "
                    + repr(alldets.min())
                )

                self.map[:, :, 0] = np.reshape(tmp[:, 0, 0], [nx, ny])
                self.map[:, :, 1] = np.reshape(tmp[:, 0, 1], [nx, ny])
                self.map[:, :, 2] = np.reshape(tmp[:, 0, 2], [nx, ny])
                self.map[:, :, 3] = np.reshape(tmp[:, 1, 1], [nx, ny])
                self.map[:, :, 4] = np.reshape(tmp[:, 1, 2], [nx, ny])
                self.map[:, :, 5] = np.reshape(tmp[:, 2, 2], [nx, ny])

        else:
            mask = self.map != 0
            self.map[mask] = 1.0 / self.map[mask]

    def pix_from_radec(
        self, ra: NDArray[np.floating], dec: NDArray[np.floating]
    ) -> NDArray[np.int32]:
        """
        Get pixellization from ra/dec.

        Parameters
        ----------
        ra : NDArray[np.floating]
            The RA TOD in radians.
            Should have shape (ndet, nsamp)
        dec : NDArray[np.floating]
            The dec TOD in radians.
            Should have shape (ndet, nsamp)

        Returns
        -------
        ipix : NDArray[np.int32]
            The pixellization.
        """
        ndet: int = ra.shape[0]
        nsamp: int = ra.shape[1]
        nn = ndet * nsamp
        coords: NDArray[np.floating] = np.zeros([nn, 2])
        coords[:, 0] = np.reshape(ra * 180 / np.pi, nn)
        coords[:, 1] = np.reshape(dec * 180 / np.pi, nn)
        pix: NDArray[np.floating] = np.asarray(self.wcs.wcs_world2pix(coords, 1))
        # -1 is to go between unit offset in FITS and zero offset in python
        xpix = np.round(np.reshape(pix[:, 0], [ndet, nsamp]) - 1)
        ypix = np.round(np.reshape(pix[:, 1], [ndet, nsamp]) - 1)
        ipix: NDArray[np.int32] = np.asarray(xpix * self.ny + ypix, dtype="int32")
        return ipix

    def get_pix(self, tod: Tod, savepix: bool = True) -> NDArray[np.int32]:
        """
        Get pixellization of a TOD.

        Parameters
        ----------
        tod : Tod
            The TOD object to pixelize.
        savepix : bool, default: True
            Save the pixelixation in the Tod at tod.info[self.tag].

        Returns
        -------
        ipix : NDArray[np.int32]
            The pixellization.
        """
        if not (self.tag is None):
            ipix = tod.get_saved_pix(self.tag)
            if not (ipix is None):
                return ipix
        if False:
            ndet = tod.info["dx"].shape[0]
            nsamp = tod.info["dx"].shape[1]
            nn = ndet * nsamp
            coords = np.zeros([nn, 2])
            coords[:, 0] = np.reshape(tod.info["dx"] * 180 / np.pi, nn)
            coords[:, 1] = np.reshape(tod.info["dy"] * 180 / np.pi, nn)
            # print coords.shape
            pix = self.wcs.wcs_world2pix(coords, 1)
            # print pix.shape
            xpix = (
                np.reshape(pix[:, 0], [ndet, nsamp]) - 1
            )  # -1 is to go between unit offset in FITS and zero offset in python
            ypix = np.reshape(pix[:, 1], [ndet, nsamp]) - 1
            xpix = np.round(xpix)
            ypix = np.round(ypix)
            ipix = np.asarray(xpix * self.ny + ypix, dtype="int32")
        else:
            ra, dec = tod.get_radec()
            ipix = self.pix_from_radec(ra, dec)
        if savepix:
            if not (self.tag is None):
                tod.save_pixellization(self.tag, ipix)
        return ipix

    def map2tod(
        self,
        tod: Tod,
        dat: NDArray[np.floating],
        do_add: bool = True,
        do_omp: bool = True,
    ):
        """
        Project a map into a tod, adding or replacing the map contents.

        Parameters
        ----------
        tod : Tod
            Tod object, used to get pixellization.
            If npol > 1 then this is assumed to have tod.info['twogamma_saved']
        dat : NDArray[np.floating]
            Array to put tod data into.
            Shape should be (ndet, nsamps).
        do_add : bool, default: True
            If True add the projected map to dat.
            If False replace dat with it.
        do_omp : bool, default: False
            Use OMP to parallelize.
        """
        ipix = self.get_pix(tod)
        if self.npol > 1:
            # polmap2tod(dat,self.map,self.poltag,tod.info['twogamma_saved'],tod.info['ipix'],do_add,do_omp)
            polmap2tod(
                dat,
                self.map,
                self.poltag,
                tod.info["twogamma_saved"],
                ipix,
                do_add,
                do_omp,
            )
        else:
            # map2tod(dat,self.map,tod.info['ipix'],do_add,do_omp)
            map2tod(dat, self.map, ipix, do_add, do_omp)

    def tod2map(
        self,
        tod: Tod,
        dat: NDArray[np.floating],
        do_add: bool = True,
        do_omp: bool = True,
    ):
        """
        Project a tod into this map.

        Parameters
        ----------
        tod : Tod
            Tod object, used to get pixellization.
            If npol > 1 then this is assumed to have tod.info['twogamma_saved']
        dat : NDArray[np.floating]
            Array to pull tod data from.
            Shape should be (ndet, nsamps).
        do_add : bool, default: True.
            If True add the projected map to this map.
            If False replace this map with it.
        do_omp : bool, default: False
            Use OMP to parallelize.
        """
        if do_add == False:
            self.clear()
        ipix = self.get_pix(tod)
        if self.npol > 1:
            tod2polmap(self.map, dat, self.poltag, tod.info["twogamma_saved"], ipix)
            if self.purge_pixellization:
                tod.clear_saved_pix(self.tag)
            return

        if not (self.caches is None):
            tod2map_cached(self.caches, dat, ipix)
        else:
            if do_omp:
                tod2map_omp(self.map, dat, ipix)
            else:
                tod2map_simple(self.map, dat, ipix)
        if self.purge_pixellization:
            tod.clear_saved_pix(self.tag)

    def r_th_maps(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Get polor coordinates for this map.
        Origin is at center of map.

        Returns
        -------
        r : NDArray[np.floating]
            Grid of radius values
        th : NDArray[np.floating]
            Grid of theta values
        """
        xvec = np.arange(self.nx)
        xvec = xvec - xvec.mean()
        yvec = np.arange(self.ny)
        yvec = yvec - yvec.mean()
        ymat, xmat = np.meshgrid(yvec, xvec)
        rmat = np.sqrt(xmat**2 + ymat**2)
        th = np.arctan2(xmat, ymat)
        return rmat, th

    def dot(self, map: Self) -> float:
        """
        Take dot product of this map with another.

        Parameters
        ----------
        map : PolMap
            Map to tahe dot product with.

        Returns
        -------
        tot : float
            The dot product
        """
        tot = np.sum(self.map * map.map)
        return tot

    def write(self, fname: str = "map.fits"):
        """
        Write map to a FITs file.

        Parameters
        ----------
        fname : str, default: 'map.fits'
            The path to save the map to.
        """
        header = self.wcs.to_header()

        if self.npol > 1:
            ind = fname.rfind(".")
            if ind > 0:
                if fname[ind + 1 :] == "fits":
                    head = fname[:ind]
                    tail = fname[ind:]
                else:
                    head = fname
                    tail = ".fits"
            else:
                head = fname
                tail = ".fits"
            tmp = np.zeros([self.ny, self.nx])
            for i in range(self.npol):
                tmp[:] = np.squeeze(self.map[:, :, i]).T
                hdu = fits.PrimaryHDU(tmp, header=header)
                try:
                    hdu.writeto(head + "_" + self.pols[i] + tail, overwrite=True)
                except:
                    hdu.writeto(head + "_" + self.pols[i] + tail, clobber=True)
            return

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
        map : PolMap
            Map to multiply by.

        Returns
        -------
        new_map : PolMap
            The result of the multiplication.

        Raises
        ------
        ValueError
            If  map.poltag+'_PRECON' != self.poltag
            or self.poltag does not contain PRECON.
        """

        if self.npol == 1:
            new_map = self.copy()
            new_map.map[:] = self.map[:] * map.map[:]
            return new_map
        else:
            if map.poltag + "_PRECON" != self.poltag:
                raise ValueError("Tag from input map does not match this map")
            new_map = map.copy()
            if self.poltag == "QU_PRECON":
                new_map.map[:, :, 0] = (
                    self.map[:, :, 0] * map.map[:, :, 0]
                    + self.map[:, :, 2] * map.map[:, :, 1]
                )
                new_map.map[:, :, 1] = (
                    self.map[:, :, 2] * map.map[:, :, 0]
                    + self.map[:, :, 1] * map.map[:, :, 1]
                )
                return new_map
            if self.poltag == "IQU_PRECON":
                # the indices are set such that the preconditioner matrix [I Q U; Q QQ QU; U QU UU] match the C code.
                # once we've inverted, the output should be the product of that matrix times [I Q U]
                new_map.map[:, :, 0] = (
                    self.map[:, :, 0] * map.map[:, :, 0]
                    + self.map[:, :, 1] * map.map[:, :, 1]
                    + self.map[:, :, 2] * map.map[:, :, 2]
                )
                new_map.map[:, :, 1] = (
                    self.map[:, :, 1] * map.map[:, :, 0]
                    + self.map[:, :, 3] * map.map[:, :, 1]
                    + self.map[:, :, 4] * map.map[:, :, 2]
                )
                new_map.map[:, :, 2] = (
                    self.map[:, :, 2] * map.map[:, :, 0]
                    + self.map[:, :, 4] * map.map[:, :, 1]
                    + self.map[:, :, 5] * map.map[:, :, 2]
                )
                return new_map
            raise ValueError(
                "unrecognized tag in PolMap.__mul__:  " + repr(self.poltag)
            )

    def mpi_reduce(self, chunksize: float = 1e5):
        """
        Reduces all maps in an MPI environment.

        Parameters
        ----------
        chunksize : float, default: 1e5
            Maximum samples to reduce at once.
        """
        # chunksize is added since at least on my laptop mpi4py barfs if it
        # tries to reduce an nside=512 healpix map, so need to break it into pieces.
        if not have_mpi:
            return
        if chunksize > 0:
            nchunk = (1.0 * self.nx * self.ny * self.npol) / chunksize
            nchunk = int(np.ceil(nchunk))
        else:
            nchunk = 1
        if nchunk == 1:
            self.map = comm.allreduce(self.map)
        else:
            inds = np.asarray(
                np.linspace(0, self.nx * self.ny * self.npol, nchunk + 1), dtype="int"
            )
            if len(self.map.shape) > 1:
                tmp = np.zeros(self.map.size)
                tmp[:] = np.reshape(self.map, len(tmp))
            else:
                tmp = self.map

            for i in range(len(inds) - 1):
                tmp[inds[i] : inds[i + 1]] = comm.allreduce(tmp[inds[i] : inds[i + 1]])
                # self.map[inds[i]:inds[i+1]]=comm.allreduce(self.map[inds[i]:inds[i+1]])
                # tmp=np.zeros(inds[i+1]-inds[i])
                # tmp[:]=self.map[inds[i]:inds[i+1]]
                # tmp=comm.allreduce(tmp)
                # self.map[inds[i]:inds[i+1]]=tmp
            if len(self.map.shape) > 1:
                self.map[:] = np.reshape(tmp, self.map.shape)


class HealPolMap(PolMap):
    """
    Subclass of PolMap with healpix pixels.
    Attributes list below are only the ones initialized by this subclass.
    See PolMap for the full list of attributes including the ones unused by this subclass.

    Attributes
    ----------
    nside : int
        The nside of the healpix map.
    nx : int
        The npix of the healpix map.
    ny : int
        Always 1, here for compatipility.
    npol : int
        The number of polarization states.
    poltag : str
        String that lists the polarization states.
    pols : list[str]
        List of polarizations in this map.
    tag : str
        Name to store pixellization under.
    purge_pixellization : bool
        Clear the pixellization when calling tod2map.
    map : NDArray[np.floating]
        The actual map.
        Shape is (nx, ny, npol) unless npol is 1 in which case it is (nx, ny).
    proj : str
        The projection of the map.
    caches : NDArray[np.floating] | None
        Cached maps for tod2map_cached.
    """

    def __init__(
        self,
        poltag: str = "I",
        proj: str = "RING",
        nside: int = 512,
        tag: str = "ipix",
        purge_pixellization: bool = False,
    ):
        """
        Initialize the HealPolMap.

        Parameters
        ----------
        poltag : str, default: 'I'
            String that lists the polarization states.
        proj : str, default: 'RING'
            The projection of the map.
        nside : int, default: 512
            The nside of the healpix map.
        tag : str, default: 'ipix'
            Name to store pixellization under.
        purge_pixellization : bool
            Clear the pixellization when calling tod2map.
        """
        if not (have_healpy):
            print("Healpix map requested, but healpy not found.")
            return
        pols: list[str] | None = _poltag2pols(poltag)
        if pols is None:
            print("Unrecognized polarization state " + poltag + " in PolMap.__init__")
            return
        self.npol: int = len(pols)
        self.proj: str = proj
        self.nside: int = nside
        self.nx: int = healpy.nside2npix(self.nside)
        self.ny: int = 1
        self.poltag: str = poltag
        self.pols: list[str] = pols
        self.caches: NDArray[np.floating] | None = None
        self.tag: str = tag
        self.purge_pixellization: bool = purge_pixellization
        self.map: NDArray[np.floating]
        if self.npol > 1:
            self.map = np.zeros([self.nx, self.ny, self.npol])
        else:
            self.map = np.zeros([self.nx, self.ny])

    def copy(self) -> Self:
        """
        Return a copy of this map
        """
        if False:
            newmap = HealPolMap(self.poltag, self.proj, self.nside, self.tag)
            newmap.map[:] = self.map[:]
            return newmap
        else:
            return copy.deepcopy(self)

    def pix_from_radec(
        self, ra: NDArray[np.floating], dec: NDArray[np.floating]
    ) -> NDArray[np.int32]:
        """
        Get pixellization from ra/dec.

        Parameters
        ----------
        ra : NDArray[np.floating]
            The RA TOD in radians.
            Should have shape (ndet, nsamp)
        dec : NDArray[np.floating]
            The dec TOD in radians.
            Should have shape (ndet, nsamp)

        Returns
        -------
        ipix : NDArray[np.int32]
            The pixellization.
        """
        ipix = healpy.ang2pix(self.nside, np.pi / 2 - dec, ra, self.proj == "NEST")
        return np.asarray(ipix, dtype="int32")

    # def get_pix(self,tod,savepix=True):
    #    if not(self.tag is None):
    #        ipix=tod.get_saved_pix(self.tag)
    #        if not(ipix is None):
    #            return ipix
    #    ra,dec=tod.get_radec()
    #    ipix=self.pix_from_radec(ra,dec)
    #    if savepix:
    #        if not(self.tag is None):
    #            tod.save_pixellization(self.tag,ipix)
    #    return ipix

    def write(self, fname: str = "map.fits", overwrite: bool = True):
        """
        Write map to a FITs file.

        Parameters
        ----------
        fname : str, default: 'map.fits'
            The path to save the map to.
        overwrite : bool, default: True
            Overwrite an existing map.

        Raises
        ------
        ValueError
            If the shape of the map is not correct for a healpix map.
        """
        if self.map.shape[1] > 1:
            raise ValueError("Map doesn't seem to be a healpix map")
        if self.npol == 1:
            healpy.write_map(
                fname, self.map[:, 0], nest=(self.proj == "NEST"), overwrite=overwrite
            )
        else:
            ind = fname.rfind(".")
            if ind > 0:
                if fname[ind + 1 :] == "fits":
                    head = fname[:ind]
                    tail = fname[ind:]
                else:
                    head = fname
                    tail = ".fits"
            else:
                head = fname
                tail = ".fits"
            tmp = np.zeros(self.nx)
            for i in range(self.npol):
                tmp[:] = np.squeeze(self.map[:, :, i]).T
                fname = head + "_" + self.pols[i] + tail
                healpy.write_map(
                    fname, tmp, nest=(self.proj == "NEST"), overwrite=overwrite
                )
