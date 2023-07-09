import copy
from typing import Callable
from astropy import wcs
from astropy.io import fits
import numpy as np
from numpy.typing import NDArray
from .utils import get_wcs
from ..minkasi import find_good_fft_lens, have_mpi, comm, nproc, get_nthread
from ..tod2map import (
    tod2map_cached,
    tod2map_omp,
    tod2map_simple,
    tod2map_everyone,
    tod2map_atomic,
)
from ..map2tod import map2tod
from ..tods import Tod, TodVec

try:
    import healpy

    have_healpy = True
except ImportError:
    have_healpy = False

try:
    from typing import Self
except ImportError:
    from typing import TypeVar

    Self = TypeVar("Self", bound="SkyMap")


class SkyMap:
    """
    Class to store and operate on a map.
    If you need polarization support use PolMap.

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
        The actual map. Shape is (nx, ny).
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
        Initialize the SkyMap.

        Parameters
        ----------
        lims : tuple[float, ...] | list[float] | NDArray[np.floating]
            The limits of ra/dec (ra_low, ra_high, dec_low, dec_high).
        pixsize : float
            The size of a pixel in radians.
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
        tod2map_method : Callable | None
           Can be set with set_tod2map.
           Presumably sets the method used by tod2map but not currently used.
        """
        self.wcs: wcs.WCS
        if mywcs is None:
            assert (
                pixsize != 0
            )  # we had better have a pixel size if we don't have an incoming WCS that contains it
            self.wcs = get_wcs(lims, pixsize, proj, cosdec, ref_equ)
        else:
            self.wcs = mywcs
            pixsize_use = mywcs.wcs.cdelt[1] * np.pi / 180
            # print('pixel size from wcs and requested are ',pixsize_use,pixsize,100*(pixsize_use-pixsize)/pixsize)
            pixsize = pixsize_use

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
        self.caches = None
        self.cosdec = cosdec
        self.tod2map_method = None
        self.lims: tuple[float, ...] | list[float] | NDArray[np.floating] = lims
        self.tag: str = tag
        self.purge_pixellization: bool = purge_pixellization
        self.pixsize: float = pixsize
        self.map: NDArray[np.floating] = np.zeros([nx, ny])
        self.proj: str = proj
        self.pad: int = pad
        self.caches: NDArray[np.floating] | None = None
        self.cosdec: float | None = cosdec
        self.tod2map_method: Callable | None = None

    def get_caches(self):
        """
        Setup caches for this map.
        """
        npix: int = self.nx * self.ny
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

    def set_tod2map(self, method: str | None = None, todvec: TodVec | None = None):
        """
        Select which method of tod2map to use.
        Currently the set method doesn't seem to be used anywhere.

        Parameters
        ----------
        method : str | None, default: None
            Method to select, options are:
            * simple (1 proc)
            * omp (everyone makes a map copy)
            * everyone (everyone loops through all the data but assigns only to their own piece)
            * atomic (no map copy, accumulate via atomic adds)
            * cached (every thread has a sticky copy of the map)
            If set to None then it will choose between simple and omp based om nproc.
        todvec : TodVec | None, default: None
            The todvec. Only needed if method is 'everyone'

        Raises
        ------
        ValueError
            If todvec is not provided when method='everyone'.
        """
        if method is None:
            if nproc == 1:
                self.tod2map_method = self.tod2map_simple
            else:
                self.tod2map_method = self.tod2map_omp
            return
        if method == "omp":
            self.tod2map_method = self.tod2map_omp
            return
        if method == "simple":
            self.tod2map_method = self.tod2map_simple
        if method == "everyone":
            if todvec is None:
                raise ValueError(
                    "need tods when setting to everyone so we can know which pieces belong to which threads"
                )
            for tod in todvec.tods:
                ipix = self.get_pix(tod, False)
                ipix = ipix.copy()
                ipix = np.ravel(ipix)
                ipix.sort()
                inds = len(ipix) * np.arange(nproc + 1) // nproc
                inds = np.asarray(inds, dtype="int32")
                tod.save_pixellization(self.tag + "_edges", inds)
                self.tod2map_method = self.tod2map_everyone
        if method == "cached":
            self.get_caches()
            self.tod2map_method = self.tod2map_cached
        if method == "atomic":
            self.tod2map_method = self.tod2map_atomic

    def tod2map_atomic(self, tod: Tod, dat: NDArray[np.floating]):
        """
        Wrapper that runs tod2map_atomic on this map.
        Since it is atomic there is no map copy, the map is accumulated with atomic adds.

        Parameters
        __________
        tod : Tod
            TOD containing pointing information.
        dat : NDArray[np.floating]
            Data to project to map space.
        """
        ipix = self.get_pix(tod)
        tod2map_omp(self.map, dat, ipix, True)

    def tod2map_omp(self, tod: Tod, dat: NDArray[np.floating]):
        """
        Wrapper that runs tod2map_omp this map.
        Runs with parallelization but every map makes its own copy

        Parameters
        __________
        tod : Tod
            TOD containing pointing information.
        dat : NDArray[np.floating]
            Data to project to map space.
        """
        ipix = self.get_pix(tod)
        tod2map_omp(self.map, dat, ipix, False)

    def tod2map_simple(self, tod: Tod, dat: NDArray[np.floating]):
        """
        Wrapper that runs tod2map_simple on this map.
        This function is not parallelized.

        Parameters
        __________
        tod : Tod
            TOD containing pointing information.
        dat : NDArray[np.floating]
            Data to project to map space.
        """
        ipix = self.get_pix(tod)
        tod2map_simple(self.map, dat, ipix)

    def tod2map_everyone(self, tod: Tod, dat: NDArray[np.floating]):
        """
        Wrapper that runs tod2map_everyone on this map.

        Parameters
        __________
        tod : Tod
            TOD containing pointing information and edges.
            Edges are expected to be saved at tod.info[self.tag + "_edges"]
        dat : NDArray[np.floating]
            Data to project to map space.

        Raises
        ------
        ValueError
            If edges not saved in tod.
        """
        if self.tag + "_edges" not in tod.info:
            raise ValueError("Edges not set")
        ipix = self.get_pix(tod)
        tod2map_everyone(self.caches, dat, ipix, tod.info[self.tag + "_edges"])

    def tod2map_cached(self, tod: Tod, dat: NDArray[np.floating]):
        """
        Wrapper that runs tod2map_cached on this map.

        Parameters
        __________
        tod : Tod
            TOD containing pointing information.
        dat : NDArray[np.floating]
            Data to project to map space.

        Raises
        ------
        ValueError
            If caches are not set.
        """
        if self.caches is None:
            raise ValueError("Caches not set")
        ipix = self.get_pix(tod)
        tod2map_cached(self.caches, dat, ipix)

    def copy(self) -> Self:
        """
        Return a copy of this map
        """
        if False:
            newmap = SkyMap(
                self.lims,
                self.pixsize,
                self.proj,
                self.pad,
                self.primes,
                cosdec=self.cosdec,
                nx=self.nx,
                ny=self.ny,
                mywcs=self.wcs,
                tag=self.tag,
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
        map : SkyMap
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
        # self.map[:,:]=arr
        self.map[:] = arr

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
        xpix = np.reshape(pix[:, 0], [ndet, nsamp]) - 1
        ypix = np.reshape(pix[:, 1], [ndet, nsamp]) - 1
        xpix = np.round(xpix)
        ypix = np.round(ypix)
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
            Save the pixellization in the Tod at tod.info[self.tag].

        Returns
        -------
        ipix : NDArray[np.int32]
            The pixellization.
        """
        if not (self.tag is None):
            ipix = tod.get_saved_pix(self.tag)
            if not (ipix is None):
                return ipix
        ra, dec = tod.get_radec()
        # ndet=tod.info['dx'].shape[0]
        # nsamp=tod.info['dx'].shape[1]
        if False:
            ndet = ra.shape[0]
            nsamp = ra.shape[1]
            nn = ndet * nsamp
            coords = np.zeros([nn, 2])
            # coords[:,0]=np.reshape(tod.info['dx']*180/np.pi,nn)
            # coords[:,1]=np.reshape(tod.info['dy']*180/np.pi,nn)
            coords[:, 0] = np.reshape(ra * 180 / np.pi, nn)
            coords[:, 1] = np.reshape(dec * 180 / np.pi, nn)

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
        dat : NDArray[np.floating]
            Array to pull tod data from.
            Shape should be (ndet, nsamps).
        do_add : bool, default: True.
            If True add the projected map to this map.
            If False replace this map with it.
        do_omp : bool, default: False
            Use OMP to parallelize.
        """
        if dat is None:
            dat = tod.get_data()
        if do_add == False:
            self.clear()
        ipix = self.get_pix(tod)

        if not (self.caches is None):
            # tod2map_cached(self.caches,dat,tod.info['ipix'])
            tod2map_cached(self.caches, dat, ipix)
        else:
            if do_omp:
                # tod2map_omp(self.map,dat,tod.info['ipix'])
                tod2map_omp(self.map, dat, ipix)
            else:
                # tod2map_simple(self.map,dat,tod.info['ipix'])
                tod2map_simple(self.map, dat, ipix)
        if self.purge_pixellization:
            tod.clear_saved_pix(self.tag)

    def tod2map_old(self, tod, dat=None, do_add=True, do_omp=True):
        if dat is None:
            dat = tod.get_data()
        if do_add == False:
            self.clear()
        ipix = self.get_pix(tod)
        if not (self.caches is None):
            # tod2map_cached(self.caches,dat,tod.info['ipix'])
            tod2map_cached(self.caches, dat, ipix)
        else:
            if do_omp:
                # tod2map_omp(self.map,dat,tod.info['ipix'])
                tod2map_omp(self.map, dat, ipix)
            else:
                # tod2map_simple(self.map,dat,tod.info['ipix'])
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
        map : SkyMap
            Map to tahe dot product with.

        Returns
        -------
        tot : float
            The dot product
        """
        tot = np.sum(self.map * map.map)
        return tot

    def plot(self, plot_info: dict | None = None):
        """
        Plot the map.
        Defaut settings are:
        * vmin: self.map.min()
        * vmax: self.map.max()
        * clf: True
        * pause: True
        * pause_len: .001

        plot_info : dict | None
           Dict that can override the defaults for vmin, vmax, clf, pause, pause_len.
        """
        vmin = self.map.min()
        vmax = self.map.max()
        clf = True
        pause = True
        pause_len = 0.001
        if not (plot_info is None):
            if "vmin" in plot_info.keys():
                vmin = plot_info["vmin"]
            if "vmax" in plot_info.keys():
                vmax = plot_info["vmax"]
            if "clf" in plot_info.keys():
                clf = plot_info["clf"]
            if "pause" in plot_info.keys():
                pause = plot_info["pause"]
            if pause_len in plot_info.keys():
                pause_len = plot_info["pause_len"]
        from matplotlib import pyplot as plt

        if clf:
            plt.clf()
        plt.imshow(self.map, vmin=vmin, vmax=vmax)
        if pause:
            plt.pause(pause_len)

    def write(self, fname: str = "map.fits"):
        """
        Write map to a FITs file.

        Parameters
        ----------
        fname : str, default: 'map.fits'
            The path to save the map to.
        """
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
        map : SKyMap
            Map to multiply by.

        Returns
        -------
        new_map : PolMap
            The result of the multiplication.
        """
        new_map = map.copy()
        new_map.map[:] = self.map[:] * map.map[:]
        return new_map

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
        if have_mpi:
            if chunksize > 0:
                nchunk = (1.0 * self.nx * self.ny) / chunksize
                nchunk = int(np.ceil(nchunk))
            else:
                nchunk = 1
            if nchunk == 1:
                self.map = comm.allreduce(self.map)
            else:
                inds = np.asarray(
                    np.linspace(0, self.nx * self.ny, nchunk + 1), dtype="int"
                )
                if len(self.map.shape) > 1:
                    tmp = np.zeros(self.map.size)
                    tmp[:] = np.reshape(self.map, len(tmp))
                else:
                    tmp = self.map

                for i in range(len(inds) - 1):
                    tmp[inds[i] : inds[i + 1]] = comm.allreduce(
                        tmp[inds[i] : inds[i + 1]]
                    )
                    # self.map[inds[i]:inds[i+1]]=comm.allreduce(self.map[inds[i]:inds[i+1]])
                    # tmp=np.zeros(inds[i+1]-inds[i])
                    # tmp[:]=self.map[inds[i]:inds[i+1]]
                    # tmp=comm.allreduce(tmp)
                    # self.map[inds[i]:inds[i+1]]=tmp
                if len(self.map.shape) > 1:
                    self.map[:] = np.reshape(tmp, self.map.shape)

            # print("reduced")

    def invert(self):
        """
        Invert the map. Note that the inverted map is stored in self.map.
        """
        mask = np.abs(self.map) > 0
        self.map[mask] = 1.0 / self.map[mask]


class SkyMapCoarse(SkyMap):
    """
    Coarse SkyMap. Used for multi expiriment mapping.
    Seems to be a WIP, docstrings will come when it is more fleshed out.
    """

    def __init__(self, map):
        self.nx = map.shape[0]
        try:
            self.ny = map.shape[1]
        except:
            self.ny = 1
        self.map = map.copy()

    def get_caches(self):
        return

    def clear_caches(self):
        return

    def copy(self):
        cp = copy.copy(self)
        cp.map = self.map.copy()
        return cp

    def get_pix(self):
        return

    def map2tod(self, *args, **kwargs):
        return

    def tod2map(self, *args, **kwargs):
        return


class HealMap(SkyMap):
    """
    Subclass of SkyMap with healpix pixels.
    Attributes list below are only the ones initialized by this subclass.
    See SkyMap for the full list of attributes including the ones unused by this subclass.

    Attributes
    ----------
    nside : int
        The nside of the healpix map.
    nx : int
        The npix of the healpix map.
    ny : int
        Always 1, here for compatipility.
    tag : str
        Name to store pixellization under.
    purge_pixellization : bool
        Clear the pixellization when calling tod2map.
    map : NDArray[np.floating]
        The actual map. Shape is (nx, ny).
    proj : str
        The projection of the map.
    caches : NDArray[np.floating] | None
        Cached maps for tod2map_cached.
    """

    def __init__(
        self,
        proj: str = "RING",
        nside: int = 512,
        tag: str = "ipix",
        purge_pixellization: bool = False,
    ):
        """
        Initialize the HealMap.

        Parameters
        ----------
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
        self.proj: str = proj
        self.nside: int = nside
        self.nx: int = healpy.nside2npix(self.nside)
        self.ny: int = 1
        self.caches: NDArray[np.floating] | None = None
        self.tag: str = tag
        self.map: NDArray[np.floating] = np.zeros([self.nx, self.ny])
        self.purge_pixellization: bool = purge_pixellization

    def copy(self) -> Self:
        """
        Return a copy of this map
        """
        newmap = HealMap(self.proj, self.nside, self.tag)
        newmap.map[:] = self.map[:]
        return newmap

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
    #    #ipix=healpy.ang2pix(self.nside,np.pi/2-tod.info['dy'],tod.info['dx'],self.proj=='NEST')
    #    ipix=healpy.ang2pix(self.nside,np.pi/2-dec,ra,self.proj=='NEST')
    #    if savepix:
    #        tod.save_pixellization(self.tag,ipix)
    #    return ipix
    def write(self, fname="map.fits", overwrite=True):
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
        healpy.write_map(
            fname, self.map[:, 0], nest=(self.proj == "NEST"), overwrite=overwrite
        )


class SkyMapCar(SkyMap):
    """
    SkyMap subclass with CAR pixellization.
    See SkyMap for list of attributes.
    """

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
        if self.cosdec is None:
            self.cosdec = np.cos(0.5 * (self.lims[2] + self.lims[3]))
        xpix = np.round((ra - self.lims[0]) * self.cosdec / self.pixsize)
        # ypix=np.round((dec-self.lims[2])/self.pixsize)
        ypix = ((dec - self.lims[2]) / self.pixsize) + 0.5
        ipix: NDArray[np.int32] = np.asarray(xpix * self.ny + ypix, dtype="int32")
        return ipix


class SkyMapCarOld:
    def __init__(self, lims, pixsize):
        try:
            self.lims = lims.copy()
        except:
            self.lims = lims[:]
        self.pixsize = pixsize
        self.cosdec = np.cos(0.5 * (lims[2] + lims[3]))
        nx = int(np.ceil((lims[1] - lims[0]) / pixsize * self.cosdec))
        ny = int(np.ceil((lims[3] - lims[2]) / pixsize))
        self.nx = nx
        self.ny = ny
        self.npix = nx * ny
        self.map = np.zeros([nx, ny])

    def copy(self):
        mycopy = SkyMapCar(self.lims, self.pixsize)
        mycopy.map[:] = self.map[:]
        return mycopy

    def clear(self):
        self.map[:, :] = 0

    def axpy(self, map, a):
        self.map[:] = self.map[:] + a * map.map[:]

    def assign(self, arr):
        assert arr.shape[0] == self.nx
        assert arr.shape[1] == self.ny
        self.map[:, :] = arr

    def get_pix(self, tod):
        xpix = np.round((tod.info["dx"] - self.lims[0]) * self.cosdec / self.pixsize)
        ypix = np.round((tod.info["dy"] - self.lims[2]) / self.pixsize)
        # ipix=np.asarray(ypix*self.nx+xpix,dtype='int32')
        ipix = np.asarray(xpix * self.ny + ypix, dtype="int32")
        return ipix

    def map2tod(self, tod, dat, do_add=True, do_omp=True):
        map2tod(dat, self.map, tod.info["ipix"], do_add, do_omp)

    def tod2map(self, tod, dat, do_add=True, do_omp=True):
        if do_add == False:
            self.clear()
        if do_omp:
            tod2map_omp(self.map, dat, tod.info["ipix"])
        else:
            tod2map_simple(self.map, dat, tod.info["ipix"])

    def r_th_maps(self):
        xvec = np.arange(self.nx)
        xvec = xvec - xvec.mean()
        yvec = np.arange(self.ny)
        yvec = yvec - yvec.mean()
        ymat, xmat = np.meshgrid(yvec, xvec)
        rmat = np.sqrt(xmat**2 + ymat**2)
        th = np.arctan2(xmat, ymat)
        return rmat, th

    def dot(self, map):
        tot = np.sum(self.map * map.map)
        return tot
