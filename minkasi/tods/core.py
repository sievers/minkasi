import copy
import time
from typing import Any, Optional, Literal, overload
import numpy as np
from numpy.typing import NDArray
from .utils import slice_with_copy
from .cuts import CutsCompact
from ..noise import NoiseModelType, WithDetWeights, NoiseSmoothedSVD, NoiseCMWhite
from ..maps import MapType, Mapset, SkyMap
from ..minkasi import have_mpi, comm, MPI
from .. import mkfftw

try:
    from typing import Self, deprecated
except:
    from typing_extensions import Self, deprecated


class Tod:
    """
    Class to store time ordered data.

    Attributes
    ----------
    info : dict
        Dict that can me used to store data associated with the TOD.
        Includes data like dat_calib, dx (RA), dy (dec), ctime, etc.
        as well as metadata like tag and noise.
    jumps : list | None
        Locations of jumps.
        If not None it should be a list of length ndet,
        where each element in the list is either None (no jumps)
        or a list of samples with jumps in them.
    cuts : list | None
        List of cuts for the TOD.
    noise : NoiseModelType | None
        Instance of noise model to use.
    noise_delayed : bool
        If True then noise model will not be initialized until noise is applied.
    noise_args : tuple
        Arguments to pass to noise class.
        Only set if set_noise is called with delayed=True.
    noise_kwargs : dict
        Keyword arguments to pass to noise class.
        Only set if set_noise is called with delayed=True.
    noise_modelclass : type[NoiseModelType]
        Class to use for noise model.
        Only set if set_noise is called with delayed=True.
    """

    def __init__(self, info):
        """
        Initialize the Tod object.

        Parameters
        ----------
        info : dict
            Dictionairy comtaining data to be stored with the TOD.
            See class docstring for more info.
        """
        self.info: dict = info.copy()
        self.jumps: list | None = None
        self.cuts: list | None = None
        self.noise: NoiseModelType | None = None
        self.noise_delayed: bool = False
        self.noise_args: tuple = ()
        self.noise_kwargs: dict = {}
        self.noise_modelclass: type[NoiseModelType]

    def lims(self) -> tuple[float, float, float, float]:
        """
        Get RA/dec limits of this TOD.

        Returns
        -------
        xmin : float
            The min RA value.
        xmax : float
            The max RA value.
        ymin : float
            The min dec value.
        ymax : float
            The max dec value.
        """
        xmin: float = self.info["dx"].min()
        xmax: float = self.info["dx"].max()
        ymin: float = self.info["dy"].min()
        ymax: float = self.info["dy"].max()
        return xmin, xmax, ymin, ymax

    def set_apix(self):
        """
        Calculates dxel normalized to +-1 from elevation.
        Stored in self.info['apix'].
        """
        # TBD pass in and calculate scan center's elevation vs time
        elev = np.mean(self.info["elev"], axis=0)
        x = np.arange(elev.shape[0]) / elev.shape[0]
        a = np.polyfit(x, elev, 2)
        ndet = self.info["elev"].shape[0]
        # Unused array is xel
        track_elev, _ = np.meshgrid(a[2] + a[1] * x + a[0] * x**2, np.ones(ndet))
        delev = self.info["elev"] - track_elev
        ml = np.max(np.abs(delev))
        self.info["apix"] = delev / ml

    def get_ndet(self) -> int:
        """
        Get number of detectors in this TOD.

        Returns
        -------
        ndet : int
            Number of detectors.
        """
        return self.info["dat_calib"].shape[0]

    def get_ndata(self) -> int:
        """
        Get number of samples per detector in this TOD.

        Returns
        -------
        ndata : int
            Number of samples per detectors.
        """
        return self.info["dat_calib"].shape[1]

    def get_nsamp(self) -> int:
        """
        Get total number of samples in this TOD.

        Returns
        -------
        nsamp : int
            Number of samples.
        """
        return self.get_ndet() * self.get_ndata()

    def get_data_dims(self) -> tuple[int, int]:
        """
        Get shape of TOD data.

        Returns
        -------
        dims : tuple[int, int]
            The dimensions of the data.
        """
        return (self.get_ndet(), self.get_ndata())

    def get_saved_pix(self, tag: str = "ipix") -> NDArray[np.int32] | None:
        """
        Get pixelization array.

        Parameters
        ----------
        tag : str , default: 'ipix'
            Tag that pixelization is saved under.

        Returns
        -------
        ipix : NDArray[np.int32] | None
            Loaded pixelization, if it doesn't exist None is returned.
        """
        return self.info.get(tag, None)

    def clear_saved_pix(self, tag: str = "ipix"):
        """
        Clear pixelization array.

        Parameters
        ----------
        tag : str , default: 'ipix'
            Tag that pixelization is saved under.
        """
        if tag in self.info.keys():
            del self.info[tag]

    def save_pixellization(self, tag: str, ipix: NDArray[np.int32]):
        """
        Save pixelization to self.info.

        Parameters
        ----------
        tag : str | None
            Tag to save pixelization under.
        ipix : NDArray[np.int32]
            Pixelization to save.
        """
        if tag in self.info.keys():
            print("warning - overwriting key ", tag, " in tod.save_pixellization.")
        self.info[tag] = ipix

    def set_tag(self, tag: Any):
        """
        Set self.info['tag'].
        Can be used to keep track of TODs in a TodVec.

        Parameters
        ----------
        tag : Any
            The value to save as the tag.
        """
        if "tag" in self.info:
            print("Warning: overwriting tag")
        self.info["tag"] = tag

    def set_pix(self, map: MapType):
        """
        Set pixelization using a map.
        Uses the same tag as the map.

        map : MapType
            Map to use for pixelization.
        """
        ipix = map.get_pix(self)
        self.save_pixellization(map.tag, ipix)

    def get_data(self, dat: NDArray[np.floating] | None = None) -> NDArray[np.floating]:
        """
        Get the data array for this TOD.
        Nominally from self.info['dat_calib'], but if a valid dat is passed in it is returned.

        Parameters
        ----------
        dat : NDArray[np.floating] | None, default: None
            If this is None then self.info["dat_calib"] is returned.
            Otherwise this is just returned.
            This can be useful to have a shared interface accross functions.

        Returns
        ------
        dat : NDArray[np.floating]
           The data array.

        Raises
        ------
        ValueError
            If both dat and self.info['dat_calib'] doesn't exist.
        """
        if dat is None:
            dat = self.info.get("dat_calib", None)
        if dat is None:
            raise ValueError(
                "This TOD has no stored data. Populare tod.info['dat_calib']"
            )
        return dat

    def get_tvec(self) -> NDArray[np.floating]:
        """
        Get the time vector from self.info['ctime']

        Returns
        ------
        tvec: NDArray[np.floating]
           The time vector.
        """
        return self.info["ctime"]

    def get_radec(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Get the RA/dec arrays for this TOD.

        Returns
        -------
        ra : NDArray[np.floating]
            The RA of the TOD.
        dec : NDArray[np.floating]
            The dec of the TOD.
        """
        return self.info["dx"], self.info["dy"]

    def get_empty(self, clear: bool = False) -> NDArray:
        """
        Get an array of size (ndet, ndata).

        Parameters
        ----------
        clear : bool, default: False
            Zero out the array before returning.

        Returns
        -------
        empty : NDArray
            ndet by nsamps array.
            dtype is set to (in order of priority):
            self.info['dtype'], self.info['dat_calib'].dtype, or float.
        """
        if "dtype" in self.info.keys():
            dtype = self.info["dtype"]
        elif "dat_calib" in self.info.keys():
            dtype = self.info["dat_calib"].dtype
        else:
            dtype = "float"
        if clear:
            return np.zeros([self.get_ndet(), self.get_ndata()], dtype=dtype)
        else:
            return np.empty([self.get_ndet(), self.get_ndata()], dtype=dtype)

    def copy(self, copy_info: bool = False) -> Self:
        """
        Make a copy of this TOD.

        Parameters
        ----------
        copy_info : bool, default: False
            Add a new copy of tod.info to the new TOD.

        Returns
        -------
        tod : Tod
            A copy of this Tod.
        """
        if copy_info:
            myinfo = self.info.copy()
            for key in myinfo.keys():
                try:
                    myinfo[key] = self.info[key].copy()
                except:
                    pass
            tod = Tod(myinfo)
        else:
            tod = Tod(self.info)
        if not (self.jumps is None):
            try:
                tod.jumps = self.jumps.copy()
            except:
                tod.jumps = self.jumps[:]
        if not (self.cuts is None):
            try:
                tod.cuts = self.cuts.copy()
            except:
                tod.cuts = self.cuts[:]
            tod.cuts = self.cuts[:]
        tod.noise = self.noise

        return tod

    def set_noise(
        self,
        modelclass: type[NoiseModelType] = NoiseSmoothedSVD,
        dat: NDArray[np.floating] | None = None,
        delayed: bool = False,
        *args,
        **kwargs
    ):
        """
        Setup the noise model for this TOD.

        Parameters
        ----------
        modelclass : type[NoiseModelType]
            Class to use for noise model.
            Must implement apply_noise and take the TOD's data as the first arg in its constructor.
        dat : NDArray[np.floating] | None, default: None
            Data to pass to noise model.
            Set to None to use self.info["dat_calib"].
        delayed : bool, default: False
            If True then model is not initialized but arguments are stored.
        *args : tuple
            Arguments to be passed to the noise model.
        *kwargs : dict
            Keyword arguments to be passed to the noise model.

        Raises
        ------
        ValueError
            If both dat and self.info['dat_calib'] are None.
        """
        if delayed:
            self.noise_args = copy.deepcopy(args)
            self.noise_kwargs = copy.deepcopy(kwargs)
            self.noise_delayed = True
            self.noise_modelclass = modelclass
        else:
            self.noise_delayed = False
            dat = self.get_data(dat)
            self.noise = modelclass(dat, *args, **kwargs)

    def get_det_weights(self) -> NDArray[np.floating] | None:
        """
        Get detector weights from noise model.

        Returns
        ------
        det_weights : NDArray[np.floating] | None
            The detector weights.
            None if noise model not set or model doesn't support weights.
        """
        if self.noise is None:
            print("noise model not set in get_det_weights.")
            return None
        if isinstance(self.noise, WithDetWeights):
            return self.noise.get_det_weights()
        print("noise model does not support detector weights in get_det_weights.")
        return None

    def set_noise_white_masked(self):
        """
        Set noise to be white and fill self.info["mywt"] with ones.
        """
        self.info["noise"] = "white_masked"
        self.info["mywt"] = np.ones(self.get_ndet())

    def apply_noise_white_masked(self, dat: NDArray[np.floating] | None = None):
        """
        Apply white noise and mask to data.
        self.info should contain 'mask' and 'mywt'.

        Parameters
        ----------
        dat : NDArray[np.floating] | None, default: None
            Data to pass to noise model.
            Set to None to use self.info["dat_calib"].

        Returns
        -------
        dd : NDArray[np.floating]
            Data with mask applied multiplied by white noise.

        Raises
        ------
        ValueError
            If both dat and self.info['dat_calib'] are None.
        """
        dat = self.get_data(dat)
        dd = (
            self.info["mask"]
            * dat
            * np.repeat([self.info["mywt"]], self.get_ndata(), axis=0).transpose()
        )
        return dd

    @deprecated("Use tod.set_noise(NoiseCMWhite) instead")
    def set_noise_cm_white(self):
        print("deprecated usage - please switch to tod.set_noise(NoiseCMWhite)")
        self.set_noise(NoiseCMWhite)
        return

        u, s, v = np.linalg.svd(self.info["dat_calib"], 0)
        ndet = len(s)
        ind = np.argmax(s)
        mode = np.zeros(ndet)
        # mode[:]=u[:,0]  #bug fixes pointed out by Fernando Zago.  19 Nov 2019
        # pred=np.outer(mode,v[0,:])
        mode[:] = u[:, ind]
        pred = np.outer(mode * s[ind], v[ind, :])

        dat_clean = self.info["dat_calib"] - pred
        myvar = np.std(dat_clean, 1) ** 2
        self.info["v"] = mode
        self.info["mywt"] = 1.0 / myvar
        self.info["noise"] = "cm_white"

    def apply_noise_cm_white(
        self, dat: NDArray[np.floating] | None = None
    ) -> NDArray[np.floating]:
        """
        Apply white noise with common mode.
        self.info should contain 'v' and 'mywt'.

        Parameters
        ----------
        dat : NDArray[np.floating] | None, default: None
            Data to pass to noise model.
            Set to None to use self.info["dat_calib"].

        Returns
        -------
        dd : NDArray[np.floating]
            Data with noise model applied.

        Raises
        ------
        ValueError
            If both dat and self.info['dat_calib'] are None.
        """
        print(
            "I'm not sure how you got here (tod.apply_noise_cm_white), but you should not have been able to.  Please complain to someone."
        )
        dat = self.get_data(dat)

        mat = np.dot(self.info["v"], np.diag(self.info["mywt"]))
        lhs = np.dot(self.info["v"], mat.transpose())
        rhs = np.dot(mat, dat)
        # if len(lhs)>1:
        if isinstance(lhs, np.ndarray):
            cm = np.dot(np.linalg.inv(lhs), rhs)
        else:
            cm = rhs / lhs
        dd = dat - np.outer(self.info["v"], cm)
        tmp = np.repeat([self.info["mywt"]], len(cm), axis=0).transpose()
        dd = dd * tmp
        return dd

    @deprecated("Use tod.set_noise(NoiseBinnedEig) instead")
    def set_noise_binned_eig(self, dat=None, freqs=None, scale_facs=None, thresh=5.0):
        dat = self.get_data(dat)
        mycov = np.dot(dat, dat.T)
        mycov = 0.5 * (mycov + mycov.T)
        ee, vv = np.linalg.eig(mycov)
        mask = ee > thresh * thresh * np.median(ee)
        vecs = vv[:, mask]
        ts = np.dot(vecs.T, dat)
        resid = dat - np.dot(vv[:, mask], ts)

        return resid

    @deprecated("Use tod.set_noise(NoiseSmoothedSVD) instead")
    def set_noise_smoothed_svd(
        self, fwhm=50, func=None, pars=None, prewhiten=False, fit_powlaw=False
    ):
        """If func comes in as not empty, assume we can call func(pars,tod) to get a predicted model for the tod that
        we subtract off before estimating the noise."""

        print(
            "deprecated usage - please switch to tod.set_noise(minkasi.NoiseSmoothedSVD)"
        )

        if func is None:
            self.set_noise(NoiseSmoothedSVD, self.info["dat_calib"])
        else:
            dat_use = func(pars, self)
            dat_use = self.info["dat_calib"] - dat_use
            self.set_noise(NoiseSmoothedSVD, dat_use)
        return

        if func is None:
            dat_use = self.info["dat_calib"]
        else:
            dat_use = func(pars, self)
            dat_use = self.info["dat_calib"] - dat_use
            # u,s,v=numpy.linalg.svd(self.info['dat_calib']-tmp,0)
        if prewhiten:
            noisevec = np.median(np.abs(np.diff(dat_use, axis=1)), axis=1)
            dat_use = dat_use / (
                np.repeat([noisevec], dat_use.shape[1], axis=0).transpose()
            )
        u, s, v = np.linalg.svd(dat_use, 0)
        print("got svd")
        ndet = s.size
        n = self.info["dat_calib"].shape[1]
        self.info["v"] = np.zeros([ndet, ndet])
        self.info["v"][:] = u.transpose()
        dat_rot = np.dot(self.info["v"], self.info["dat_calib"])
        if fit_powlaw:
            spec_smooth = 0 * dat_rot
            for ind in range(ndet):
                fitp, datsqr, C = fit_ts_ps(dat_rot[ind, :])
                spec_smooth[ind, 1:] = C
        else:
            dat_trans = mkfftw.fft_r2r(dat_rot)
            spec_smooth = smooth_many_vecs(dat_trans**2, fwhm)
        spec_smooth[:, 1:] = 1.0 / spec_smooth[:, 1:]
        spec_smooth[:, 0] = 0
        if prewhiten:
            self.info["noisevec"] = noisevec.copy()
        self.info["mywt"] = spec_smooth
        self.info["noise"] = "smoothed_svd"
        # return dat_rot

    def apply_noise(
        self, dat: NDArray[np.floating] | None = None
    ) -> NDArray[np.floating]:
        """
        Apply noise model to the TOD.

        Parameters
        ----------
        dat : NDArray[np.floating] | None, default: None
            Data to pass to noise model.
            Set to None to use self.info["dat_calib"].

        Returns
        -------
        dd : NDArray[np.floating]
            The data with the noise model applied.

        Raises
        ------
        ValueError
            If dat somehow becomes None when applying noise.
        """
        # the .copy() is here so you don't overwrite data stored in the TOD.
        dat = self.get_data(dat)
        if self.noise_delayed:
            self.noise = self.noise_modelclass(
                dat, *(self.noise_args), **(self.noise_kwargs)
            )
            self.noise_delayed = False
        if isinstance(self.noise, NoiseModelType):
            return self.noise.apply_noise(dat)
        print("unable to use class-based noised, falling back onto hardwired.")

        if self.info["noise"] == "cm_white":
            return self.apply_noise_cm_white(dat)
        if self.info["noise"] == "white_masked":
            return self.apply_noise_white_masked(dat)

        noisemat = 1.0
        if "noisevec" in self.info:
            noisemat = np.repeat(
                [self.info["noisevec"]], dat.shape[1], axis=0
            ).transpose()
            dat = dat / noisemat
        dat_rot = np.dot(self.info["v"], dat)

        datft = mkfftw.fft_r2r(dat_rot)
        nn = datft.shape[1]
        datft = datft * self.info["mywt"][:, 0:nn]
        dat_rot: NDArray[np.floating] = mkfftw.fft_r2r(datft)
        dat = np.dot(self.info["v"].transpose(), dat_rot)
        # if self.info.has_key('noisevec'):
        if dat is None:
            raise ValueError("Something odd has happened, dat is None")
        if "noisevec" in self.info:
            dat = dat / noisemat
        dat[:, 0] = 0.5 * dat[:, 0]
        dat[:, -1] = 0.5 * dat[:, -1]

        return dat

    def mapset2tod(self, mapset: Mapset, dat: NDArray[np.floating] | None = None):
        """
        Project a mapset into a TOD.

        Arguments
        --------
        mapset : Mapset
            The mapset to make the TOD from.
        dat : NDArray[np.floating] | None, default: None
            Array to output the TOD to.
            If None, an array of zeros will be made from self.info['dat_calib']

        Returns
        -------
        dat : NDArray[np.floating]
            The mapset as a TOD.
        """
        if dat is None:
            dat = self.get_empty(True)
        for map in mapset.maps:
            map.map2tod(self, dat)
        return dat

    def tod2mapset(self, mapset: Mapset, dat: NDArray[np.floating] | None = None):
        """
        Fill a mapset with maps projected from this TOD.

        Parameters
        ----------
        mapset : Mapset
            Mapset with the maps to fill.
        dat : NDArray[np.floating] | None
            Data to project to maps.
            If None self.info['dat_calib'] is used.
        """
        if dat is None:
            dat = self.get_data()
        for map in mapset.maps:
            map.tod2map(self, dat)

    @overload
    def dot(
        self, mapset: Mapset, mapset_out: Mapset, times: Literal[True] = True
    ) -> NDArray[np.floating]:
        ...

    @overload
    def dot(
        self, mapset: Mapset, mapset_out: Mapset, times: Literal[False] = False
    ) -> None:
        ...

    @overload
    def dot(
        self, mapset: Mapset, mapset_out: Mapset, times: bool = False
    ) -> Optional[NDArray[np.floating]]:
        ...

    def dot(
        self, mapset: Mapset, mapset_out: Mapset, times: bool = False
    ) -> Optional[NDArray[np.floating]]:
        """
        Project mapset into a TOD, apply noise model, and reproject into maps.

        Parameters
        ----------
        mapset : Mapset
            Input mapset. All maps will get summed into one TOD.
        mapset_out : Mapset
            Output mapset.
        times : bool, default: False
            If True compute the time each step takes and return it as a (3,) array.

        Returns
        ------
        dt : Optional[NDArray[np.floating]]
            Array containing the time it took to make a TOD from the mapset,
            to filter the TOD, and to fill the output mapset from it.
            Only returned it time is True.
        """
        t1 = time.time()
        tmp = self.mapset2tod(mapset)
        t2 = time.time()
        tmp = self.apply_noise(tmp)
        t3 = time.time()
        self.tod2mapset(mapset_out, tmp)
        t4 = time.time()
        if times:
            return np.asarray([t2 - t1, t3 - t2, t4 - t3])

    def set_jumps(self, jumps: list | None):
        """
        Set the jumps attribute for the TOD.

        Parameters
        ----------
        jumps : list | None
            Locations of jumps.
            See Tod class docstring for details.
        """
        self.jumps = jumps

    def cut_detectors(self, isgood: NDArray[np.bool_]):
        """
        Cut all detectors not in boolean array isgood.

        Parameters
        ---------
        isgood : NDArray[np.bool_]
            Detectors to keep.
        """
        isbad = ~isgood.astype(bool)
        bad_inds = np.where(isbad)
        # need to delete indices in reverse
        bad_inds = np.fliplr(bad_inds)
        bad_inds = bad_inds[0]
        print(bad_inds)
        for key in self.info.keys():
            if isinstance(self.info[key], np.ndarray):
                self.info[key] = slice_with_copy(self.info[key], isgood)
        if self.jumps is not None:
            for i in bad_inds:
                print("i in bad_inds is ", i)
                del self.jumps[i]
        if self.cuts is not None:
            for i in bad_inds:
                del self.cuts[i]

    def timestream_chisq(self, dat: NDArray[np.floating] | None = None) -> float:
        """
        Get chi squared of TOD.

        Parameters
        ----------
        dat : NDArray[np.floating] | None, default: None
            Data to calculate chisq of.
            If None, self.info['dat_calib'] will be used.

        Returns
        -------
        chisq : float
            The chisq of the data.
        """
        dat = self.get_data(dat)
        dat_filt = self.apply_noise(dat)
        chisq = np.sum(dat_filt * dat)
        return chisq

    def prior_from_skymap(self, skymap: SkyMap) -> CutsCompact:
        """.
        Given e.g. the gradient of a map that has been zeroed under some threshold,
        return a CutsCompact object that can be used as a prior for solving for per-sample deviations
        due to strong map gradients.  This is to reduce X's around bright sources.

        Parameters
        ---------
        skymap : SkyMap
            SkyMap that is non-zero where one wishes to solve for the per-sample deviations.

        Returns
        -------
        prior : CutsCompact
            Prior that can be uesd for solving for per-sample deviations.
            Will have the 1/intput squared weight in its map.

        Raises
        ------
        ValueError
            If prior.map is doesn't populate properly.
        """
        tmp = np.zeros(self.info["dat_calib"].shape)
        skymap.map2tod(self, tmp)
        mask = tmp == 0
        prior = CutsCompact(self)
        prior.cuts_from_array(mask)
        prior.get_imap()
        prior.tod2map(self, tmp)
        if prior.map is None:
            raise ValueError("Prior map is None, something went wrong")
        prior.map = 1.0 / prior.map**2
        return prior


class TodVec:
    """
    Class to store a collection of TODs.

    Attributes
    ----------
    tods : list[Tod]
        The stored TODs.
    ntod : int
        Number of stored TODs.
    """

    def __init__(self):
        """
        Initialize an empty TodVec.
        """
        self.tods: list[Tod] = []
        self.ntod: int = 0

    def add_tod(self, tod: Tod, copy_info: bool = False):
        """
        Add a TOD to the TodVec.
        Makes a copy of the TOD when it is added.

        Parameters
        ----------
        tod : Tod
            The TOD to add.

        copy_info : bool, default: False
            Make a full copy of tod.info when adding the TOD.
        """
        self.tods.append(tod.copy(copy_info))
        self.tods[-1].set_tag(self.ntod)
        self.ntod = self.ntod + 1

    def lims(self) -> Optional[tuple[float, float, float, float]]:
        """
        Get global limits of all TODs.
        This function is MPI aware, so the limits are the same across processes.
        If ntod is 0 then None is returned.

        Returns
        -------
        xmin : float
            The min RA value.
        xmax : float
            The max RA value.
        ymin : float
            The min dec value.
        ymax : float
            The max dec value.
        """
        if self.ntod == 0:
            return None
        xmin, xmax, ymin, ymax = self.tods[0].lims()
        for i in range(1, self.ntod):
            x1, x2, y1, y2 = self.tods[i].lims()
            xmin = min(x1, xmin)
            xmax = max(x2, xmax)
            ymin = min(y1, ymin)
            ymax = max(y2, ymax)
        if have_mpi:
            print("before reduction lims are ", [xmin, xmax, ymin, ymax])
            xmin = comm.allreduce(xmin, op=MPI.MIN)
            xmax = comm.allreduce(xmax, op=MPI.MAX)
            ymin = comm.allreduce(ymin, op=MPI.MIN)
            ymax = comm.allreduce(ymax, op=MPI.MAX)
            print("after reduction lims are ", [xmin, xmax, ymin, ymax])
        return xmin, xmax, ymin, ymax

    def get_nsamp(self, reduce: bool = True) -> int:
        """
        Get total number of samples in TodVec.
        This function can be MPI aware.

        Parameters
        ----------
        reduce : bool, default: True
            If True calculate number of samples across all processes.

        Returns
        -------
        tot : int
            The total number of samples.
        """
        tot: int = 0
        for tod in self.tods:
            tot = tot + tod.get_nsamp()
        if reduce and have_mpi:
            tot = comm.allreduce(tot)
        return tot

    def set_pix(self, map: MapType):
        """
        Calculate pixelization for all TODs.

        Parameters
        ----------
        map : MapType
            The map to use for pixelization.
        """
        for tod in self.tods:
            tod.set_pix(map)

    def set_apix(self):
        """
        Calculates dxel normalized to +-1 from elevation for all TODs.
        Stored in tod.info['apix'].
        """
        for tod in self.tods:
            tod.set_apix()

    def dot_cached(self, mapset: Mapset, mapset_out: Mapset | None = None) -> Mapset:
        """
        Take dot of all the TODs in this TodVec.
        This function uses caches maps, so tod2map_cached is called.

        Parameters
        ----------
        mapset : Mapset
            Input Mapset.
            See Tod.dot for details.
        mapset_out : Mapset | None.
            Output Mapset. If None then a blank copy of Mapset is used.
            See Tod.dot for details.

        Returns
        -------
        mapset_out : Mapset
            Output mapset filled with values from calling Tod.dot.
        """
        if mapset_out is None:
            mapset_out = mapset.copy()
            mapset_out.clear()
        mapset_out.get_caches()
        for tod in self.tods:
            tod.dot(mapset, mapset_out)
        mapset_out.clear_caches()
        if have_mpi:
            mapset_out.mpi_reduce()

        return mapset_out

    @overload
    def dot(
        self,
        mapset: Mapset,
        mapset_out: Mapset | None = None,
        report_times: Literal[False] = False,
        cache_maps: bool = False,
    ) -> Mapset:
        ...

    @overload
    def dot(
        self,
        mapset: Mapset,
        mapset_out: Mapset | None = None,
        report_times: Literal[True] = True,
        cache_maps: bool = False,
    ) -> tuple[Mapset, NDArray[np.floating]]:
        ...

    @overload
    def dot(
        self,
        mapset: Mapset,
        mapset_out: Mapset | None = None,
        report_times: bool = False,
        cache_maps: bool = False,
    ) -> Mapset | tuple[Mapset, NDArray[np.floating]]:
        ...

    def dot(
        self,
        mapset: Mapset,
        mapset_out: Mapset | None = None,
        report_times: bool = False,
        cache_maps: bool = False,
    ) -> Mapset | tuple[Mapset, NDArray[np.floating]]:
        """
        Take dot of all the TODs in this TodVec.
        Also prints out the sum of times from Tod.dot.

        Parameters
        ----------
        mapset : Mapset
            Input Mapset.
            See Tod.dot for details.
        mapset_out : Mapset | None.
            Output Mapset. If None then a blank copy of Mapset is used.
            See Tod.dot for details.
        report_times : bool, default: False
            If True return the time it takes to run dot on each TOD.
        cache_maps : bool, default: False
            Run TodVec.dot_cached.

        Returns
        -------
        mapset_out : Mapset
            Output mapset filled with values from calling Tod.dot.
        times : NDArray[np.floating]
            (ntod,) array with the time it took to run dot for each TOD.
            Only returned if report_times is True.
        """
        if mapset_out is None:
            mapset_out = mapset.copy()
            mapset_out.clear()

        if cache_maps:
            mapset_out = self.dot_cached(mapset, mapset_out)
            return mapset_out

        times = np.zeros(self.ntod)
        tot_times = np.zeros(3)
        for i in range(self.ntod):
            tod = self.tods[i]
            t1 = time.time()
            mytimes = tod.dot(mapset, mapset_out, True)
            t2 = time.time()
            tot_times = tot_times + mytimes
            times[i] = t2 - t1
        if have_mpi:
            mapset_out.mpi_reduce()
        print(tot_times)
        if report_times:
            return mapset_out, times
        return mapset_out

    def make_rhs(self, mapset: Mapset, do_clear: bool = False):
        """
        Apply noise to TODs and then project into maps.

        Parameters
        ----------
        mapset : Mapset
            Mapset with maps to project to the TODs into.
        do_clear : bool, default: False
            If True then clears the maps first.
        """
        if do_clear:
            mapset.clear()
        for tod in self.tods:
            dat_filt = tod.apply_noise()
            for map in mapset.maps:
                map.tod2map(tod, dat_filt)

        if have_mpi:
            mapset.mpi_reduce()
