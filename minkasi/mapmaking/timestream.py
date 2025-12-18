import copy

import numpy as np

from ..parallel import comm, have_mpi

try:
    from ..tools import fft
except ImportError:
    from ..tools import py_fft as fft
from .map2tod import map2tod_binned_det, map2tod_destriped, map2todbowl
from .tod2map import tod2map_binned_det, tod2map_destriped, tod2mapbowl


def scaled_airmass_from_el(mat):
    """
    Generates a simple 1/cos(el) model for airmass.

    Parameters
    ----------
    mat : np.array(float)
        Array of elevation values

    Returns
    -------
    airmass : np.array(float)
        1/cos(elv), normalized by the average value of the same. A typical simple model for airmass
    """

    airmass = 1 / np.cos(mat)
    airmass = airmass - airmass.mean()
    # airmass=airmass/np.std(airmass)
    return airmass


class tsGeneric:
    """
    Generic timestream model class. Used as a parent for other timestream/map classes. Defines multiple common methods
    TODO: IDK if there's more documentation to be written here vs. the methods

    """

    def __init__(self, tod=None):
        """
        Initialization function

        Parameters
        ----------
        tod : minkasi.tod
            tod from which the timestream is to be derived

        Returns
        -------
        none
        """
        # Set file name if tod specified
        self.fname = tod.info["fname"]

    def __mul__(self, to_mul):
        """
        Multiplies two timeseries. Returns a a new ts class with the multiplied parameters.

        Parameters
        ----------
        to_mul : minkasi.tod
            A tod to multiply with the tod associated with this instance of tsGeneric

        Returns
        -------
        tt : minkasi.tod
            a copy of tsGeneric but with the parameters multiplied

        """
        tt = self.copy()
        tt.params = self.params * to_mul.params
        return tt

    def clear(self):
        # clears parameters
        self.params[:] = 0

    def dot(self, common=None):
        # Returns the dot product of a ts class. If common is not specified, returns the self dot product, else returns the dot product with common
        if common is None:
            return np.sum(self.params * self.params)
        else:
            return np.sum(self.params * common.params)

    def axpy(self, common, a):
        # returns ts + a*common
        self.params = self.params + a * common.params

    def apply_prior(self, x, Ax):
        # TODO: not sure exactly what this does
        Ax.params = Ax.params + self.params * x.params

    def copy(self):
        # Returns a copy of the TS object
        return copy.deepcopy(self)

    def write(self, fname=None):
        # Currently not implemented
        pass


class tsVecs(tsGeneric):
    """
    Generic class for timestreams involving vectors, including tools to go from parameters (maps) to tods and back.

     Example would be fitting polynomials to timestreams. In this case the self.vecs would be the fit polynomials, self.params are the fit parameters of that polynomial. Then map2tod returns the tods predicted by the polynomials and fit parameters, i.e. self.vec@self.params, while tod2map returns the fit parameters that would generate a given tod.

    Attributes
    ----------
    fname: str
        name of the tod
    tod: tod object
        tods corresponding to the timestream
    vecs: n_data x nvec matrix
        vectors to which fit parameters are applied.
    ndet: int
        number of detectors in timestream
    nvec: int
        number of fit vectors
    params: np.array, float, nvec x ndet
        fit parameters for the vecs.

    """

    def __init__(self, tod, vecs):
        """
        Parameters
        ----------
        tod: tod object
             tod corresponding to the timestream
        vecs: n_data x n_predictors matrix
            vectors to which fit parameters are applied.
        """

        self.vecs = vecs
        # self.ndet=tod.info['dat_calib'].shape[0]
        self.ndet = tod.get_data_dims()[0]
        self.vecs = vecs
        self.nvec = vecs.shape[0]
        self.params = np.zeros([self.nvec, self.ndet])

    def tod2map(self, tod, mat=None, do_add=True, do_omp=False):
        """
        Computes the parameters of vecs which yield the given tod.

        In general Am = d for A the vecs, m the parameters, and d the tod. tod2map returns m given A and d. Essentially the inverse of map2tod.

        Parameters
        ----------
        tod: tod object
            The tod to convert to parameters
        mat: tod data object, optional
            d, the data corresponding to the tod we wish to convert to parameters. If not speicified the data is taken from tod
        do_add: bool, optional, default = True
            If true, adds the resulting parameters matrix to the existing parameters matrix. If false, overwrites the existing parameters
        do_omp: bool, optional, default = False
            Defines whether to use omp parallelization. Currently not implemented (?)

        Returns
        -------
        No returns

        Side effects
        ------------
        Updates params with the values infered from tod or mat.
        """

        if mat is None:
            # mat=tod.info['dat_calib']
            mat = tod.get_data()
        if do_add:
            self.params[:] = self.params[:] + np.dot(self.vecs, mat.T)
        else:
            self.params[:] = np.dot(self.vecs, mat.T)

    def map2tod(self, tod, mat=None, do_add=True, do_omp=False):
        """
        Given parameters and vecs, compute the corresponding tod.

        Given Am = for A the vecs, m the parameters, and d the tod data, return the d corresponding to the specified A and m. Essentially the inverse of tod2map.

        Parameters
        ----------
        tod: tod object
            The tod, needed for getting the data if to_add is True
        mat: tod data object, optional
            d, the data corresponding to the tod, only used if to_add is True. If not speicified the data is taken from tod
        do_add: bool, optional, default = True
            If true, adds the resulting tod data to the existing tod data. If false, overwrites the tod data.
        do_omp: bool, optional, default = False
            Defines whether to use omp parallelization. Currently not implemented (?)

        Returns
        -------
        No returns

        Side effects
        ------------
        Updates tod data with the values infered from params and vecs.
        """
        if mat is None:
            # mat=tod.info['dat_calib']
            mat = tod.get_data()
        if do_add:
            mat[:] = mat[:] + np.dot(self.params.T, self.vecs)
        else:
            mat[:] = np.dot(self.params.T, self.vecs)


class tsNotch(tsGeneric):
    def __init__(self, tod, numin, numax):
        self.fname = tod.info["fname"]
        tvec = tod.get_tvec()
        dt = tvec[-1] - tvec[0]
        bw = numax - numin
        dnu = 1 / dt
        nfreq = int(
            np.ceil(2 * bw / dnu)
        )  # factor of 2 is to account for partial waves
        ndet = tod.get_ndet()
        self.freqs = np.linspace(numin, numax, nfreq)
        self.nfreq = nfreq
        self.params = np.zeros([2 * nfreq, ndet])

    def get_vecs(self, tvec):
        tvec = tvec - tvec[0]
        vecs = np.zeros([self.nfreq * 2, len(tvec)])
        for i in range(self.nfreq):
            vecs[2 * i, :] = np.cos(tvec * self.freqs[i])
            vecs[2 * i + 1, :] = np.sin(tvec * self.freqs[i])
        return vecs

    def map2tod(self, tod, mat=None, do_add=True, do_omp=False):
        tvec = tod.get_tvec()
        vecs = self.get_vecs(tvec)
        pred = self.params.T @ vecs
        if mat is None:
            mat = tod.get_data()
        if do_add:
            mat[:] = mat[:] + pred
        else:
            mat[:] = pred

    def tod2map(self, tod, mat=None, do_add=True, do_omp=False):
        tvec = tod.get_tvec()
        vecs = self.get_vecs(tvec)
        if mat is None:
            mat = tod.get_data()
        # tmp=mat@(vecs.T)
        tmp = vecs @ mat.T
        if do_add:
            self.params[:] = self.params[:] + tmp
        else:
            self.params[:] = tmp


class tsPoly(tsVecs):
    """
    Class for fitting legandre polynomials to tods. Inheritted from tsVecs. Currently no map2tod or tod2map is implemented so this does not work.

    Attributes
    ----------
    fname: str
        name of the tod
    tod: tod object
        tods corresponding to the timestream
    vecs: n_data x nvec matrix
        Legandre polynomials to which fit parameters are applied.
    ndet: int
        number of detectors in timestream
    nvec: int
        number of fit polynomials
    params: np.array, float, nvec x ndet
        fit parameters for the vecs.

    """

    def __init__(self, tod, order=10):
        """Inherits directly from tsVecs. Methods and arguments are the same, the changes are only to how vecs is defined.

        Parameters
        ----------
        tod: tod object
             tod corresponding to the timestream
        order: int
             order of the legandre polynomial to fit to the tod
        """

        self.fname = tod.info["fname"]

        dims = tod.get_data_dims()
        self.ndata = dims[1]
        self.order = order
        self.ndet = dims[0]
        xvec = np.linspace(-1, 1, self.ndata)
        self.vecs = (np.polynomial.legendre.legvander(xvec, order).T).copy()
        self.nvec = self.vecs.shape[0]
        self.params = np.zeros([self.nvec, self.ndet])


class tsBowl(tsVecs):
    """
    Class for fitting legandre polynomials to tod elevation.

    Mustang 2 has observed a consistent problem with gradients in its maps, refered to as bowling. Current thinking is that the ultimate source of the bowling is elevation depended gradients due to the atmosphere. As the sky revolves around a target thru the course of a night, this elevation gradient becomes a radial one. Previous attempts to remove this have fit polynomials to the telescope elevation and subtracted them from the tods, which has been moderately successful. This was done outside the minkasi framework; this class implements that method within the framework.

    Attributes
    ----------
    fname: str
        name of the tod
    tod: tod object
        tods corresponding to the timestream
    vecs: n_data x nvec matrix
        Legandre polynomials to which fit parameters are applied.
    ndet: int
        number of detectors in timestream
    nvec: int
        number of fit polynomials
    params: np.array, float, nvec x ndet
        fit parameters for the vecs.
    """

    def __init__(self, tod, order=3):
        """
        Inherits directly from tsVecs. Methods and arguments are the same, the changes are only to how vecs is defined.

        Parameters
        ----------
        tod: tod object
             tod corresponding to the timestream
        order: int
             order of the legandre polynomial to fit to the tod elevation
        """

        self.fname = tod.info["fname"]
        self.order = order

        dims = tod.get_data_dims()
        self.ndet = dims[0]
        self.ndata = dims[1]

        try:
            # Apix is the elevation relative to the source. Try to just load it first incase its already computed, otherwise compute it then set it
            self.apix = tod.info["apix"]
        except KeyError:
            tod.set_apix()
            self.apix = tod.info["apix"]

        # Normalize apix to run from -1 to 1, to preserve linear independce of leg polys
        self.apix /= np.max(np.abs(self.apix), axis=0)
        # TODO: swap legvander to legval
        # Array(len(apix), order) of the legander polynomials evaluated at self.apix
        self.vecs = (np.polynomial.legendre.legvander(self.apix, order)).copy()
        self.nvec = self.vecs.shape[-1]

        # Parameters c_ij for the legandre polynomials
        self.params = np.zeros([self.ndet, self.nvec])

    def map2tod(self, tod, mat=None, do_add=True, do_omp=False):
        """
        Given parameters and vecs, compute the corresponding tod.

        Given Am = for A the vecs, m the parameters, and d the tod data, return the d corresponding to the specified A and m. This will try to use the jit compiled map2todbowl if it is available, else it will fall back to a slower routine.

        Parameters
        ----------
        tod: tod object
            The tod, needed for getting the data if to_add is True
        mat: tod data object, optional
            d, the data corresponding to the tod, only used if to_add is True. If not speicified the data is taken from tod
        do_add: bool, optional, default = True
            If true, adds the resulting tod data to the existing tod data. If false, overwrites the tod data.
        do_omp : bool, optional
            Dummy variable to match parameters of other map2tod

        Returns
        -------
        No returns

        Side effects
        ------------
        Updates tod data with the values infered from params and vecs.
        """
        if mat is None:
            mat = tod.get_data()
        if do_add:
            mat[:] = mat[:] + map2todbowl(self.vecs, self.params)
        else:
            mat[:] = map2todbowl(self.vecs, self.params)

    def tod2map(self, tod, mat=None, do_add=True, do_omp=False):
        """
        Given legandre vecs and TOD data, computes the corresponding parameters.

        tod2map is the transpose of the linear transformation map2tod. This will use the jit compiled tod2mapbowl if available, else it will fall back on a slower routine.

        Parameters
        ----------
        tod : 'minkasi.TOD'
            minkasi tod object
        mat : np.array, optional
            data from tod. If not specified, mat is taken from passed tod
        do_add : bool
            If true, adds resulting param to existing param. If false, overwrites it.
        do_omp : bool
            Dummy variable to match parameters of other tod2map

        Returns
        -------
        No returns

        Side effects
        ------------
        Updates params with the values infered from mat
        """

        if mat is None:
            mat = tod.get_data()
        if do_add:
            self.params = self.params + tod2mapbowl(self.vecs, mat)
        else:
            self.paras = tod2mapbowl(self.vecs, mat)

    def fit_apix(self, tod):
        if tod.info["fname"] != self.fname:
            print(
                "Error: bowling fitting can only be performed with the tod used to initialize this timestream; {}".format(
                    tod.info["fname"]
                )
            )
            return
        for i in range(self.ndet):
            self.params[i, ...] = np.polynomial.legendre.legfit(
                self.apix[i], tod.info["dat_calib"][i] - self.drift[i], self.order
            )


def partition_interval(start, stop, seg_len=100, round_up=False):
    # print('partitioning ',start,stop,seg_len)
    # make sure we behave correctly if the interval is shorter than the desired segment
    if (stop - start) <= seg_len:
        return np.asarray([start, stop], dtype="int")
    nseg = (stop - start) // seg_len
    if nseg * seg_len < (stop - start):
        if round_up:
            nseg = nseg + 1
    seg_len = (stop - start) // nseg
    nextra = (stop - start) - seg_len * nseg
    inds = np.arange(start, stop + 1, seg_len)
    if nextra > 0:
        vec = np.zeros(len(inds), dtype="int")
        vec[1 : nextra + 1] = 1
        vec = np.cumsum(vec)
        inds = inds + vec
    return inds


def split_partitioned_vec(start, stop, breaks=[], seg_len=100):
    if len(breaks) == 0:
        return partition_interval(start, stop, seg_len)
    if breaks[0] == start:
        breaks = breaks[1:]
        if len(breaks) == 0:
            return partition_interval(start, stop, seg_len)
    if breaks[-1] == stop:
        breaks = breaks[:-1]
        if len(breaks) == 0:
            return partition_interval(start, stop, seg_len)
    breaks = np.hstack([start, breaks, stop])
    nseg = len(breaks) - 1
    segs = [None] * (nseg)
    for i in range(nseg):
        inds = partition_interval(breaks[i], breaks[i + 1], seg_len)
        if i < (nseg - 1):
            inds = inds[:-1]
        segs[i] = inds
    segs = np.hstack(segs)
    return segs


# breaks,stop,start=0,seg_len=100)
class tsStripes(tsGeneric):
    def __init__(self, tod, seg_len=500, do_slope=False, tthresh=10):
        dims = tod.get_data_dims()
        tvec = tod.get_tvec()
        dt = np.median(np.diff(tvec))
        splits = np.where(np.abs(np.diff(tvec)) > tthresh * dt)[0]

        dims = tod.get_data_dims()
        inds = split_partitioned_vec(0, dims[1], splits, seg_len)

        self.inds = inds
        self.splits = splits
        self.nseg = len(self.inds) - 1
        self.params = np.zeros([dims[0], self.nseg])

    def tod2map(self, tod, dat=None, do_add=True, do_omp=False):
        if dat is None:
            print("need dat in tod2map destriper")
            return
        tod2map_destriped(dat, self.params, self.inds, do_add)

    def map2tod(self, tod, dat=None, do_add=True, do_omp=False):
        if dat is None:
            print("need dat in map2tod destriper")
            return
        map2tod_destriped(dat, self.params, self.inds, do_add)

    def copy(self):
        return copy.deepcopy(self)

    def set_prior_from_corr(self, corrvec, thresh=0.5):
        assert corrvec.shape[0] == self.params.shape[0]
        n = self.params.shape[1]
        corrvec = corrvec[:, :n].copy()
        corrft = fft.fft_r2r(corrvec)
        if thresh > 0:
            for i in range(corrft.shape[0]):
                tt = thresh * np.median(corrft[i, :])
                ind = corrft[i, :] < tt
                corrft[i, ind] = tt
        self.params = 1.0 / corrft / (2 * (n - 1))

    def apply_prior(self, x, Ax):
        xft = fft.fft_r2r(x.params)
        Ax.params = Ax.params + fft.fft_r2r(xft * self.params)


class tsBinnedAz(tsGeneric):
    def __init__(self, tod, lims=(0, 2 * np.pi), nbin=360):
        # print('nbin is',nbin)
        ndet = tod.get_ndet()
        self.params = np.zeros([ndet, nbin])
        self.lims = (lims[0], lims[1])
        self.nbin = nbin

    def map2tod(self, tod, dat=None, do_add=True, do_omp=False):
        if dat is None:
            dat = tod.get_data()
        map2tod_binned_det(
            dat, self.params, tod.info["az"], self.lims, self.nbin, do_add
        )

    def tod2map(self, tod, dat=None, do_add=True, do_omp=False):
        if dat is None:
            dat = tod.get_data()
        tod2map_binned_det(
            dat, self.params, tod.info["az"], self.lims, self.nbin, do_add
        )


class tsBinnedAzShared(tsGeneric):
    # """class to have az shared amongst TODs (say, if you think the ground is constant for a while)"""
    def __init__(self, ndet=2, lims=(0, 2 * np.pi), nbin=360):
        self.params = np.zeros([ndet, nbin])
        self.lims = (lims[0], lims[1])
        self.nbin = nbin

    def map2tod(self, tod, dat=None, do_add=True, do_omp=False):
        if dat is None:
            dat = tod.get_data()
        map2tod_binned_det(
            dat, self.params, tod.info["az"], self.lims, self.nbin, do_add
        )

    def tod2map(self, tod, dat=None, do_add=True, do_omp=False):
        if dat is None:
            dat = tod.get_data()
        print("nbin is", self.nbin)
        print(self.params.dtype)
        tod2map_binned_det(
            dat, self.params, tod.info["az"], self.lims, self.nbin, do_add
        )


class tsDetAz(tsGeneric):
    def __init__(self, tod, npoly=4):
        if isinstance(
            tod, tsDetAz
        ):  # we're starting a new instance from an old one, e.g. from copy
            self.fname = tod.fname
            self.az = tod.az
            self.azmin = tod.azmin
            self.azmax = tod.azmax
            self.npoly = tod.npoly
            self.ndet = tod.ndet
        else:
            self.fname = tod.info["fname"]
            self.az = tod.info["AZ"]
            self.azmin = np.min(self.az)
            self.azmax = np.max(self.az)
            self.npoly = npoly
            # self.ndet=tod.info['dat_calib'].shape[0]
            self.ndet = tod.get_ndet()
        # self.params=np.zeros([self.ndet,self.npoly])
        self.params = np.zeros([self.ndet, self.npoly - 1])

    def _get_polys(self):
        polys = np.zeros([self.npoly, len(self.az)])
        polys[0, :] = 1.0
        az_scale = (self.az - self.azmin) / (self.azmax - self.azmin) * 2.0 - 1.0
        if self.npoly > 1:
            polys[1, :] = az_scale
        for i in range(2, self.npoly):
            polys[i, :] = 2 * az_scale * polys[i - 1, :] - polys[i - 2, :]
        polys = polys[1:, :].copy()
        return polys

    def map2tod(self, tod, dat=None, do_add=True, do_omp=False):
        if dat is None:
            # dat=tod.info['dat_calib']
            dat = tod.get_data()
        if do_add:
            dat[:] = dat[:] + np.dot(self.params, self._get_polys())
        else:
            dat[:] = np.dot(self.params, self._get_polys())

    def tod2map(self, tod, dat=None, do_add=True, do_omp=False):
        if dat is None:
            # dat=tod.info['dat_calib']
            dat = tod.get_data()
        if do_add:
            # print("params shape is ",self.params.shape)
            # self.params[:]=self.params[:]+np.dot(self._get_polys(),dat)
            self.params[:] = self.params[:] + np.dot(dat, self._get_polys().T)
        else:
            # self.params[:]=np.dot(self._get_polys(),dat)
            self.params[:] = np.dot(dat, self._get_polys().T)


class tsAirmass:
    def __init__(self, tod=None, order=3):
        if tod is None:
            self.sz = np.asarray([0, 0], dtype="int")
            self.params = np.zeros(1)
            self.fname = ""
            self.order = 0
            self.airmass = None
        else:
            # self.sz=tod.info['dat_calib'].shape
            self.sz = tod.get_data_dims()
            self.fname = tod.info["fname"]
            self.order = order
            self.params = np.zeros(order)
            if not ("apix" in tod.info.keys()):
                # tod.info['apix']=scaled_airmass_from_el(tod.info['elev'])
                self.airmass = scaled_airmass_from_el(tod.info["elev"])
            else:
                self.airmass = tod.info["apix"]

    def copy(self, copyMat=False):
        cp = tsAirmass()
        cp.sz = self.sz
        cp.params = self.params.copy()
        cp.fname = self.fname
        cp.order = self.order
        if copyMat:
            cp.airmass = self.airmass.copy()
        else:
            cp.airmass = (
                self.airmass
            )  # since this shouldn't change, use a pointer to not blow up RAM
        return cp

    def clear(self):
        self.params[:] = 0.0

    def dot(self, ts):
        return np.sum(self.params * ts.params)

    def axpy(self, ts, a):
        self.params = self.params + a * ts.params

    def _get_current_legmat(self):
        x = np.linspace(-1, 1, self.sz[1])
        m1 = np.polynomial.legendre.legvander(x, self.order)
        return m1

    def _get_current_model(self):
        x = np.linspace(-1, 1, self.sz[1])
        m1 = self._get_current_legmat()
        poly = np.dot(m1, self.params)
        mat = np.repeat([poly], self.sz[0], axis=0)
        mat = mat * self.airmass
        return mat

    def tod2map(self, tod, dat, do_add=True, do_omp=False):
        tmp = np.zeros(self.order)
        for i in range(self.order):
            # tmp[i]=np.sum(tod.info['apix']**(i+1)*dat)
            tmp[i] = np.sum(self.airmass ** (i + 1) * dat)
        if do_add:
            self.params[:] = self.params[:] + tmp
        else:
            self.params[:] = tmp
        # poly=self._get_current_legmat()
        # vec=np.sum(dat*self.airmass,axis=0)
        # atd=np.dot(vec,poly)
        # if do_add:
        #    self.params[:]=self.params[:]+atd
        # else:
        #    self.params[:]=atd

    def map2tod(self, tod, dat, do_add=True, do_omp=False):
        mat = 0.0
        for i in range(self.order):
            # mat=mat+self.params[i]*tod.info['apix']**(i+1)
            mat = mat + self.params[i] * self.airmass ** (i + 1)

        # mat=self._get_current_model()
        if do_add:
            dat[:] = dat[:] + mat
        else:
            dat[:] = mat

    def __mul__(self, to_mul):
        tt = self.copy()
        tt.params = self.params * to_mul.params
        return tt

    def write(self, fname=None):
        pass


class tsCommon:
    def __init__(self, tod=None, *args, **kwargs):
        if tod is None:
            self.sz = np.asarray([0, 0], dtype="int")
            self.params = np.zeros(1)
            self.fname = ""
        else:
            # self.sz=tod.info['dat_calib'].shape
            self.sz = tod.get_data_dims()
            self.params = np.zeros(self.sz[1])
            self.fname = tod.info["fname"]

    def copy(self):
        cp = tsCommon()
        try:
            cp.sz = self.sz.copy()
        except:  # if the size doesn't have a copy function, then it's probably a number you can just assign
            cp.sz = self.sz
            cp.fname = self.fname
            cp.params = self.params.copy()
            return cp

    def clear(self):
        self.params[:] = 0.0

    def dot(self, common=None):
        if common is None:
            return np.dot(self.params, self.params)
        else:
            return np.dot(self.params, common.params)

    def axpy(self, common, a):
        self.params = self.params + a * common.params

    def tod2map(self, tod, dat, do_add=True, do_omp=False):
        # assert(self.fname==tod.info['fname']
        nm = tod.info["fname"]
        if do_add == False:
            self.clear()
        self.params[:] = self.params[:] + np.sum(dat, axis=0)

    def map2tod(self, tod, dat, do_add=True, do_omp=True):
        nm = tod.info["fname"]
        dat[:] = dat[:] + np.repeat([self.params], dat.shape[0], axis=0)

    def write(self, fname=None):
        pass

    def __mul__(self, to_mul):
        tt = self.copy()
        tt.params = self.params * to_mul.params
        return tt


class tsCalib:
    def __init__(self, tod=None, model=None):
        if tod is None:
            self.sz = 1
            self.params = np.zeros(1)
            self.fname = ""
            self.pred = None
        else:
            # self.sz=tod.info['dat_calib'].shape[0]
            self.sz = tod.get_ndet()
            self.params = np.zeros(self.sz)
            self.pred = model[tod.info["fname"]].copy()
            self.fname = tod.info["fname"]

    def copy(self):
        cp = tsCalib()
        cp.sz = self.sz
        cp.params = self.params.copy()
        cp.fname = self.fname
        cp.pred = self.pred
        return cp

    def clear(self):
        self.params[:] = 0

    def dot(self, other=None):
        if other is None:
            return np.dot(self.params, self.params)
        else:
            return np.dot(self.params, other.params)

    def axpy(self, common, a):
        self.params = self.params + a * common.params

    def tod2map(self, tod, dat, do_add=True, do_omp=False):
        if do_add == False:
            self.clear()
        if self.pred.ndim == 1:
            self.params[:] = self.params[:] + np.dot(dat, self.pred)
        else:
            self.params[:] = self.params[:] + np.sum(dat * self.pred, axis=1)

    def map2tod(self, tod, dat, do_add=True, do_omp=False):
        if do_add == False:
            dat[:] = 0
        if self.pred.ndim == 1:
            dat[:] = dat[:] + np.outer(self.params, self.pred)
        else:
            dat[:] = dat[:] + (self.pred.transpose() * self.params).transpose()

    def write(self, fname=None):
        pass

    def __mul__(self, to_mul):
        tt = self.copy()
        tt.params = self.params * to_mul.params
        return tt


class tsModel:
    def __init__(self, todvec=None, modelclass=None, *args, **kwargs):
        self.data = {}
        if todvec is None:
            return
        for tod in todvec.tods:
            nm = tod.info["fname"]
            self.data[nm] = modelclass(tod, *args, **kwargs)

    def copy(self):
        new_tsModel = tsModel()
        for nm in self.data.keys():
            new_tsModel.data[nm] = self.data[nm].copy()
        return new_tsModel

    def tod2map(self, tod, dat, do_add=True, do_omp=False):
        nm = tod.info["fname"]
        if do_add == False:
            self.clear()
        self.data[nm].tod2map(tod, dat, do_add, do_omp)

    def map2tod(self, tod, dat, do_add=True, do_omp=True):
        nm = tod.info["fname"]
        if do_add == False:
            dat[:] = 0.0
        self.data[nm].map2tod(tod, dat, do_add, do_omp)

    def apply_prior(self, x, Ax):
        for nm in self.data.keys():
            self.data[nm].apply_prior(x.data[nm], Ax.data[nm])

    def dot(self, tsmodels=None):
        tot = 0.0
        for nm in self.data.keys():
            if tsmodels is None:
                tot = tot + self.data[nm].dot(self.data[nm])
            else:
                # if tsmodels.data.has_key(nm):
                if nm in tsmodels.data:
                    tot = tot + self.data[nm].dot(tsmodels.data[nm])
                else:
                    print("error in tsModel.dot - missing key ", nm)
                    assert 1 == 0  # pretty sure we want to crash if missing names
        if have_mpi:
            tot = comm.allreduce(tot)
        return tot

    def clear(self):
        for nm in self.data.keys():
            self.data[nm].clear()

    def axpy(self, tsmodel, a):
        for nm in self.data.keys():
            self.data[nm].axpy(tsmodel.data[nm], a)

    def __mul__(
        self, tsmodel
    ):  # this is used in preconditioning - need to fix if ts-based preconditioning is desired
        tt = self.copy()
        for nm in self.data.keys():
            tt.data[nm] = self.data[nm] * tsmodel.data[nm]
        return tt
        # for nm in tt.data.keys():
        #    tt.params[nm]=tt.params[nm]*tsmodel.params[nm]

    def mpi_reduce(self):
        pass

    def get_caches(self):
        for nm in self.data.keys():
            try:
                self.data[nm].get_caches()
            except:
                pass

    def clear_caches(self):
        for nm in self.data.keys():
            try:
                self.data[nm].clear_caches()
            except:
                pass

    def mpi_reduce(self):
        pass


class tsMultiModel(tsModel):
    """A class to hold timestream models that are shared between groups of TODs."""

    def __init__(
        self,
        todvec=None,
        todtags=None,
        modelclass=None,
        tag="ts_multi_model",
        *args,
        **kwargs
    ):
        self.data = {}
        self.tag = tag
        if not (todtags is None):
            alltags = comm.allgather(todtags)
            alltags = np.hstack(alltags)
            alltags = np.unique(alltags)
            if not (modelclass is None):
                for mytag in alltags:
                    self.data[mytag] = modelclass(*args, **kwargs)
            if not (todvec is None):
                for i, tod in enumerate(todvec.tods):
                    tod.info[tag] = todtags[i]

    def copy(self):
        return copy.deepcopy(self)

    def tod2map(self, tod, dat, do_add=True, do_omp=False):
        self.data[tod.info[self.tag]].tod2map(tod, dat, do_add, do_omp)

    def map2tod(self, tod, dat, do_add=True, do_omp=False):
        self.data[tod.info[self.tag]].map2tod(tod, dat, do_add, do_omp)

    def dot(self, tsmodels=None):
        tot = 0.0
        for nm in self.data.keys():
            if tsmodels is None:
                tot = tot + self.data[nm].dot(self.data[nm])
            else:
                if nm in tsmodels.data:
                    tot = tot + self.data[nm].dot(tsmodels.data[nm])
                else:
                    print("error in tsMultiModel.dot - missing key ", nm)
                    assert 1 == 0
        return tot
