import sys

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfinv

from ..fitting.power_spectrum import fit_ts_ps
from ..maps import MapType
from ..maps.utils import read_fits_map
from ..tods import Tod, TodVec
from ..tools import fft
from ..tools.array_ops import axpy_in_place, scale_matrix_by_vector
from ..tools.smooth import smooth_many_vecs
from .tod2map import make_hits

if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable


def get_grad_mask_2d(
    map: MapType,
    todvec: TodVec | None = None,
    thresh: float = 4.0,
    noisemap: MapType | None = None,
    hitsmap: MapType | None = None,
) -> NDArray[np.floating]:
    """
    Make a mask that has an estimate of the gradient within a pixel.
    Look at the  rough expected noise to get an idea of which gradients
    are substantially larger than the map noise.

    Parameters
    ----------
    map : MapType
        The map to get gradient mask for.
    todvec : TodVec | None, default: None
        TODs to use to make hits and noise maps.
        If you already have these maps to pass in this is unused and can be None.
    thresh : float, default: 4.0
        Threshold at which to flag gradients.
        This is in units of noise so the flag is: grad[i, j] > thresh*noise[i, j].
        Flagged gradients are set to 0 in the output.
    noisemap : MapType | None, default: None
        The noise map, if None will be made from todvec.
    hitsmap : MapType | None, default: None
        The hits map, if None will be made from todvec.

    Returns
    -------
    grad : NDArray[np.floating]
        The per pixel gradient of the map.
        Gradients that are too high are set to 0 (see thresh for details).
    """
    if noisemap is None:
        noisemap = make_hits(todvec, map, do_weights=True)
        noisemap.invert()
        noisemap.map = np.sqrt(noisemap.map)
    if hitsmap is None:
        hitsmap = make_hits(todvec, map, do_weights=False)

    grad = (map.map - np.roll(map.map, 1, axis=0)) ** 2
    grad += (map.map - np.roll(map.map, -1, axis=0)) ** 2
    grad += (map.map - np.roll(map.map, -1, axis=1)) ** 2
    grad += (map.map - np.roll(map.map, 1, axis=1)) ** 2
    grad = np.sqrt(0.25 * grad)

    # find the typical timestream noise in a pixel, which should be the noise map times sqrt(hits)
    hitsmask = hitsmap.map > 0
    noise = noisemap.map.copy()
    noise[hitsmask] *= np.sqrt(hitsmap.map[hitsmask])

    mask = grad > (thresh * noise)
    frac = 1.0 * np.sum(mask) / mask.size
    print("Cutting " + repr(frac * 100) + "% of map pixels in get_grad_mask_2d.")
    grad[np.logical_not(mask)] = 0
    return grad


@runtime_checkable
class NoiseModelType(Protocol):
    def __init__(self, dat: NDArray[np.floating], /):
        pass

    def apply_noise(self, dat: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.empty(0)


@runtime_checkable
class WithDetWeights(NoiseModelType):
    def get_det_weights(self) -> NDArray[np.floating]:
        return np.empty(0)


class MapNoiseWhite:
    """
    Simple map space noise model with only white noise.

    Attributes
    ----------
    ivar : NDArray[np.floating]
        The inverse variance map.
    """

    def __init__(self, ivar_map: str, isinv: bool = True, nfac: float = 1.0):
        """
        Initialize an instance of MapNoiseWhite.
        Pretty just sets up and stores ivar.

        Parameters
        ----------
        ivar_map : str
            The path to the inverse variance map's FITS file.
        isinv : bool, default: True
            If True tho map at ivar_map is actually inverted.
            If False, its just variance and will be inverted by this function.
        nfac : float, default: 1.0
            Factor to scale the inverse variance map by.
        """
        self.ivar = read_fits_map(ivar_map)
        if not (isinv):
            mask = self.ivar > 0
            self.ivar[mask] = 1.0 / self.ivar[mask]
        self.ivar = self.ivar * nfac

    def apply_noise(self, map: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Apply the inverse variance to a map.

        Parameters
        ----------
        map : NDArray[np.floating]
            The map to apply the noise model to.
            Note that this is an array and not an instance of SkyMap.
        """
        return map * self.ivar


class NoiseWhite:
    """
    Simple noise model with just white noise.

    Attributes
    ----------
    sigs : NDArray[np.floating]
        Ratio between the median absolute deviation of the diff and sigma.
    weights : NDArray[np.floating]
        Per detector weight.
    """

    def __init__(self, dat: NDArray[np.floating]):
        """
        Initialize an instance of NoiseWhite.
        This is where sigs and weights are computed.
        """
        fac = erfinv(0.5) * 2
        sigs = np.median(np.abs(np.diff(dat, axis=1)), axis=1) / fac
        self.sigs: NDArray[np.floating] = sigs
        self.weights: NDArray[np.floating] = 1 / sigs**2

    def apply_noise(self, dat: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Apply weight to each detector.

        Parameters
        ----------
        dat : NDArray[np.floating]
            The TOD data to apply noise to, should be (ndet, ndata).
            Will also be modified in place.

        Returns
        -------
        dat : NDArray[np.floating]
            The TOD data with noise applied.
        """
        assert dat.shape[0] == len(self.weights)
        dat *= self.weights[..., np.newaxis]
        return dat


class NoiseWhiteNotch:
    """
    Noise model with white noise and a notch filter.

    Attributes
    ----------
    sigs : NDArray[np.floating]
        Ratio between the median absolute deviation of the diff and sigma.
    weights : NDArray[np.floating]
        Per detector weight.
    istart : int
        Index corresponding to the start of the notch in the FFT.
    istop : int
        Index corresponding to the end of the notch in the FFT.
    """

    def __init__(self, dat: NDArray[np.floating], numin: float, numax: float, tod: Tod):
        """
        Initialize an instance of NoiseWhiteNotch.
        This is where sigs and weights are computed and the notch indices are figured out.

        Parameters
        ----------
        dat : NDArray[np.floating]
            The TOD data to make the noise model for.
        numin : float
            Starting freq of noise model.
        numax : float
            Stopping freq of noise model.
        tod : Tod
            The TOD to make thi noise model for.
            Time information from this is used to figure out freqs.
        """
        fac = erfinv(0.5) * 2
        sigs = np.median(np.abs(np.diff(dat, axis=1)), axis=1) / fac
        self.sigs: NDArray[np.floating] = sigs
        self.weights: NDArray[np.floating] = 1 / sigs**2
        self.weights = self.weights / (
            2 * (dat.shape[1] - 1)
        )  # fold in fft normalization to the weights

        tvec = tod.get_tvec()
        dt = np.median(np.diff(tvec))
        tlen = tvec[-1] - tvec[0]
        dnu = 1.0 / (2 * tlen - dt)
        self.istart: int = int(np.floor(numin / dnu))
        self.istop: int = int(np.ceil(numax / dnu)) + 1

    def apply_noise(self, dat: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Notch filter data and then apply weight to each detector.

        Parameters
        ----------
        dat : NDArray[np.floating]
            The TOD data to apply noise to, should be (ndet, ndata).
            Will also be modified in place.

        Returns
        -------
        dat : NDArray[np.floating]
            The TOD data with noise applied.
        """
        assert dat.shape[0] == len(self.weights)
        datft = fft.fft_r2r(dat)
        datft[:, self.istart : self.istop] = 0
        dat = fft.fft_r2r(datft)

        dat *= self.weights[..., np.newaxis]
        return dat


class NoiseCMWhite:
    """
    Noise model that computes the common mode with the SVD
    and uses the common mode subtracted data to compute white noise.

    Attributes
    ----------
    ndet : int
        The number of detectors.
    v : NDArray[np.floating]
        The singular vector corresponding to the common mode.
    mywt : NDArray[np.floating]
        The per detector white noise level.
    """

    def __init__(self, dat: NDArray):
        """
        Initialize an instance ot NoiseCMWhite.
        This is where the SVD is taken and v and mywt are set.

        Parameters
        ----------
        dat : NDArray[np.floating]
            The TOD data to make the noise model for.
        """
        print("setting up noise cm white")
        u, s, v = np.linalg.svd(dat, False)
        self.ndet: int = len(s)
        ind = np.argmax(s)

        self.v: NDArray[np.floating] = np.zeros(self.ndet)
        self.v[:] = u[:, ind]

        pred = np.outer(self.v * s[ind], v[ind, :])
        dat_clean = dat - pred
        myvar = np.std(dat_clean, 1) ** 2
        self.mywt: NDArray[np.floating] = 1.0 / myvar

    def apply_noise(
        self, dat: NDArray[np.floating], dd: NDArray[np.floating] | None = None
    ) -> NDArray[np.floating]:
        """
        Apply the noise model.
        Seems like it removes the CM and then applies the white noise weights.

        Parameters
        ----------
        dat : NDArray[np.floating]
            The TOD data to apply the model to, should be (ndet, ndata).
        dd : NDArray[np.floating] | None, default: None
            Array to store data with model applied.
            If provided it will be modified in place and should be the same shape as dat.
            If you just want to have this data returned pass in None.

        Returns
        -------
        dd : NDArray[np.floating]
            The data with the noise model applied.
        """
        if dd is None:
            dd = np.empty(dat.shape)
        elif dd.shape != dat.shape:
            print("Provided dd has incorrect shape, initializing a new one")
            dd = np.empty(dat.shape)
        dd = np.array(dd)

        mat = np.dot(self.v, np.diag(self.mywt))
        lhs = np.dot(self.v, mat.T)
        rhs = np.dot(mat, dat)
        if isinstance(lhs, np.ndarray):
            cm = np.dot(np.linalg.inv(lhs), rhs)
        else:
            cm = rhs / lhs

        if have_numba:
            np.outer(-self.v, cm, dd)
            axpy_in_place(dd, dat)
            scale_matrix_by_vector(dd, self.mywt)
        else:
            dd = dat - np.outer(self.v, cm)
            tmp = np.repeat([self.mywt], len(cm), axis=0).T
            dd *= tmp

        return dd

    def get_det_weights(self) -> NDArray[np.floating]:
        """
        Get a copy of the detector weights.

        Returns
        -------
        weights : NDArray[np.floating]
            The detector weights.
        """
        return self.mywt.copy()


class NoiseBinnedDet:
    """
    Noise model where the power spectra of each detector is
    binned into the provided freqency bins.

    Attributes
    ----------
    ndet : int
        The number of detectors.
    ndata : int
        The number of samples per detector.
    nn : int
        The number of frequencies in the FFTs.
    bins : NDArray[np.int64]
        Frequency bins in terms of the FFT indices.
    nbin : int
        The number of freqency bins.
    det_ps : NDArray[np.floating]
        The bin averaged power spectrum of each detector.
        Has shape (ndet, nbin).
    """

    def __init__(
        self, dat: NDArray[np.floating], dt: float, freqs: NDArray[np.floating]
    ):
        """
        Initialize an instance of NoiseBinnedDet.
        This will compute the freq bins and the binned spectra.

        Parameters
        ----------
        dat : NDArray[np.floating]
            The TOD data to make the noise model for.
        dt : float
            The time between samples in the TOD.
        freqs : NDArray[np.floating]
            The edges of the freqency bins.
        """
        ndet = dat.shape[0]
        ndata = dat.shape[1]
        nn = 2 * (ndata - 1)
        dnu = 1 / (nn * dt)
        self.ndet: int = ndet
        self.ndata: int = ndata
        self.nn: int = nn

        bins = np.asarray(freqs / dnu, dtype="int")
        bins = bins[bins < ndata]
        bins = np.hstack([bins, ndata])
        if bins[0] > 0:
            bins = np.hstack([0, bins])
        elif bins[0] < 0:
            bins[0] = 0
        self.bins: NDArray[np.int64] = bins
        self.nbin: int = len(bins) - 1

        det_ps = np.zeros([ndet, self.nbin])
        datft = fft.fft_r2r(dat)
        for i in range(self.nbin):
            det_ps[:, i] = 1.0 / np.mean(datft[:, bins[i] : bins[i + 1]] ** 2, axis=1)
        self.det_ps: NDArray[np.floating] = det_ps

    def apply_noise(self, dat: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Apply noise by FFTing the input and multiplying in each bin
        by the appropriate values from det_ps.

        Parameters
        ----------
        dat : NDArray[np.floating]
            The TOD data to apply the model to, should be (ndet, ndata).

        Returns
        -------
        dd : NDArray[np.floating]
            The data with the noise model applied.
        """
        datft = fft.fft_r2r(dat)
        for i in range(self.nbin):
            datft[:, self.bins[i] : self.bins[i + 1]] = datft[
                :, self.bins[i] : self.bins[i + 1]
            ] * np.outer(self.det_ps[:, i], np.ones(self.bins[i + 1] - self.bins[i]))
        dd = fft.fft_r2r(datft)
        dd[:, 0] = 0.5 * dd[:, 0]
        dd[:, -1] = 0.5 * dd[:, -1]
        return dd


class NoiseBinnedEig:
    """
    Similar to NoiseBinnedDet but use the covarience to
    seperate out the strongest modes and compute
    and bin it's spectra as well as the residual's.

    Attributes
    ----------
    ndet : int
        The number of detectors.
    ndata : int
        The number of samples per detector.
    nn : int
        The number of frequencies in the FFTs.
    bins : NDArray[np.int64]
        Frequency bins in terms of the FFT indices.
    nbin : int
        The number of freqency bins.
    det_ps : NDArray[np.floating]
        The bin averaged power spectrum of each detector.
        This is with the strongest modes subtracted.
        Has shape (ndet, nbin).
    mode_ps : NDArray[np.floating]
        The bin averaged power spectrum of the strongest modes.
        Has shape (nmode, nbin).
    """

    def __init__(self, dat, dt, freqs, thresh=5.0):
        """
        Initialize an instance of NoiseBinnedEig.
        This computes the frequency bins and the strongest
        modes and then stores the binned spectra.

        Parameters
        ----------
        dat : NDArray[np.floating]
            The TOD data to make the noise model for.
        dt : float
            The time between samples in the TOD.
        freqs : NDArray[np.floating]
            The edges of the freqency bins.
        thresh : float, default: 5.0
            The threshold for what eigenvectors to keep.
            The ones that are kept have an eigenmodes which are
            thresh^2 times larger than the mean eigenvalue.
        """
        ndet = dat.shape[0]
        ndata = dat.shape[1]
        nn = 2 * (ndata - 1)
        dnu = 1 / (nn * dt)
        print("dnu is " + repr(dnu))
        self.ndet: int = ndet
        self.ndata: int = ndata
        self.nn: int = nn
        nn = 2 * (ndata - 1)

        mycov = np.dot(dat, dat.T)
        mycov = 0.5 * (mycov + mycov.T)
        ee, vv = np.linalg.eig(mycov)
        mask = ee > thresh * thresh * np.median(ee)
        vecs = vv[:, mask]
        mode = np.dot(vecs.T, dat)
        resid = dat - np.dot(vv[:, mask], mode)

        bins = np.asarray(freqs / dnu, dtype="int")
        bins = bins[bins < ndata]
        bins = np.hstack([bins, ndata])
        if bins[0] > 0:
            bins = np.hstack([0, bins])
        if bins[0] < 0:
            bins[0] = 0
        self.bins: NDArray[np.int64] = bins
        self.nbin: int = len(bins) - 1

        nmode = mode.shape[0]
        det_ps = np.zeros([ndet, self.nbin])
        mode_ps = np.zeros([nmode, self.nbin])
        residft = fft.fft_r2r(resid)
        modeft = fft.fft_r2r(mode)
        for i in range(self.nbin):
            det_ps[:, i] = 1.0 / np.mean(residft[:, bins[i] : bins[i + 1]] ** 2, axis=1)
            mode_ps[:, i] = 1.0 / np.mean(modeft[:, bins[i] : bins[i + 1]] ** 2, axis=1)
        self.modes = vecs.copy()
        if not (np.all(np.isfinite(det_ps))):
            print(
                "warning - have non-finite numbers in noise model.  This should not be unexpected."
            )
            det_ps[~np.isfinite(det_ps)] = 0.0
        self.det_ps: NDArray[np.floating] = det_ps
        self.mode_ps: NDArray[np.floating] = mode_ps

    def apply_noise(self, dat: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Apply noise by FFTing the input and applying mode_ps and det_ps.

        Parameters
        ----------
        dat : NDArray[np.floating]
            The TOD data to apply the model to, should be (ndet, ndata).

        Returns
        -------
        dd : NDArray[np.floating]
            The data with the noise model applied.
        """
        assert dat.shape[0] == self.ndet
        assert dat.shape[1] == self.ndata

        datft = fft.fft_r2r(dat)
        for i in range(self.nbin):
            n = self.bins[i + 1] - self.bins[i]

            tmp = self.modes * np.outer(self.det_ps[:, i], np.ones(self.modes.shape[1]))
            mat = np.dot(self.modes.T, tmp)
            mat = mat + np.diag(self.mode_ps[:, i])
            mat_inv = np.linalg.inv(mat)

            Ax = datft[:, self.bins[i] : self.bins[i + 1]] * np.outer(
                self.det_ps[:, i], np.ones(n)
            )

            tmp = np.dot(self.modes.T, Ax)
            tmp = np.dot(mat_inv, tmp)
            tmp = np.dot(self.modes, tmp)
            tmp = Ax - tmp * np.outer(self.det_ps[:, i], np.ones(n))

            datft[:, self.bins[i] : self.bins[i + 1]] = tmp

        dd = fft.fft_r2r(datft)
        dd[:, 0] = 0.5 * dd[:, 0]
        dd[:, -1] = 0.5 * dd[:, -1]

        return dd


class NoiseSmoothedSVD:
    """
    Noise model that rotates the data into singular space
    and then computes smoothed spetra.

    This is the most commonly used noise model and generally performs well.

    Attributes
    ----------
    noisevec : NDArray[np.floating] | None
        The values used for prewhitening.
        If prewhiten is False this is None
    v : NDArray[np.floating]
        The right singular vectors.
    vT : NDArray[np.floating]
        The conjugate transpose of V/
    mywt : NDArray[np.floating]
        The smoothed spectra.
    """

    def __init__(
        self,
        dat_use: NDArray[np.floating],
        fwhm: float = 50.0,
        prewhiten: bool = False,
        fit_powlaw: bool = False,
        u_in: NDArray[np.floating] | None = None,
    ):
        """
        Initialize an instance of NoiseSmoothedSVD.
        Builds the noise model using the following steps:
        * Appy an optional white noise model
        * SVD the TOD
        * Apply the rotation from the right singular vectors
        * FFT the rotated data to get a spectra per detector
        * Smooth the spectra

        Parameters
        ----------
        dat_use : NDArray[np.floating]
            The TOD data to build the noise model for.
        fwhm : float, default: 50
            The FWHM of the smoothing kernal applied to the spectra.
            Not used in fit_powlaw is True.
            (Units are Hz? Samples?)
        prewhiten : bool, default: True
            If True a model similar to NoiseWhite will
            be applied before building this noise model.
        fit_powlaw : bool, default: False
            If True then instead of smoothing the spectra with a gaussian
            a power law will be fit instead.
        u_in : NDArray[np.floating] | None, default: None
            The left singular vectors of the TOD.
            If None they will be computed via the SVD in this function.
        """
        self.noisevec: NDArray[np.floating] | None
        if prewhiten:
            noisevec = np.median(np.abs(np.diff(dat_use, axis=1)), axis=1)
            dat_use = dat_use / (
                np.repeat([noisevec], dat_use.shape[1], axis=0).transpose()
            )
            self.noisevec = noisevec.copy()
        else:
            self.noisevec = None

        if u_in is None:
            u, s, _ = np.linalg.svd(dat_use, True)
            ndet = s.size
        else:
            u = u_in
            assert u.shape[0] == u.shape[1]
            ndet = u.shape[0]

        self.v: NDArray[np.floating] = np.zeros([ndet, ndet])
        self.v[:] = u.transpose()
        self.vT: NDArray[np.floating]
        if u_in is None:
            self.vT = self.v.T
        else:
            self.vT = np.linalg.inv(self.v)

        print("got svd")

        dat_rot = np.dot(self.v, dat_use)
        if fit_powlaw:
            spec_smooth = 0 * dat_rot
            for ind in range(ndet):
                _, __, C = fit_ts_ps(dat_rot[ind, :])
                spec_smooth[ind, 1:] = C
        else:
            dat_trans = fft.fft_r2r(dat_rot)
            spec_smooth = smooth_many_vecs(dat_trans**2, fwhm)
        spec_smooth[:, 1:] = 1.0 / spec_smooth[:, 1:]
        spec_smooth[:, 0] = 0
        self.mywt: NDArray[np.floating] = spec_smooth

    def apply_noise(self, dat: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Apply the noise model with the following steps:
        * Rotate the TOD into singular space
        * Take the FFT
        * Multiply the FFT by the noise model
        * Take the inverse FFT
        * Rotate back into the original space
        If prewhiten was True then that noise vector is also applied.

        Parameters
        ----------
        dat : NDArray[np.floating]
            The data to apply the noise model to.

        Returns
        -------
        dd : NDArray[np.floating]
            The data with the noise model applied.
        """
        noisemat = None
        if self.noisevec is not None:
            noisemat = np.repeat([self.noisevec], dat.shape[1], axis=0).transpose()
            dat = dat / noisemat
        dat_rot: NDArray[np.float64] = np.dot(self.v, dat)
        datft = fft.fft_r2r(dat_rot)
        nn = datft.shape[1]
        datft = datft * self.mywt[:, :nn]
        dat_rot = fft.fft_r2r(datft)
        dd = np.dot(self.vT, dat_rot)
        dd[:, 0] = 0.5 * dd[:, 0]
        dd[:, -1] = 0.5 * dd[:, -1]
        if noisemat is not None:
            dd = dd / noisemat
        return dd

    def apply_noise_wscratch(
        self,
        dat: NDArray[np.float64],
        tmp: NDArray[np.float64],
        tmp2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Same process as apply_noise but with some preallocated buffers provided.

        Parameters
        ----------
        dat : NDArray[np.float64]
            The data to apply the noise model to.
        tmp : NDArray[np.float64]
            The first buffer.
            Should be the same size as dat.
        tmp2 : NDArray[np.float64]
            The second buffer.
            Should be the same size as dat.

        Returns
        -------
        dd : NDArray[np.float64]
            The data with the noise model applied.
        """
        noisemat = None
        if self.noisevec is not None:
            noisemat = np.repeat([self.noisevec], dat.shape[1], axis=0).transpose()
            dat = dat / noisemat
        dat_rot = tmp
        dat_rot = np.dot(self.v, dat, dat_rot)
        dat = tmp2
        datft = dat
        datft = fft.fft_r2r(dat_rot, datft)
        nn = datft.shape[1]
        datft[:] = datft * self.mywt[:, :nn]
        dat_rot = tmp
        dat_rot = fft.fft_r2r(datft, dat_rot)
        dd = np.dot(self.vT, dat_rot, dat)
        dd[:, 0] = 0.5 * dat[:, 0]
        dd[:, -1] = 0.5 * dat[:, -1]
        if noisemat is not None:
            dd = dd / noisemat
        return dd

    def get_det_weights(self) -> NDArray[np.floating]:
        """
        Find the per-detector weights for use in making actual noise maps.

        Returns
        -------
        det_weights : NDArray[np.floating]
            The per detector weights.
        """
        mode_wt = np.sum(self.mywt, axis=1)
        tmp = np.dot(self.vT, np.dot(np.diag(mode_wt), self.v))
        return np.diag(tmp).copy() * 2.0
