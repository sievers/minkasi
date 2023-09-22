"""
Place for deprecated functions to live.
Set the environment variable MINKASI_USE_DEPRECATED to true use the old code,
otherwise the modern equivalent is called.

These functions may not be actively maintained.
"""
import os
from distutils.util import strtobool

import numpy as np
from typing_extensions import deprecated

from ..fitting.power_spectrum import fit_ts_ps
from ..mapmaking.noise import NoiseBinnedEig, NoiseCMWhite, NoiseSmoothedSVD
from ..tools import fft
from ..tools.smooth import smooth_many_vecs

try:
    use_dep = strtobool(os.environ.get("MINKASI_USE_DEPRECATED", "false"))
except ValueError:
    use_dep = False


@deprecated("Use tod.set_noise(NoiseCMWhite) instead")
def set_noise_cm_white(tod):
    print("set_noise_cm_white has been deprecated")
    if not use_dep:
        print("Calling tod.set_noise(NoiseCMWhite) for you")
        tod.set_noise(NoiseCMWhite)
        return
    print("WARNING: Running deprecated version")
    u, s, v = np.linalg.svd(tod.info["dat_calib"], False)
    ndet = len(s)
    ind = np.argmax(s)
    mode = np.zeros(ndet)
    # mode[:]=u[:,0]  #bug fixes pointed out by Fernando Zago.  19 Nov 2019
    # pred=np.outer(mode,v[0,:])
    mode[:] = u[:, ind]
    pred = np.outer(mode * s[ind], v[ind, :])

    dat_clean = tod.info["dat_calib"] - pred
    myvar = np.std(dat_clean, 1) ** 2
    tod.info["v"] = mode
    tod.info["mywt"] = 1.0 / myvar
    tod.info["noise"] = "cm_white"


@deprecated("Use tod.set_noise(NoiseBinnedEig) instead")
def set_noise_binned_eig(tod, dat=None, freqs=None, scale_facs=None, thresh=5.0):
    print("set_noise_binned_eig has been deprecated")
    if not use_dep:
        print("Calling tod.set_noise(NoiseBinnedEig) for you")
        tod.set_noise(
            NoiseBinnedEig,
            dat=dat,
            dt=tod.get_tvec(),
            freqs=freqs,
            scale_face=scale_facs,
            thresh=thresh,
        )
        return
    print("WARNING: Running deprecated version")
    dat = tod.get_data(dat)
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
    tod, fwhm=50, func=None, pars=None, prewhiten=False, fit_powlaw=False
):
    """If func comes in as not empty, assume we can call func(pars,tod) to get a predicted model for the tod that
    we subtract off before estimating the noise."""

    print("set_noise_binned_eig has been deprecated")
    if not use_dep:
        kwargs = {"fwhm": fwhm, "prewhiten": prewhiten, "fit_powlaw": fit_powlaw}
        print("Calling tod.set_noise(NoiseSmoothedSVD) for you")
        if func is None:
            tod.set_noise(NoiseSmoothedSVD, tod.info["dat_calib"], **kwargs)
        else:
            dat_use = func(pars, tod)
            dat_use = tod.info["dat_calib"] - dat_use
            tod.set_noise(NoiseSmoothedSVD, dat_use, **kwargs)
        return
    print("WARNING: Running deprecated version")

    if func is None:
        dat_use = tod.info["dat_calib"]
    else:
        dat_use = func(pars, tod)
        dat_use = tod.info["dat_calib"] - dat_use
        # u,s,v=numpy.linalg.svd(tod.info['dat_calib']-tmp,0)
    if prewhiten:
        noisevec = np.median(np.abs(np.diff(dat_use, axis=1)), axis=1)
        dat_use = dat_use / (
            np.repeat([noisevec], dat_use.shape[1], axis=0).transpose()
        )
        tod.info["noisevec"] = noisevec.copy()
    u, s, v = np.linalg.svd(dat_use, False)
    print("got svd")
    ndet = s.size
    n = tod.info["dat_calib"].shape[1]
    tod.info["v"] = np.zeros([ndet, ndet])
    tod.info["v"][:] = u.transpose()
    dat_rot = np.dot(tod.info["v"], tod.info["dat_calib"])
    if fit_powlaw:
        spec_smooth = 0 * dat_rot
        for ind in range(ndet):
            fitp, datsqr, C = fit_ts_ps(dat_rot[ind, :])
            spec_smooth[ind, 1:] = C
    else:
        dat_trans = fft.fft_r2r(dat_rot)
        spec_smooth = smooth_many_vecs(dat_trans**2, fwhm)
    spec_smooth[:, 1:] = 1.0 / spec_smooth[:, 1:]
    spec_smooth[:, 0] = 0
    tod.info["mywt"] = spec_smooth
    tod.info["noise"] = "smoothed_svd"
    # return dat_rot
