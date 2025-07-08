import numpy as np

from ..lib.minkasi import (
    fill_gauss_derivs_c,
    fill_gauss_src_c,
    fill_isobeta_c,
    fill_isobeta_derivs_c,
)

try:
    import numba as nb
except ImportError:
    from ..tools import no_numba as nb


def timestreams_from_gauss(ra, dec, fwhm, tod, pred=None):
    if pred is None:
        # pred=np.zeros(tod.info['dat_calib'].shape)
        pred = tod.get_empty(True)
    # n=tod.info['dat_calib'].size
    n = np.product(tod.get_data_dims())
    assert pred.size == n
    npar_src = 4  # x,y,sig,amp
    dx = tod.info["dx"]
    dy = tod.info["dy"]
    pp = np.zeros(npar_src)
    pp[0] = ra
    pp[1] = dec
    pp[2] = fwhm / np.sqrt(8 * np.log(2)) * np.pi / 180 / 3600
    pp[3] = 1
    fill_gauss_src_c(
        pp.ctypes.data, dx.ctypes.data, dy.ctypes.data, pred.ctypes.data, n
    )
    return pred


def timestreams_from_isobeta_c(params, tod, pred=None):
    if pred is None:
        # pred=np.zeros(tod.info['dat_calib'].shape)
        pred = tod.get_empty(True)
    # n=tod.info['dat_calib'].size
    n = np.product(tod.get_data_dims())
    assert pred.size == n
    dx = tod.info["dx"]
    dy = tod.info["dy"]
    fill_isobeta_c(
        params.ctypes.data, dx.ctypes.data, dy.ctypes.data, pred.ctypes.data, n
    )

    npar_beta = 5  # x,y,theta,beta,amp
    npar_src = 4  # x,y,sig,amp
    nsrc = (params.size - npar_beta) // npar_src
    for i in range(nsrc):
        pp = np.zeros(npar_src)
        ioff = i * npar_src + npar_beta
        pp[:] = params[ioff : (ioff + npar_src)]
        fill_gauss_src_c(
            pp.ctypes.data, dx.ctypes.data, dy.ctypes.data, pred.ctypes.data, n
        )

    return pred


def derivs_from_elliptical_isobeta(params, tod, *args, **kwargs):
    npar = len(params)
    assert npar == 7
    pred = tod.get_empty()
    dims = np.hstack([npar, pred.shape])
    derivs = np.empty(dims)

    dx = tod.info["dx"]
    dy = tod.info["dy"]
    fill_elliptical_isobeta_derivs(params, dx, dy, pred, derivs)
    return derivs, pred


def derivs_from_elliptical_gauss(params, tod, *args, **kwargs):
    npar = len(params)
    assert npar == 6
    pred = tod.get_empty()
    dims = np.hstack([npar, pred.shape])
    derivs = np.empty(dims)

    dx = tod.info["dx"]
    dy = tod.info["dy"]
    fill_elliptical_gauss_derivs(params, dx, dy, pred, derivs)
    return derivs, pred


def derivs_from_isobeta_c(params, tod, *args, **kwargs):
    npar = 5
    # n=tod.info['dat_calib'].size
    dims = tod.get_data_dims()
    n = np.product(dims)
    # sz_deriv=np.append(npar,tod.info['dat_calib'].shape)
    sz_deriv = np.append(npar, dims)
    # pred=np.zeros(tod.info['dat_calib'].shape)
    pred = tod.get_empty(True)
    derivs = np.zeros(sz_deriv)

    dx = tod.info["dx"]
    dy = tod.info["dy"]
    fill_isobeta_derivs_c(
        params.ctypes.data,
        dx.ctypes.data,
        dy.ctypes.data,
        pred.ctypes.data,
        derivs.ctypes.data,
        n,
    )

    return derivs, pred


def derivs_from_gauss_c(params, tod, *args, **kwargs):
    npar = 4
    # n=tod.info['dat_calib'].size
    n = tod.get_nsamp()
    # sz_deriv=np.append(npar,tod.info['dat_calib'].shape)
    sz_deriv = np.append(npar, tod.get_data_dims())

    # pred=np.zeros(tod.info['dat_calib'].shape)
    pred = tod.get_empty(True)

    derivs = np.zeros(sz_deriv)

    dx = tod.info["dx"]
    dy = tod.info["dy"]
    fill_gauss_derivs_c(
        params.ctypes.data,
        dx.ctypes.data,
        dy.ctypes.data,
        pred.ctypes.data,
        derivs.ctypes.data,
        n,
    )

    return derivs, pred

def lc_derivs_from_gauss(params, tod, *args, **kwargs):
    tod_dims = tod.get_data_dims()
    try:
        lc_bins = kwargs["lc_bins"].copy()
        cur_bin = kwargs["fun_num"]
    except:
        raise ValueError("Error, must pass lc_bins and fun_num in kwargs")
    if lc_bins[-1] >= tod_dims[-1]:
        raise ValueError("Error: last lc_bin should be less than the tod length")
    derivs = np.zeros(np.append(len(params), tod_dims))
    pred = tod.get_empty(True)
    lc_bins.append(tod_dims[-1])
    dx = tod.info["dx"]
    dy = tod.info["dy"]

    idx_low, idx_hi = lc_bins[cur_bin], lc_bins[cur_bin+1]
    cur_derivs, cur_pred = derivs_from_gauss_c(params, tod)
    derivs[...,idx_low:idx_hi] = cur_derivs[...,idx_low:idx_hi]
    pred[...,idx_low:idx_hi] = cur_pred[...,idx_low:idx_hi]

    return derivs, pred

def derivs_from_map(pars, tod, fun, map, dpar, do_symm=False, *args, **kwargs):
    # print('do_symm is ',do_symm)
    pred = tod.get_empty()
    fun(map, pars, *args, **kwargs)
    map.map2tod(tod, pred, False)
    npar = len(pars)
    tmp = tod.get_empty()
    if do_symm:
        tmp2 = tod.get_empty()
    derivs = np.empty([npar, pred.shape[0], pred.shape[1]])
    for i in range(npar):
        pp = pars.copy()
        pp[i] = pp[i] + dpar[i]
        fun(map, pp, *args, **kwargs)
        tmp[
            :
        ] = 0  # strictly speaking, we shouldn't need this, but it makes us more robust to bugs elsewhere
        map.map2tod(tod, tmp, False)
        if do_symm:
            pp = pars.copy()
            pp[i] = pp[i] - dpar[i]
            fun(map, pp, *args, **kwargs)
            tmp2[:] = 0
            map.map2tod(tod, tmp2, False)
            derivs[i, :, :] = (tmp - tmp2) / (2 * dpar[i])
        else:
            derivs[i, :, :] = (tmp - pred) / (dpar[i])
    # pred=np.reshape(pred,pred.size)
    # derivs=np.reshape(derivs,[derivs.shape[0],derivs.shape[1]*derivs.shape[2]])
    return derivs, pred


def timestreams_from_isobeta(params, tod):
    npar_beta = 5  # x,y,theta,beta,amp
    npar_src = 4  # x,y,sig,amp
    nsrc = (params.size - npar_beta) // npar_src
    assert params.size == nsrc * npar_src + npar_beta
    x0 = params[0]
    y0 = params[1]
    theta = params[2]
    beta = params[3]
    amp = params[4]
    cosdec = np.cos(y0)

    dx = (tod.info["dx"] - x0) * cosdec
    dy = tod.info["dy"] - y0
    rsqr = dx * dx + dy * dy
    rsqr = rsqr / theta**2
    # print rsqr.max()
    pred = amp * (1 + rsqr) ** (0.5 - 1.5 * beta)
    for i in range(nsrc):
        src_x = params[i * npar_src + npar_beta + 0]
        src_y = params[i * npar_src + npar_beta + 1]
        src_sig = params[i * npar_src + npar_beta + 2]
        src_amp = params[i * npar_src + npar_beta + 3]

        dx = tod.info["dx"] - src_x
        dy = tod.info["dy"] - src_y
        rsqr = (dx * np.cos(src_y)) ** 2 + dy**2
        pred = pred + src_amp * np.exp(-0.5 * rsqr / src_sig**2)

    return pred


def isobeta_src_chisq(params, tods):
    chisq = 0.0
    for tod in tods.tods:
        pred = timestreams_from_isobeta_c(params, tod)
        # chisq=chisq+tod.timestream_chisq(tod.info['dat_calib']-pred)
        chisq = chisq + tod.timestream_chisq(tod.get_data() - pred)

    return chisq
    npar_beta = 5  # x,y,theta,beta,amp
    npar_src = 4  # x,y,sig,amp
    nsrc = (params.size - npar_beta) // npar_src
    assert params.size == nsrc * npar_src + npar_beta
    x0 = params[0]
    y0 = params[1]
    theta = params[2]
    beta = params[3]
    amp = params[4]
    cosdec = np.cos(y0)
    chisq = 0.0
    for tod in tods.tods:
        dx = tod.info["dx"] - x0
        dy = tod.info["dy"] - y0
        rsqr = (dx * cosdec) ** 2 + dy**2
        pred = amp * (1 + rsqr / theta**2) ** (0.5 - 1.5 * beta)
        for i in range(nsrc):
            src_x = params[i * npar_src + npar_beta + 0]
            src_y = params[i * npar_src + npar_beta + 1]
            src_sig = params[i * npar_src + npar_beta + 2]
            src_amp = params[i * npar_src + npar_beta + 3]

            dx = tod.info["dx"] - src_x
            dy = tod.info["dy"] - src_y
            rsqr = (dx * np.cos(src_y)) ** 2 + dy**2
            pred = pred + src_amp * np.exp(-0.5 * rsqr / src_sig**2)
        # chisq=chisq+tod.timestream_chisq(tod.info['dat_calib']-pred)
        chisq = chisq + tod.timestream_chisq(tod.get_data() - pred)
    return chisq


def get_derivs_tod_isosrc(pars, tod, niso=None):
    np_src = 4
    np_iso = 5
    # nsrc=(len(pars)-np_iso)/np_src
    npp = len(pars)
    if niso is None:
        niso = (npp % np_src) / (np_iso - np_src)
    nsrc = (npp - niso * np_iso) / np_src
    # print nsrc,niso

    fitp_iso = np.zeros(np_iso)
    fitp_iso[:] = pars[:np_iso]
    # print 'fitp_iso is ',fitp_iso
    derivs_iso, f_iso = derivs_from_isobeta_c(fitp_iso, tod)

    # nn=tod.info['dat_calib'].size
    nn = tod.get_nsamp()
    derivs = np.reshape(derivs_iso, [np_iso, nn])
    pred = f_iso

    for ii in range(nsrc):
        fitp_src = np.zeros(np_src)
        istart = np_iso + ii * np_src
        fitp_src[:] = pars[istart : istart + np_src]
        derivs_src, f_src = derivs_from_gauss_c(fitp_src, tod)
        pred = pred + f_src
        derivs_src_tmp = np.reshape(derivs_src, [np_src, nn])
        derivs = np.append(derivs, derivs_src_tmp, axis=0)
    return derivs, pred


def get_curve_deriv_tod_manygauss(pars, tod, return_vecs=False):
    npp = 4
    nsrc = len(pars) // npp
    fitp_gauss = np.zeros(npp)
    # dat=tod.info['dat_calib']
    dat = tod.get_data()
    big_derivs = np.zeros([npp * nsrc, dat.shape[0], dat.shape[1]])
    pred = 0
    curve = np.zeros([npp * nsrc, npp * nsrc])
    deriv = np.zeros([npp * nsrc])
    for i in range(nsrc):
        fitp_gauss[:] = pars[i * npp : (i + 1) * npp]
        derivs, src_pred = derivs_from_gauss_c(fitp_gauss, tod)
        pred = pred + src_pred
        big_derivs[i * npp : (i + 1) * npp, :, :] = derivs
    delt = dat - pred
    delt_filt = tod.apply_noise(delt)
    chisq = 0.5 * np.sum(delt[:, 0] * delt_filt[:, 0])
    chisq = chisq + 0.5 * np.sum(delt[:, -1] * delt_filt[:, -1])
    chisq = chisq + np.sum(delt[:, 1:-1] * delt_filt[:, 1:-1])
    for i in range(npp * nsrc):
        deriv_filt = tod.apply_noise(big_derivs[i, :, :])
        for j in range(i, npp * nsrc):
            curve[i, j] = curve[i, j] + 0.5 * np.sum(
                deriv_filt[:, 0] * big_derivs[j, :, 0]
            )
            curve[i, j] = curve[i, j] + 0.5 * np.sum(
                deriv_filt[:, -1] * big_derivs[j, :, -1]
            )
            curve[i, j] = curve[i, j] + np.sum(
                deriv_filt[:, 1:-1] * big_derivs[j, :, 1:-1]
            )
            curve[j, i] = curve[i, j]
            # print i,j,curve[i,j]
        deriv[i] = deriv[i] + 0.5 * np.sum(deriv_filt[:, 0] * delt[:, 0])
        deriv[i] = deriv[i] + 0.5 * np.sum(deriv_filt[:, -1] * delt[:, -1])
        deriv[i] = deriv[i] + np.sum(deriv_filt[:, 1:-1] * delt[:, 1:-1])
    return curve, deriv, chisq


def get_curve_deriv_tod_isosrc(pars, tod, return_vecs=False):
    np_src = 4
    np_iso = 5
    nsrc = (len(pars) - np_iso) / np_src
    # print 'nsrc is ',nsrc
    fitp_iso = np.zeros(np_iso)
    fitp_iso[:] = pars[:np_iso]
    # print 'fitp_iso is ',fitp_iso
    derivs_iso, f_iso = derivs_from_isobeta_c(fitp_iso, tod)
    derivs_iso_filt = 0 * derivs_iso
    # tmp=0*tod.info['dat_calib']
    tmp = tod.get_empty(True)
    # nn=tod.info['dat_calib'].size
    nn = tod.get_nsamp
    for i in range(np_iso):
        tmp[:, :] = derivs_iso[i, :, :]
        derivs_iso_filt[i, :, :] = tod.apply_noise(tmp)
    derivs = np.reshape(derivs_iso, [np_iso, nn])
    derivs_filt = np.reshape(derivs_iso_filt, [np_iso, nn])
    pred = f_iso

    for ii in range(nsrc):
        fitp_src = np.zeros(np_src)
        istart = np_iso + ii * np_src
        fitp_src[:] = pars[istart : istart + np_src]
        # print 'fitp_src is ',fitp_src
        derivs_src, f_src = derivs_from_gauss_c(fitp_src, tod)
        pred = pred + f_src
        derivs_src_filt = 0 * derivs_src
        for i in range(np_src):
            tmp[:, :] = derivs_src[i, :, :]
            derivs_src_filt[i, :, :] = tod.apply_noise(tmp)
        derivs_src_tmp = np.reshape(derivs_src, [np_src, nn])
        derivs = np.append(derivs, derivs_src_tmp, axis=0)
        derivs_src_tmp = np.reshape(derivs_src_filt, [np_src, nn])
        derivs_filt = np.append(derivs_filt, derivs_src_tmp, axis=0)

    # delt_filt=tod.apply_noise(tod.info['dat_calib']-pred)
    delt_filt = tod.apply_noise(tod.get_data() - pred)
    delt_filt = np.reshape(delt_filt, nn)

    # dvec=np.reshape(tod.info['dat_calib'],nn)
    dvec = np.ravel(tod.get_data())
    # predvec=np.reshape(pred,nn)
    predvec = np.ravel(pred)
    delt = dvec - predvec

    grad = np.dot(derivs_filt, delt)
    grad2 = np.dot(derivs, delt_filt)
    curve = np.dot(derivs_filt, derivs.transpose())
    # return pred
    if return_vecs:
        return grad, grad2, curve, derivs, derivs_filt, delt, delt_filt
    else:
        return grad, grad2, curve


@nb.njit(parallel=True)
def fill_elliptical_isobeta(params, dx, dy, pred):
    ndet = dx.shape[0]
    n = dx.shape[1]
    x0 = params[0]
    y0 = params[1]
    theta1 = params[2]
    theta2 = params[3]
    theta1_inv = 1 / theta1
    theta2_inv = 1 / theta2
    theta1_inv_sqr = theta1_inv**2
    theta2_inv_sqr = theta2_inv**2
    psi = params[4]
    beta = params[5]
    amp = params[6]
    cosdec = np.cos(y0)
    cospsi = np.cos(psi)
    sinpsi = np.sin(psi)
    mypow = 0.5 - 1.5 * beta
    for det in nb.prange(ndet):
        for j in np.arange(n):
            delx = (dx[det, j] - x0) * cosdec
            dely = dy[det, j] - y0
            xx = delx * cospsi + dely * sinpsi
            yy = dely * cospsi - delx * sinpsi
            rr = 1 + theta1_inv_sqr * xx * xx + theta2_inv_sqr * yy * yy
            pred[det, j] = amp * (rr**mypow)


@nb.njit(parallel=True)
def fill_elliptical_isobeta_derivs(params, dx, dy, pred, derivs):
    """Fill model/derivatives for an isothermal beta model.
    Parameters should be [ra,dec,theta axis 1,theta axis 2,angle,beta,amplitude.
    Beta should be positive (i.e. 0.7, not -0.7)."""

    ndet = dx.shape[0]
    n = dx.shape[1]
    x0 = params[0]
    y0 = params[1]
    theta1 = params[2]
    theta2 = params[3]
    theta1_inv = 1 / theta1
    theta2_inv = 1 / theta2
    theta1_inv_sqr = theta1_inv**2
    theta2_inv_sqr = theta2_inv**2
    psi = params[4]
    beta = params[5]
    amp = params[6]
    cosdec = np.cos(y0)
    sindec = np.sin(y0) / np.cos(y0)
    # cosdec=np.cos(dy[0,0])
    # cosdec=1.0
    cospsi = np.cos(psi)
    cc = cospsi**2
    sinpsi = np.sin(psi)
    ss = sinpsi**2
    cs = cospsi * sinpsi
    mypow = 0.5 - 1.5 * beta
    for det in nb.prange(ndet):
        for j in np.arange(n):
            delx = (dx[det, j] - x0) * cosdec
            dely = dy[det, j] - y0
            xx = delx * cospsi + dely * sinpsi
            yy = dely * cospsi - delx * sinpsi
            xfac = theta1_inv_sqr * xx * xx
            yfac = theta2_inv_sqr * yy * yy
            # rr=1+theta1_inv_sqr*xx*xx+theta2_inv_sqr*yy*yy
            rr = 1 + xfac + yfac
            rrpow = rr**mypow

            pred[det, j] = amp * rrpow
            dfdrr = rrpow / rr * mypow
            drdx = (
                -2 * delx * (cc * theta1_inv_sqr + ss * theta2_inv_sqr)
                - 2 * dely * (theta1_inv_sqr - theta2_inv_sqr) * cs
            )
            # drdy=-2*dely*(cc*theta2_inv_sqr+ss*theta1_inv_sqr)-2*delx*(theta1_inv_sqr-theta2_inv_sqr)*cs
            drdy = -(
                2 * xx * theta1_inv_sqr * (cospsi * sindec * delx + sinpsi)
                + 2 * yy * theta2_inv_sqr * (-sinpsi * sindec * delx + cospsi)
            )
            drdtheta = (
                2
                * (theta1_inv_sqr - theta2_inv_sqr)
                * (cs * (dely**2 - delx**2) + delx * dely * (cc - ss))
            )
            # drdtheta=-2*delx**2*cs*(theta_1_inv_sqr-theta_2_inv_sqr)+2*dely*delx*(theta_1_inv_sqr-theta_2_inv_sqr)*(cc-ss)+2*dely**2*cs*(

            derivs[0, det, j] = dfdrr * drdx * cosdec
            derivs[1, det, j] = dfdrr * drdy
            derivs[2, det, j] = dfdrr * xfac * (-2 * theta1_inv)
            derivs[3, det, j] = dfdrr * yfac * (-2 * theta2_inv)
            derivs[4, det, j] = dfdrr * drdtheta
            derivs[5, det, j] = -1.5 * np.log(rr) * amp * rrpow
            derivs[6, det, j] = rrpow


@nb.njit(parallel=True)
def fill_elliptical_gauss_derivs(params, dx, dy, pred, derivs):
    """Fill model/derivatives for an elliptical gaussian model.
    Parameters should be [ra,dec,sigma axis 1,sigmaaxis 2,angle,amplitude."""

    ndet = dx.shape[0]
    n = dx.shape[1]
    x0 = params[0]
    y0 = params[1]
    theta1 = params[2]
    theta2 = params[3]
    theta1_inv = 1 / theta1
    theta2_inv = 1 / theta2
    theta1_inv_sqr = theta1_inv**2
    theta2_inv_sqr = theta2_inv**2
    psi = params[4]
    amp = params[5]
    cosdec = np.cos(y0)
    sindec = np.sin(y0) / np.cos(y0)
    cospsi = np.cos(psi)
    cc = cospsi**2
    sinpsi = np.sin(psi)
    ss = sinpsi**2
    cs = cospsi * sinpsi

    for det in nb.prange(ndet):
        for j in np.arange(n):
            delx = (dx[det, j] - x0) * cosdec
            dely = dy[det, j] - y0
            xx = delx * cospsi + dely * sinpsi
            yy = dely * cospsi - delx * sinpsi
            xfac = theta1_inv_sqr * xx * xx
            yfac = theta2_inv_sqr * yy * yy
            # rr=1+theta1_inv_sqr*xx*xx+theta2_inv_sqr*yy*yy
            rr = xfac + yfac
            rrpow = np.exp(-0.5 * rr)

            pred[det, j] = amp * rrpow
            dfdrr = -0.5 * rrpow
            drdx = (
                -2 * delx * (cc * theta1_inv_sqr + ss * theta2_inv_sqr)
                - 2 * dely * (theta1_inv_sqr - theta2_inv_sqr) * cs
            )
            # drdy=-2*dely*(cc*theta2_inv_sqr+ss*theta1_inv_sqr)-2*delx*(theta1_inv_sqr-theta2_inv_sqr)*cs
            drdy = -(
                2 * xx * theta1_inv_sqr * (cospsi * sindec * delx + sinpsi)
                + 2 * yy * theta2_inv_sqr * (-sinpsi * sindec * delx + cospsi)
            )

            drdtheta = (
                2
                * (theta1_inv_sqr - theta2_inv_sqr)
                * (cs * (dely**2 - delx**2) + delx * dely * (cc - ss))
            )
            # drdtheta=-2*delx**2*cs*(theta_1_inv_sqr-theta_2_inv_sqr)+2*dely*delx*(theta_1_inv_sqr-theta_2_inv_sqr)*(cc-ss)+2*dely**2*cs*(

            derivs[0, det, j] = dfdrr * drdx * cosdec
            # derivs[1,det,j]=dfdrr*(drdy-2*sindec*delx**2*theta1_inv_sqr)
            derivs[1, det, j] = dfdrr * drdy
            derivs[2, det, j] = dfdrr * xfac * (-2 * theta1_inv)
            derivs[3, det, j] = dfdrr * yfac * (-2 * theta2_inv)
            derivs[4, det, j] = dfdrr * drdtheta
            derivs[5, det, j] = rrpow
