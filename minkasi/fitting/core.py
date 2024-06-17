import sys
import time

import numpy as np

from ..parallel import comm, have_mpi, myrank


def update_lamda(lamda, success):
    if success:
        if lamda < 0.2:
            return 0
        else:
            return lamda / np.sqrt(2)
    else:
        if lamda == 0.0:
            return 1.0
        else:
            return 2.0 * lamda


def invsafe(mat, thresh=1e-14):
    u, s, v = np.linalg.svd(mat, 0)
    ii = np.abs(s) < thresh * s.max()
    # print ii
    s_inv = 1 / s
    s_inv[ii] = 0
    tmp = np.dot(np.diag(s_inv), u.transpose())
    return np.dot(v.transpose(), tmp)


def invscale(mat, do_invsafe=False):
    vec = 1 / np.sqrt(abs(np.diag(mat)))
    vec[np.where(vec == np.inf)[0]] = 1e-10
    mm = np.outer(vec, vec)
    mat = mm * mat
    # ee,vv=np.linalg.eig(mat)
    # print 'rcond is ',ee.max()/ee.min(),vv[:,np.argmin(ee)]
    if do_invsafe:
        return mm * invsafe(mat)
    else:
        try:
            return mm * np.linalg.inv(mat)
        except:
            return mm * np.linalg.pinv(mat)


def get_timestream_chisq_from_func(func, pars, tods):
    chisq = 0.0
    for tod in tods.tods:
        derivs, pred = func(pars, tod)
        # delt=tod.info['dat_calib']-pred
        delt = tod.get_data() - pred
        delt_filt = tod.apply_noise(delt)
        delt_filt[:, 0] = delt_filt[:, 0] * 0.5
        delt_filt[:, -1] = delt_filt[:, -1] * 0.5
        chisq = chisq + np.sum(delt * delt_filt)
    return chisq


def get_timestream_chisq_curve_deriv_from_func(
    func, pars, tods, rotmat=None, *args, **kwargs
):
    chisq = 0.0
    grad = 0.0
    curve = 0.0
    # print 'inside func, len(tods) is ',len(tods.tods),len(pars)
    for tod in tods.tods:
        # print 'type of tod is ',type(tod)
        derivs, pred = func(pars, tod, *args, **kwargs)
        if not (rotmat is None):
            derivs = np.dot(rotmat.transpose(), derivs)
        derivs = np.reshape(derivs, [derivs.shape[0], np.product(derivs.shape[1:])])
        derivs_filt = 0 * derivs
        # print('derivs_filt shape is ',derivs_filt.shape)
        # derivs_filt=np.reshape(derivs_filt,[derivs_filt.shape[0],np.product(derivs_filt.shape[1:])])
        # sz=tod.info['dat_calib'].shape
        sz = tod.get_data_dims()
        tmp = np.zeros(sz)
        npp = derivs.shape[0]
        nn = np.product(derivs.shape[1:])
        # delt=tod.info['dat_calib']-pred
        delt = tod.get_data() - pred
        delt_filt = tod.apply_noise(delt)

        for i in range(npp):
            tmp[:, :] = np.reshape(derivs[i, :], sz)
            tmp_filt = tod.apply_noise(tmp)
            # tmp_filt[:,1:-1]=tmp_filt[:,1:-1]*2
            tmp_filt[:, 0] = tmp_filt[:, 0] * 0.5
            tmp_filt[:, -1] = tmp_filt[:, -1] * 0.5
            derivs_filt[i, :] = np.reshape(tmp_filt, nn)
        delt = np.reshape(delt, nn)
        delt_filt = np.reshape(delt_filt, nn)
        grad1 = np.dot(derivs, delt_filt)
        grad2 = np.dot(derivs_filt, delt)
        # print 'grad error is ',np.mean(np.abs((grad1-grad2)/(0.5*(np.abs(grad1)+np.abs(grad2)))))
        grad = grad + 0.5 * (grad1 + grad2)
        curve = curve + np.dot(derivs, derivs_filt.transpose())
        chisq = chisq + np.dot(delt, delt_filt)
    curve = 0.5 * (curve + curve.transpose())
    if have_mpi:
        curve = comm.allreduce(curve)
        grad = comm.allreduce(grad)
        chisq = comm.allreduce(chisq)
    return chisq, grad, curve


def get_ts_derivs_many_funcs(
    tod, pars, npar_fun, funcs, func_args=None, *args, **kwargs
):
    # ndet=tod.info['dat_calib'].shape[0]
    # ndat=tod.info['dat_calib'].shape[1]
    ndet = tod.get_ndet()
    ndat = tod.get_ndata()
    npar = np.sum(np.asarray(npar_fun), dtype="int")
    # vals=np.zeros([ndet,ndat])

    pred = 0
    derivs = np.zeros([npar, ndet, ndat])
    icur = 0
    for i in range(len(funcs)):
        tmp = pars[icur : icur + npar_fun[i]].copy()
        myderivs, mypred = funcs[i](tmp, tod, *args, **kwargs)
        pred = pred + mypred
        derivs[icur : icur + npar_fun[i], :, :] = myderivs
        icur = icur + npar_fun[i]
    return derivs, pred
    # derivs,pred=funcs[i](pars,tod)


def get_ts_curve_derivs_many_funcs(
    todvec, pars, npar_fun, funcs, driver=get_ts_derivs_many_funcs, *args, **kwargs
):
    curve = 0
    grad = 0
    chisq = 0
    for tod in todvec.tods:
        derivs, pred = driver(tod, pars, npar_fun, funcs, *args, **kwargs)
        npar = derivs.shape[0]
        ndet = derivs.shape[1]
        ndat = derivs.shape[2]

        # pred_filt=tod.apply_noise(pred)
        derivs_filt = np.empty(derivs.shape)
        for i in range(npar):
            derivs_filt[i, :, :] = tod.apply_noise(derivs[i, :, :])

        derivs = np.reshape(derivs, [npar, ndet * ndat])
        derivs_filt = np.reshape(derivs_filt, [npar, ndet * ndat])
        # delt=tod.info['dat_calib']-pred
        delt = tod.get_data() - pred
        delt_filt = tod.apply_noise(delt)
        chisq = chisq + np.sum(delt * delt_filt)
        delt = np.reshape(delt, ndet * ndat)
        # delt_filt=np.reshape(delt_filt,[1,ndet*ndat])
        grad = grad + np.dot(derivs_filt, delt.T)
        # grad2=grad2+np.dot(derivs,delt_filt.T)
        curve = curve + np.dot(derivs_filt, derivs.T)
    if have_mpi:
        chisq = comm.allreduce(chisq)
        grad = comm.allreduce(grad)
        curve = comm.allreduce(curve)
    return chisq, grad, curve


def _par_step(grad, curve, to_fit, lamda, flat_priors=None, return_full=False):
    curve_use = curve + lamda * np.diag(np.diag(curve))
    if to_fit is None:
        step = np.dot(invscale(curve_use, True), grad)
        errs = np.sqrt(np.diag(invscale(curve_use, True)))
    else:
        curve_use = curve_use[to_fit, :]
        curve_use = curve_use[:, to_fit]
        grad_use = grad[to_fit]
        step = np.dot(invscale(curve_use), grad_use)
        step_use = np.zeros(len(to_fit))
        step_use[to_fit] = step
        errs_tmp = np.sqrt(np.diag(invscale(curve_use, True)))
        errs = np.zeros(len(to_fit))
        errs[to_fit] = errs_tmp
        step = step_use
    # print('step shape ',step.shape,step)
    if return_full:
        return step, errs
    else:
        return step


def fit_timestreams_with_derivs_manyfun(
    funcs,
    pars,
    npar_fun,
    tods,
    to_fit=None,
    to_scale=None,
    tol=1e-2,
    chitol=1e-4,
    maxiter=10,
    scale_facs=None,
    driver=get_ts_derivs_many_funcs,
    priors=None,
    prior_vals=None,
):
    lamda = 0
    t1 = time.time()
    chisq, grad, curve = get_ts_curve_derivs_many_funcs(
        tods, pars, npar_fun, funcs, driver=driver
    )
    t2 = time.time()
    if myrank == 0:
        print(
            "starting chisq is ", chisq, " with ", t2 - t1, " seconds to get curvature"
        )
    if to_fit is None:
        # If to_fit is not already defined, define it an intialize it to true
        # we're going to use it to handle not stepping for flat priors
        to_fit = np.ones(len(pars), dtype="bool")

    for iter in range(maxiter):
        temp_to_fit = np.copy(
            to_fit
        )  # Make a copy of to fit, so we can temporarily set values to false
        if np.any(priors):
            # first build a mask that will identify parameters with flat priors
            flat_mask = np.where((priors == np.array("flat")))[0]

            for flat_id in flat_mask:
                #print(pars[flat_id])
                if (pars[flat_id] == prior_vals[flat_id][0]) or (
                    pars[flat_id] == prior_vals[flat_id][1]
                ):
                    # Check to see if we're at the boundry values, if so don't fit for this iter
                    temp_to_fit[flat_id] = False

            # Make the new step
            pars_new = pars + _par_step(grad, curve, temp_to_fit, lamda)
            # check to see if we're outside the range for the flat priors: if so, peg them
            #print("old gamma: ", pars_new[flat_id])
            for flat_id in flat_mask:
                if pars_new[flat_id] < prior_vals[flat_id][0]:
                    pars_new[flat_id] = prior_vals[flat_id][0]
                elif pars_new[flat_id] > prior_vals[flat_id][1]:
                    pars_new[flat_id] = prior_vals[flat_id][1]
            #print("new gamma: ", pars_new[flat_id])
        else:
            pars_new = pars + _par_step(grad, curve, to_fit, lamda)
        chisq_new, grad_new, curve_new = get_ts_curve_derivs_many_funcs(
            tods, pars_new, npar_fun, funcs, driver=driver
        )
        if chisq_new < chisq:
            if myrank == 0:
                print(
                    "accepting with delta_chisq ",
                    chisq_new - chisq,
                    " and lamda ",
                    lamda,
                    pars_new.shape,
                )
                print(repr(pars_new))
            pars = pars_new
            curve = curve_new
            grad = grad_new
            lamda = update_lamda(lamda, True)
            if (chisq - chisq_new < chitol) & (lamda == 0):
                step, errs = _par_step(
                    grad, curve, temp_to_fit, lamda, return_full=True
                )
                return pars, chisq_new, curve_new, errs
            else:
                chisq = chisq_new
        else:
            if myrank == 0:
                print(
                    "rejecting with delta_chisq ",
                    chisq_new - chisq,
                    " and lamda ",
                    lamda,
                )
            lamda = update_lamda(lamda, False)
        sys.stdout.flush()
    if myrank == 0:
        print(
            "fit_timestreams_with_derivs_manyfun failed to converge after ",
            maxiter,
            " iterations.",
        )
    step, errs = _par_step(grad, curve, temp_to_fit, lamda, return_full=True)
    return pars, chisq, curve, errs


def fit_timestreams_with_derivs(
    func,
    pars,
    tods,
    to_fit=None,
    to_scale=None,
    tol=1e-2,
    chitol=1e-4,
    maxiter=10,
    scale_facs=None,
):
    if not (to_fit is None):
        # print 'working on creating rotmat'
        to_fit = np.asarray(to_fit, dtype="int64")
        inds = np.unique(to_fit)
        nfloat = np.sum(to_fit == 1)
        ncovary = np.sum(inds > 1)
        nfit = nfloat + ncovary
        rotmat = np.zeros([len(pars), nfit])

        solo_inds = np.where(to_fit == 1)[0]
        icur = 0
        for ind in solo_inds:
            rotmat[ind, icur] = 1.0
            icur = icur + 1
        if ncovary > 0:
            group_inds = inds[inds > 1]
            for ind in group_inds:
                ii = np.where(to_fit == ind)[0]
                rotmat[ii, icur] = 1.0
                icur = icur + 1
    else:
        rotmat = None

    iter = 0
    converged = False
    pp = pars.copy()
    lamda = 0.0
    chi_ref, grad, curve = get_timestream_chisq_curve_deriv_from_func(
        func, pp, tods, rotmat
    )
    chi_cur = chi_ref
    iter = 0
    while (converged == False) and (iter < maxiter):
        iter = iter + 1
        curve_tmp = curve + lamda * np.diag(np.diag(curve))
        # curve_inv=np.linalg.inv(curve_tmp)
        curve_inv = invscale(curve_tmp)
        shifts = np.dot(curve_inv, grad)
        if not (rotmat is None):
            shifts_use = np.dot(rotmat, shifts)
        else:
            shifts_use = shifts
        pp_tmp = pp + shifts_use
        chi_new = get_timestream_chisq_from_func(func, pp_tmp, tods)
        if (
            chi_new <= chi_cur + chitol
        ):  # add in a bit of extra tolerance in chi^2 in case we're bopping about the minimum
            success = True
        else:
            success = False
        if success:
            pp = pp_tmp
            chi_cur = chi_new
            chi_tmp, grad, curve = get_timestream_chisq_curve_deriv_from_func(
                func, pp, tods, rotmat
            )
        lamda = update_lamda(lamda, success)
        if (lamda == 0) & success:
            errs = np.sqrt(np.diag(curve_inv))
            conv_fac = np.max(np.abs(shifts / errs))
            if conv_fac < tol:
                print("we have converged")
                converged = True
        else:
            conv_fac = None
        to_print = np.asarray(
            [
                3600 * 180.0 / np.pi,
                3600 * 180.0 / np.pi,
                3600 * 180.0 / np.pi,
                1.0,
                1.0,
                3600 * 180.0 / np.pi,
                3600 * 180.0 / np.pi,
                3600 * 180.0 / np.pi * np.sqrt(8 * np.log(2)),
                1.0,
            ]
        ) * (pp - pars)
        print(
            "iter",
            iter,
            " max_shift is ",
            conv_fac,
            " with lamda ",
            lamda,
            chi_ref - chi_cur,
            chi_ref - chi_new,
        )
    return pp, chi_cur
