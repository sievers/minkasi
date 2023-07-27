import numpy as np
from numpy.typing import NDArray

try:
    from typing import Protocol, runtime_checkable
except:
    from typing_extensions import Protocol, runtime_checkable

def apply_noise(tod,dat=None):
    if dat is None:
        #dat=tod['dat_calib']
        dat=tod.get_data().copy()
    dat_rot=np.dot(tod['v'],dat)
    datft=mkfftw.fft_r2r(dat_rot)
    nn=datft.shape[1]
    datft=datft*tod['mywt'][:,0:nn]
    dat_rot=mkfftw.fft_r2r(datft)
    dat=np.dot(tod['v'].transpose(),dat_rot)
    #for fft_r2r, the first/last samples get counted for half of the interior ones, so 
    #divide them by 2 in the post-filtering.  Makes symmetry much happier...
    #print 'hello'
    dat[:,0]=dat[:,0]*0.5
    dat[:,-1]=dat[:,-1]*0.5


    return dat

def get_grad_mask_2d(map,todvec=None,thresh=4.0,noisemap=None,hitsmap=None):
    """make a mask that has an estimate of the gradient within a pixel.  Look at the 
    rough expected noise to get an idea of which gradients are substantially larger than
    the map noise."""
    if  noisemap is None:
        noisemap=make_hits(todvec,map,do_weights=True)
        noisemap.invert()
        noisemap.map=np.sqrt(noisemap.map)
    if hitsmap is None:
        hitsmap=make_hits(todvec,map,do_weights=False)
    mygrad=(map.map-np.roll(map.map,1,axis=0))**2
    mygrad=mygrad+(map.map-np.roll(map.map,-1,axis=0))**2
    mygrad=mygrad+(map.map-np.roll(map.map,-1,axis=1))**2
    mygrad=mygrad+(map.map-np.roll(map.map,1,axis=1))**2
    mygrad=np.sqrt(0.25*mygrad)






    #find the typical timestream noise in a pixel, which should be the noise map times sqrt(hits)
    hitsmask=hitsmap.map>0
    tmp=noisemap.map.copy()
    tmp[hitsmask]=tmp[hitsmask]*np.sqrt(hitsmap.map[hitsmask])
    #return mygrad,tmp
    mask=(mygrad>(thresh*tmp))
    frac=1.0*np.sum(mask)/mask.size
    print("Cutting " + repr(frac*100) + "% of map pixels in get_grad_mask_2d.")
    mygrad[np.logical_not(mask)]=0
    #return mygrad,tmp,noisemap
    return mygrad

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
    def __init__(self, ivar_map, isinv=True, nfac=1.0):
        self.ivar = read_fits_map(ivar_map)
        if not (isinv):
            mask = self.ivar > 0
            self.ivar[mask] = 1.0 / self.ivar[mask]
        self.ivar = self.ivar * nfac

    def apply_noise(self, map):
        return map * self.ivar


class NoiseBinnedDet:
    def __init__(self, dat, dt, freqs=None, scale_facs=None):
        ndet = dat.shape[0]
        ndata = dat.shape[1]
        nn = 2 * (ndata - 1)
        dnu = 1 / (nn * dt)
        bins = np.asarray(freqs / dnu, dtype="int")
        bins = bins[bins < ndata]
        bins = np.hstack([bins, ndata])
        if bins[0] > 0:
            bins = np.hstack([0, bins])
        if bins[0] < 0:
            bins[0] = 0
        self.bins = bins
        nbin = len(bins) - 1
        self.nbin = nbin
        det_ps = np.zeros([ndet, nbin])
        datft = mkfftw.fft_r2r(dat)
        for i in range(nbin):
            det_ps[:, i] = 1.0 / np.mean(datft[:, bins[i] : bins[i + 1]] ** 2, axis=1)
        self.det_ps = det_ps
        self.ndata = ndata
        self.ndet = ndet
        self.nn = nn

    def apply_noise(self, dat):
        datft = mkfftw.fft_r2r(dat)
        for i in range(self.nbin):
            # datft[:,self.bins[i]:self.bins[i+1]]=datft[:,self.bins[i]:self.bins[i+1]]*np.outer(self.det_ps[:,i],self.bins[i+1]-self.bins[i])
            datft[:, self.bins[i] : self.bins[i + 1]] = datft[
                :, self.bins[i] : self.bins[i + 1]
            ] * np.outer(self.det_ps[:, i], np.ones(self.bins[i + 1] - self.bins[i]))
        dd = mkfftw.fft_r2r(datft)
        dd[:, 0] = 0.5 * dd[:, 0]
        dd[:, -1] = 0.5 * dd[:, -1]
        return dd


class NoiseWhite:
    def __init__(self, dat):
        # this is the ratio between the median absolute
        # deviation of the diff and sigma
        fac = scipy.special.erfinv(0.5) * 2
        sigs = np.median(np.abs(np.diff(dat, axis=1)), axis=1) / fac
        self.sigs = sigs
        self.weights = 1 / sigs**2

    def apply_noise(self, dat):
        assert dat.shape[0] == len(self.weights)
        ndet = dat.shape[0]
        for i in range(ndet):
            dat[i, :] = dat[i, :] * self.weights[i]
        return dat


class NoiseWhiteNotch:
    def __init__(self, dat, numin, numax, tod):
        fac = scipy.special.erfinv(0.5) * 2
        sigs = np.median(np.abs(np.diff(dat, axis=1)), axis=1) / fac
        self.sigs = sigs
        self.weights = 1 / sigs**2
        self.weights = self.weights / (
            2 * (dat.shape[1] - 1)
        )  # fold in fft normalization to the weights
        tvec = tod.get_tvec()
        dt = np.median(np.diff(tvec))
        tlen = tvec[-1] - tvec[0]
        dnu = 1.0 / (2 * tlen - dt)
        self.istart = int(np.floor(numin / dnu))
        self.istop = int(np.ceil(numax / dnu)) + 1

    def apply_noise(self, dat):
        assert dat.shape[0] == len(self.weights)
        datft = mkfftw.fft_r2r(dat)
        datft[:, self.istart : self.istop] = 0
        dat = mkfftw.fft_r2r(datft)

        ndet = dat.shape[0]
        for i in range(ndet):
            dat[i, :] = dat[i, :] * self.weights[i]
        return dat


class NoiseBinnedEig:
    def __init__(self, dat, dt, freqs=None, scale_facs=None, thresh=5.0):

        ndet = dat.shape[0]
        ndata = dat.shape[1]
        nn = 2 * (ndata - 1)

        mycov = np.dot(dat, dat.T)
        mycov = 0.5 * (mycov + mycov.T)
        ee, vv = np.linalg.eig(mycov)
        mask = ee > thresh * thresh * np.median(ee)
        vecs = vv[:, mask]
        ts = np.dot(vecs.T, dat)
        resid = dat - np.dot(vv[:, mask], ts)
        dnu = 1 / (nn * dt)
        print("dnu is " + repr(dnu))
        bins = np.asarray(freqs / dnu, dtype="int")
        bins = bins[bins < ndata]
        bins = np.hstack([bins, ndata])
        if bins[0] > 0:
            bins = np.hstack([0, bins])
        if bins[0] < 0:
            bins[0] = 0
        self.bins = bins
        nbin = len(bins) - 1
        self.nbin = nbin

        nmode = ts.shape[0]
        det_ps = np.zeros([ndet, nbin])
        mode_ps = np.zeros([nmode, nbin])
        residft = mkfftw.fft_r2r(resid)
        modeft = mkfftw.fft_r2r(ts)

        for i in range(nbin):
            det_ps[:, i] = 1.0 / np.mean(residft[:, bins[i] : bins[i + 1]] ** 2, axis=1)
            mode_ps[:, i] = 1.0 / np.mean(modeft[:, bins[i] : bins[i + 1]] ** 2, axis=1)
        self.modes = vecs.copy()
        if not (np.all(np.isfinite(det_ps))):
            print(
                "warning - have non-finite numbers in noise model.  This should not be unexpected."
            )
            det_ps[~np.isfinite(det_ps)] = 0.0
        self.det_ps = det_ps
        self.mode_ps = mode_ps
        self.ndata = ndata
        self.ndet = ndet
        self.nn = nn

    def apply_noise(self, dat):
        assert dat.shape[0] == self.ndet
        assert dat.shape[1] == self.ndata
        datft = mkfftw.fft_r2r(dat)
        for i in range(self.nbin):
            n = self.bins[i + 1] - self.bins[i]
            # print('bins are ',self.bins[i],self.bins[i+1],n,datft.shape[1])
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
            # print(tmp.shape,mat.shape)
        dd = mkfftw.fft_r2r(datft)
        dd[:, 0] = 0.5 * dd[:, 0]
        dd[:, -1] = 0.5 * dd[:, -1]
        return dd


class NoiseCMWhite:
    def __init__(self, dat):
        print("setting up noise cm white")
        u, s, v = np.linalg.svd(dat, 0)
        self.ndet = len(s)
        ind = np.argmax(s)
        self.v = np.zeros(self.ndet)
        self.v[:] = u[:, ind]
        pred = np.outer(self.v * s[ind], v[ind, :])
        dat_clean = dat - pred
        myvar = np.std(dat_clean, 1) ** 2
        self.mywt = 1.0 / myvar

    def apply_noise(self, dat, dd=None):
        t1 = time.time()
        mat = np.dot(self.v, np.diag(self.mywt))
        lhs = np.dot(self.v, mat.T)
        rhs = np.dot(mat, dat)
        if isinstance(lhs, np.ndarray):
            cm = np.dot(np.linalg.inv(lhs), rhs)
        else:
            cm = rhs / lhs
        t2 = time.time()
        if dd is None:
            dd = np.empty(dat.shape)
        if have_numba:
            np.outer(-self.v, cm, dd)
            t3 = time.time()
            # dd[:]=dd[:]+dat
            minkasi_nb.axpy_in_place(dd, dat)
            minkasi_nb.scale_matrix_by_vector(dd, self.mywt)
        else:
            dd = dat - np.outer(self.v, cm)
            # print(dd[:4,:4])
            t3 = time.time()
            tmp = np.repeat([self.mywt], len(cm), axis=0).T
            dd = dd * tmp
        t4 = time.time()
        # print(t2-t1,t3-t2,t4-t3)
        return dd

    def get_det_weights(self):
        return self.mywt.copy()


class NoiseSmoothedSVD:
    def __init__(self, dat_use, fwhm=50, prewhiten=False, fit_powlaw=False, u_in=None):
        if prewhiten:
            noisevec = np.median(np.abs(np.diff(dat_use, axis=1)), axis=1)
            dat_use = dat_use / (
                np.repeat([noisevec], dat_use.shape[1], axis=0).transpose()
            )
        if u_in is None:
            u, s, v = np.linalg.svd(dat_use, 0)
            ndet = s.size
        else:
            u = u_in
            assert u.shape[0] == u.shape[1]
            ndet = u.shape[0]

        # print(u.shape,s.shape,v.shape)
        print("got svd")

        n = dat_use.shape[1]
        self.v = np.zeros([ndet, ndet])
        self.v[:] = u.transpose()
        if u_in is None:
            self.vT = self.v.T
        else:
            self.vT = np.linalg.inv(self.v)
        dat_rot = np.dot(self.v, dat_use)
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
            self.noisevec = noisevec.copy()
        else:
            self.noisevec = None
        self.mywt = spec_smooth

    def apply_noise(self, dat: NDArray[np.floating]) -> NDArray[np.floating]:
        if not (self.noisevec is None):
            noisemat = np.repeat([self.noisevec], dat.shape[1], axis=0).transpose()
            dat = dat / noisemat
        dat_rot = np.dot(self.v, dat)
        datft = mkfftw.fft_r2r(dat_rot)
        nn = datft.shape[1]
        datft = datft * self.mywt[:, :nn]
        dat_rot = mkfftw.fft_r2r(datft)
        # dat=np.dot(self.v.T,dat_rot)
        dat = np.dot(self.vT, dat_rot)
        dat[:, 0] = 0.5 * dat[:, 0]
        dat[:, -1] = 0.5 * dat[:, -1]
        if not (self.noisevec is None):
            # noisemat=np.repeat([self.noisevec],dat.shape[1],axis=0).transpose()
            dat = dat / noisemat
        return dat

    def apply_noise_wscratch(self, dat, tmp, tmp2):
        if not (self.noisevec is None):
            noisemat = np.repeat([self.noisevec], dat.shape[1], axis=0).transpose()
            dat = dat / noisemat
        dat_rot = tmp
        dat_rot = np.dot(self.v, dat, dat_rot)
        dat = tmp2
        datft = dat
        datft = mkfftw.fft_r2r(dat_rot, datft)
        nn = datft.shape[1]
        datft[:] = datft * self.mywt[:, :nn]
        dat_rot = tmp
        dat_rot = mkfftw.fft_r2r(datft, dat_rot)
        # dat=np.dot(self.v.T,dat_rot)
        dat = np.dot(self.vT, dat_rot, dat)
        dat[:, 0] = 0.5 * dat[:, 0]
        dat[:, -1] = 0.5 * dat[:, -1]
        if not (self.noisevec is None):
            # noisemat=np.repeat([self.noisevec],dat.shape[1],axis=0).transpose()
            dat = dat / noisemat
        return dat

    def get_det_weights(self):
        """Find the per-detector weights for use in making actual noise maps."""
        mode_wt = np.sum(self.mywt, axis=1)
        # tmp=np.dot(self.v.T,np.dot(np.diag(mode_wt),self.v))
        tmp = np.dot(self.vT, np.dot(np.diag(mode_wt), self.v))
        return np.diag(tmp).copy() * 2.0
