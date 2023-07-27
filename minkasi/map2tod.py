try:
    import numba as nb
except ImportError:
    import no_numba as nb

def map2tod(dat,map,ipix,do_add=False,do_omp=True):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    if do_omp:
        map2tod_omp_c(dat.ctypes.data, map.ctypes.data, ndet, ndata, ipix.ctypes.data, do_add)
    else:
        map2tod_simple_c(dat.ctypes.data,map.ctypes.data,ndet,ndata,ipix.ctypes.data,do_add)
    
def polmap2tod(dat,map,poltag,twogamma,ipix,do_add=False,do_omp=True):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    fun=None
    if poltag=='QU':
        fun=map2tod_qu_omp_c
    if poltag=='IQU':
        fun=map2tod_iqu_omp_c
    if poltag=='QU_PRECON':
        fun=map2tod_qu_precon_omp_c
    if poltag=='IQU_PRECON':
        fun=map2tod_iqu_precon_omp_c
    if fun is None:
        print('unknown poltag ' + repr(poltag) + ' in polmap2tod.')
        return
    #print('calling ' + repr(fun))
    fun(dat.ctypes.data,map.ctypes.data,twogamma.ctypes.data,ndet,ndata,ipix.ctypes.data,do_add)

@nb.jit(nopython=True)
def map2todbowl(vecs, params):
    """
    Converts parameters to tods for the tsBowl class.

    Parameters
    ----------
    vecs: np.array(order, ndata, ndet)
        pseudo-Vandermonde matrix
    params: np.array(order, ndet)
        corresponding weights for pseudo-Vandermonde matrix
    """
    
    #Return tod should have shape ndet x ndata
    to_return = np.zeros((vecs.shape[0], vecs.shape[-2]))
    for i in range(vecs.shape[0]):
        to_return[i] = np.dot(vecs[i,...], params[i,...])
    
    return to_return       

@nb.njit(parallel=True)
def map2tod_destriped(mat,pars,lims,do_add=True):
    ndet=mat.shape[0]
    nseg=len(lims)-1
    for seg in nb.prange(nseg):
        for det in range(ndet):
            if do_add:
                for i in range(lims[seg],lims[seg+1]):
                    mat[det,i]=mat[det,i]+pars[det,seg]
            else:
                for i in range(lims[seg],lims[seg+1]):
                    mat[det,i]=pars[det,seg]

@nb.njit(parallel=True)
def __map2tod_binned_det_loop(pars,inds,mat,ndet,n):
    for det in nb.prange(ndet):
        for i in range(n):
            mat[det][i]=mat[det][i]+pars[det][inds[i]]
            #pars[det][inds[i]]=pars[det][inds[i]]+mat[det][i]


def map2tod_binned_det(mat,pars,vec,lims,nbin,do_add=True):
    n=mat.shape[1]
    #print('range is ',pars.min(),pars.max())
    #inds=np.empty(n,dtype='int64')
    fac=nbin/(lims[1]-lims[0]) 
    inds=np.asarray((vec-lims[0])*fac,dtype='int64')
    #print('ind range is ',inds.min(),inds.max())
    #for i in nb.prange(n):
    #    inds[i]=(vec[i]-lims[0])*fac
    ndet=mat.shape[0]
    if do_add==False:
        mat[:]=0
    __map2tod_binned_det_loop(pars,inds,mat,ndet,n)
    #for det in nb.prange(ndet):
    #    for i in np.arange(n):
    #        pars[det][inds[i]]=pars[det][inds[i]]+mat[det][i]

#@nb.njit(parallel=True)
#def map2tod_binned_det(mat,pars,vec,lims,nbin,do_add=True):
#    n=mat.shape[1]
#    inds=np.empty(n,dtype='int')
#    fac=nbin/(lims[1]-lims[0]) 
#    for i in nb.prange(n):
#        inds[i]=(vec[i]-lims[0])*fac
#    ndet=mat.shape[0]
#    if do_add==False:
#        mat[:]=0
#    for det in np.arange(ndet):
#        for i in nb.prange(n):
#            mat[det][i]=mat[det][i]+pars[det][inds[i]]
