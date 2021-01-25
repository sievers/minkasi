import numpy as np
import numba as nb

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
def tod2map_destriped(mat,pars,lims,do_add=True):
    ndet=mat.shape[0]
    nseg=len(lims)-1
    for seg in nb.prange(nseg):
        for det in range(ndet):
            if do_add==False:
                pars[det,seg]=0
            for i in range(lims[seg],lims[seg+1]):
                pars[det,seg]=pars[det,seg]+mat[det,i]
                

@nb.njit(parallel=True)
def map2tod_binned_det(mat,pars,vec,lims,nbin,do_add=True):
    n=mat.shape[1]
    inds=np.empty(n,dtype='int')
    fac=nbin/(lims[1]-lims[0]) 
    for i in nb.prange(n):
        inds[i]=(vec[i]-lims[0])*fac
    ndet=mat.shape[0]
    if do_add==False:
        pars[:]=0
    for det in nb.prange(ndet):
        for i in np.arange(n):
            pars[det][inds[i]]=pars[det][inds[i]]+mat[det][i]


@nb.njit(parallel=True)
def map2tod_binned_det(mat,pars,vec,lims,nbin,do_add=True):
    n=mat.shape[1]
    inds=np.empty(n,dtype='int')
    fac=nbin/(lims[1]-lims[0]) 
    for i in nb.prange(n):
        inds[i]=(vec[i]-lims[0])*fac
    ndet=mat.shape[0]
    if do_add==False:
        mat[:]=0
    for det in np.arange(ndet):
        for i in nb.prange(n):
            mat[det][i]=mat[det][i]+pars[det][inds[i]]


