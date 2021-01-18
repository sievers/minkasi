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
                

