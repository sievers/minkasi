import numpy as np
import ctypes
import time
import mkfftw
#import pyfits
from astropy.io import fits as pyfits
import astropy
from astropy import wcs
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo #choose your cosmology here
import scipy
import copy
import sys
try:
    import healpy
    have_healpy=True
except:
    have_healpy=False
try: 
    import numba as nb
    import minkasi_nb
    have_numba=True
except:
    have_numba=False

try:
    import qpoint as qp
    have_qp=True
except:
    have_qp=False

try:
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nproc=comm.Get_size()
    if nproc>1:
        have_mpi=True
    else:
        have_mpi=False
except:
    have_mpi=False
    myrank=0
    nproc=1
#try:
#    import numba as nb
#    have_numba=True
#else:
#    have_numba=False


mylib=ctypes.cdll.LoadLibrary("libminkasi.so")
tod2map_simple_c=mylib.tod2map_simple
tod2map_simple_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

tod2map_omp_c=mylib.tod2map_omp
tod2map_omp_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]

tod2map_cached_c=mylib.tod2map_cached
tod2map_cached_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]

map2tod_simple_c=mylib.map2tod_simple
map2tod_simple_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]

map2tod_omp_c=mylib.map2tod_omp
map2tod_omp_c.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,ctypes.c_int]

map2tod_iqu_omp_c=mylib.map2tod_iqu_omp
map2tod_iqu_omp_c.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]

map2tod_qu_omp_c=mylib.map2tod_qu_omp
map2tod_qu_omp_c.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]

tod2map_iqu_simple_c=mylib.tod2map_iqu_simple
tod2map_iqu_simple_c.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

tod2map_qu_simple_c=mylib.tod2map_qu_simple
tod2map_qu_simple_c.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

tod2map_iqu_precon_simple_c=mylib.tod2map_iqu_precon_simple
tod2map_iqu_precon_simple_c.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

tod2map_qu_precon_simple_c=mylib.tod2map_qu_precon_simple
tod2map_qu_precon_simple_c.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

scan_map_c=mylib.scan_map
scan_map_c.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]

tod2cuts_c=mylib.tod2cuts
tod2cuts_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]

cuts2tod_c=mylib.cuts2tod
cuts2tod_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]

set_nthread_c=mylib.set_nthread
set_nthread_c.argtypes=[ctypes.c_int]

get_nthread_c=mylib.get_nthread
get_nthread_c.argtypes=[ctypes.c_void_p]

fill_isobeta_c=mylib.fill_isobeta
fill_isobeta_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]

fill_isobeta_derivs_c=mylib.fill_isobeta_derivs
fill_isobeta_derivs_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]

fill_gauss_derivs_c=mylib.fill_gauss_derivs
fill_gauss_derivs_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]

fill_gauss_src_c=mylib.fill_gauss_src
fill_gauss_src_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]



def y2rj(freq=90):
    """conversion to multiply a y map by to get a Rayleigh-Jeans normalized map
    note that it doesn't have the T_cmb at the end, so the value for low frequencies
    is -2."""
    kb=1.38064852e-16
    h=6.62607004e-27
    T=2.725
    x=freq*1e9*h/kb/T
    
    ex=np.exp(x)
    f=x**2*ex/(ex-1)**2*( x*(ex+1)/(ex-1)-4)
    return f

def planck_g(freq=90):
    """conversion between T_CMB and T_RJ as a function of frequency."""
    kb=1.38064852e-16
    h=6.62607004e-27
    T=2.725
    x=freq*1e9*h/kb/T
    ex=np.exp(x)
    return x**2*ex/( (ex-1)**2)

def report_mpi():
    if have_mpi:
        print('myrank is ',myrank,' out of ',nproc)
    else:
        print('mpi not found')

def barrier():
    if have_mpi:
        comm.barrier()
    else:
        pass

def invsafe(mat,thresh=1e-14):
    u,s,v=np.linalg.svd(mat,0)
    ii=np.abs(s)<thresh*s.max()
    #print ii
    s_inv=1/s
    s_inv[ii]=0
    tmp=np.dot(np.diag(s_inv),u.transpose())
    return np.dot(v.transpose(),tmp)

def tod2map_simple(map,dat,ipix):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    if not(ipix.dtype=='int32'):
        print("Warning - ipix is not int32 in tod2map_simple.  this is likely to produce garbage results.")
    tod2map_simple_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data)

def tod2map_omp(map,dat,ipix):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    if not(ipix.dtype=='int32'):
        print("Warning - ipix is not int32 in tod2map_omp.  this is likely to produce garbage results.")
    tod2map_omp_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data,map.size)

def tod2map_cached(map,dat,ipix):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    if not(ipix.dtype=='int32'):
        print("Warning - ipix is not int32 in tod2map_cached.  this is likely to produce garbage results.")
    tod2map_cached_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data,map.shape[1])
    
def tod2polmap(map,dat,poltag,twogamma,ipix):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    fun=None
    if poltag=='QU':
        fun=tod2map_qu_simple_c
    if poltag=='IQU':
        fun=tod2map_iqu_simple_c
    if poltag=='QU_PRECON':
        fun=tod2map_qu_precon_simple_c
    if poltag=='IQU_PRECON':
        fun=tod2map_iqu_precon_simple_c
    if fun is None:
        print('unrecognized poltag ' + repr(poltag) + ' in tod2polmap.')
    #print('calling ' + repr(fun))
    fun(map.ctypes.data,dat.ctypes.data,twogamma.ctypes.data,ndet,ndata,ipix.ctypes.data)

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
    
def read_fits_map(fname,hdu=0,do_trans=True):
    f=fits.open(fname)
    raw=f[hdu].data
    tmp=raw.copy()
    f.close()
    if do_trans:
        tmp=(tmp.T).copy()
    return tmp
def write_fits_map_wheader(map,fname,header,do_trans=True):
    if do_trans:
        map=(map.T).copy()
    hdu=fits.PrimaryHDU(map,header=header)
    try:
        hdu.writeto(fname,overwrite=True)
    except:
        hdu.writeto(fname,clobber=True)
def get_ft_vec(n):
    x=np.arange(n)
    x[x>n/2]=x[x>n/2]-n
    return x


def set_nthread(nthread):
    set_nthread_c(nthread)

def get_nthread():
    nthread=np.zeros([1,1],dtype='int32')
    get_nthread_c(nthread.ctypes.data)
    return nthread[0,0]


def segs_from_vec(vec,pad=True):
    """ segs_from_vec(vec,pad=True)
    return the starting/stopping points of regions marked False in vec.  For use in e.g. generating
    cuts from a vector/array.  If pad is False, assume vector is already True-padded"""
    #insert input vector into a True-padded vector do make reasoning about starting/stopping points
    #of False regions easier.
    if pad:
        vv=np.ones(len(vec)+2,dtype='bool')
        vv[1:-1]=vec
    else:
        if vec.dtype=='bool':
            vv=vec
        else:
            vv=np.ones(len(vec),dtype='bool')
            vv[:]=vec
    if vv.min()==True:
        nseg=0
        istart=[]
        istop=[]
    else:
        inds=np.where(np.diff(vv))[0]
        assert(len(inds)%2==0)
        nseg=len(inds)//2
        istart=[]
        istop=[]
        for i in range(nseg):
            istart.append(inds[2*i])
            istop.append(inds[2*i+1])
    return nseg,istart,istop

def cut_blacklist(tod_names,blacklist):
    mydict={}
    for nm in tod_names:
        tt=nm.split('/')[-1]
        mydict[tt]=nm
    ncut=0
    for nm in blacklist:
        tt=nm.split('/')[-1]
        #if mydict.has_key(tt):
        if tt in mydict:
            ncut=ncut+1
            del(mydict[tt])
    if ncut>0:
        print('deleted ',ncut,' bad files.')
        mynames=mydict.values()
        mynames.sort()
        return mynames
    else:
        return tod_names 


def find_spikes(dat,inner=1,outer=10,rad=0.25,thresh=8,pad=2):
    #find spikes in a block of timestreams
    n=dat.shape[1];
    ndet=dat.shape[0]
    x=np.arange(n);
    filt1=np.exp(-0.5*x**2/inner**2)
    filt1=filt1+np.exp(-0.5*(x-n)**2/inner**2);
    filt1=filt1/filt1.sum()

    filt2=np.exp(-0.5*x**2/outer**2)
    filt2=filt2+np.exp(-0.5*(x-n)**2/outer**2);
    filt2=filt2/filt2.sum()
    
    filt=filt1-filt2 #make a filter that is the difference of two Gaussians, one narrow, one wide
    filtft=np.fft.rfft(filt)
    datft=np.fft.rfft(dat,axis=1)
    datfilt=np.fft.irfft(filtft*datft,axis=1,n=n)
    jumps=[None]*ndet
    mystd=np.median(np.abs(datfilt),axis=1)
    for i in range(ndet):
        while np.max(np.abs(datfilt[i,:]))>thresh*mystd[i]:
            ind=np.argmax(np.abs(datfilt[i,:]))
            if jumps[i] is None:
                jumps[i]=[ind]
            else:
                jumps[i].append(ind)
            datfilt[i,ind]=0
    return jumps,datfilt
    return mystd
def make_rings_wSlope(edges,cent,vals,map,pixsize=2.0,fwhm=10.0,amps=None,aa=1.0,bb=1.0,rot=0.0):
    xvec=np.arange(map.nx)
    yvec=np.arange(map.ny)
    xvec[map.nx//2:]=xvec[map.nx//2:]-map.nx
    yvec[map.ny//2:]=yvec[map.ny//2:]-map.ny

    xmat=np.repeat([xvec],map.ny,axis=0).transpose()
    ymat=np.repeat([yvec],map.nx,axis=0)

    rmat=np.sqrt(xmat**2+ymat**2)*pixsize
    if isinstance(fwhm,int)|isinstance(fwhm,float):
        sig=fwhm/np.sqrt(8*np.log(2.))
        src_map=np.exp(-0.5*rmat**2./sig**2)
        src_map=src_map/src_map.sum()
    else:
        sig=fwhm[0]/np.sqrt(8*np.log(2))
        src_map=np.exp(-0.5*rmat**2/sig**2)*amps[0]
        for i in range(1,len(fwhm)):
            sig=fwhm[i]/np.sqrt(8*np.log(2))
            src_map=src_map+np.exp(-0.5*rmat**2/sig**2)*amps[i]

        src_map=src_map/src_map.sum()
        beam_area=pixsize**2/src_map.max()
        beam_area=beam_area/3600**2/(360**2/np.pi)
        print('beam_area is ',beam_area*1e9,' nsr')
    nring=len(edges)-1
    rings=np.zeros([nring,map.nx,map.ny])
    mypix=map.wcs.wcs_world2pix(cent[0],cent[1],1)
    print('mypix is ',mypix)

    xvec=np.arange(map.nx)
    yvec=np.arange(map.ny)
    xmat=np.repeat([xvec],map.ny,axis=0).transpose()
    ymat=np.repeat([yvec],map.nx,axis=0)

    srcft=np.fft.fft2(src_map)
    xtr  = (xmat-mypix[0])*np.cos(rot) + (ymat-mypix[1])*np.sin(rot) # Rotate and translate x coords
    ytr  = (ymat-mypix[1])*np.cos(rot) - (xmat-mypix[0])*np.sin(rot) # Rotate and translate y coords
    rmat = np.sqrt( (xtr/aa)**2 + (ytr/bb)**2 ) * pixsize            # Elliptically scale x,y
    myvals = vals[:nring]*1.0   # Get just the values that correspond to rings
    myvals -= np.max(myvals) # Set it such that the maximum value approaches 0
    pk2pk = np.max(myvals) - np.min(myvals)
    myvals -= pk2pk/50.0       # Let's assume we're down about a factor of 50 at the outskirts.

    for i in range(nring):
        #rings[i,(rmat>=edges[i])&(rmat<edges[i+1]=1.0
        if i == nring-1:
            slope=0.0
        else:
            slope = (myvals[i]-myvals[i+1])/(edges[i+1]-edges[i]) # expect positve slope; want negative one.
        rgtinedge = (rmat>=edges[i])
        rfromin   = (rmat-edges[i])
        initline  = rfromin[rgtinedge]*slope
        if vals[i] != 0:
            rings[i,rgtinedge] = (myvals[i] - initline)/myvals[i]  # Should be normalized to 1 now.
        else:
            rings[i,rgtinedge] = 1.0
        rgtoutedge = (rmat>=edges[i+1])
        rings[i,rgtoutedge]=0.0
        myannul = [ c1 and not(c2) for c1,c2 in zip(rgtinedge.ravel(),rgtoutedge.ravel())]
        rannul  = rmat.ravel()[myannul]
        rmin    = (rmat == np.min(rannul))
        rmout   = (rmat == np.max(rannul))
        rings[i,:,:]=np.real(np.fft.ifft2(np.fft.fft2(rings[i,:,:])*srcft))
    return rings

def make_rings(edges,cent,map,pixsize=2.0,fwhm=10.0,amps=None,iswcs=True):
    xvec=np.arange(map.nx)
    yvec=np.arange(map.ny)
    ix=np.int(map.nx/2)
    iy=np.int(map.ny/2)
    xvec[ix:]=xvec[ix:]-map.nx
    yvec[iy:]=yvec[iy:]-map.ny
    #xvec[map.nx/2:]=xvec[map.nx/2:]-map.nx
    #yvec[map.ny/2:]=yvec[map.ny/2:]-map.ny

    xmat=np.repeat([xvec],map.ny,axis=0).transpose()
    ymat=np.repeat([yvec],map.nx,axis=0)

    rmat=np.sqrt(xmat**2+ymat**2)*pixsize
    if isinstance(fwhm,int)|isinstance(fwhm,float):
        sig=fwhm/np.sqrt(8*np.log(2))
        src_map=np.exp(-0.5*rmat**2/sig**2)
        src_map=src_map/src_map.sum()
    else:
        sig=fwhm[0]/np.sqrt(8*np.log(2))
        src_map=np.exp(-0.5*rmat**2/sig**2)*amps[0]
        for i in range(1,len(fwhm)):
            sig=fwhm[i]/np.sqrt(8*np.log(2))
            src_map=src_map+np.exp(-0.5*rmat**2/sig**2)*amps[i]

        src_map=src_map/src_map.sum()
        beam_area=pixsize**2/src_map.max()
        beam_area=beam_area/3600**2/(360**2/np.pi)
        print('beam_area is ',beam_area*1e9,' nsr')
    nring=len(edges)-1
    rings=np.zeros([nring,map.nx,map.ny])
    if iswcs:
        mypix=map.wcs.wcs_world2pix(cent[0],cent[1],1)
    else:
        mypix=cent

    print('mypix is ',mypix)

    xvec=np.arange(map.nx)
    yvec=np.arange(map.ny)
    xmat=np.repeat([xvec],map.ny,axis=0).transpose()
    ymat=np.repeat([yvec],map.nx,axis=0)


    srcft=np.fft.fft2(src_map)
    rmat=np.sqrt( (xmat-mypix[0])**2+(ymat-mypix[1])**2)*pixsize
    for i in range(nring):
        #rings[i,(rmat>=edges[i])&(rmat<edges[i+1]=1.0
        rings[i,(rmat>=edges[i])]=1.0
        rings[i,(rmat>=edges[i+1])]=0.0
        rings[i,:,:]=np.real(np.fft.ifft2(np.fft.fft2(rings[i,:,:])*srcft))
    return rings
        


def find_jumps(dat,width=10,pad=2,thresh=10,rat=0.5):
    #find jumps in a block of timestreams, preferably with the common mode removed
    #width is width in pixels to average over when looking for a jump
    #pad is the length in units of width to mask at beginning/end of timestream
    #thresh is threshold in units of filtered data median absolute deviation to qualify as a jump
    #rat is the ratio of largest neighboring opposite-sign jump to the found jump.  If
    #  there is an opposite-sign jump nearby, the jump finder has probably just picked up a spike.
    n=dat.shape[1]
    ndet=dat.shape[0]

    #make a filter template that is a gaussian with sigma with, sign-flipped in the center
    #so, positive half-gaussian starting from zero, and negative half-gaussian at the end
    x=np.arange(n)
    myfilt=np.exp(-0.5*x**2/width**2)
    myfilt=myfilt-np.exp( (-0.5*(x-n)**2/width**2))
    fac=np.abs(myfilt).sum()/2.0
    myfilt=myfilt/fac

    dat_filt=np.fft.rfft(dat,axis=1)

    myfilt_ft=np.fft.rfft(myfilt)
    dat_filt=dat_filt*np.repeat([myfilt_ft],ndet,axis=0)
    dat_filt=np.fft.irfft(dat_filt,axis=1,n=n)
    dat_filt_org=dat_filt.copy()

    print(dat_filt.shape)
    dat_filt[:,0:pad*width]=0
    dat_filt[:,-pad*width:]=0
    det_thresh=thresh*np.median(np.abs(dat_filt),axis=1)
    dat_dejump=dat.copy()
    jumps=[None]*ndet
    print('have filtered data, now searching for jumps')
    for i in range(ndet):
        while np.max(np.abs(dat_filt[i,:]))>det_thresh[i]:            
            ind=np.argmax(np.abs(dat_filt[i,:]))+1 #+1 seems to be the right index to use
            imin=ind-width
            if imin<0:
                imin=0
            imax=ind+width
            if imax>n:
                imax=n
            val=dat_filt[i,ind]
            if val>0:
                val2=np.min(dat_filt[i,imin:imax])
            else:
                val2=np.max(dat_filt[i,imin:imax])
            
            
            print('found jump on detector ',i,' at sample ',ind)
            if np.abs(val2/val)>rat:
                print('I think this is a spike due to ratio ',np.abs(val2/val))
            else:
                if jumps[i] is None:
                    jumps[i]=[ind]
                else:
                    jumps[i].append(ind)
            #independent of if we think it is a spike or a jump, zap that stretch of the data
            dat_dejump[i,ind:]=dat_dejump[i,ind:]+dat_filt[i,ind]
            dat_filt[i,ind-pad*width:ind+pad*width]=0
        if not(jumps[i] is None):
            jumps[i]=np.sort(jumps[i])
    #return dat_dejump,jumps,dat_filt_org
    return jumps

def fit_jumps_from_cm(dat,jumps,cm,cm_order=1,poly_order=1):
    jump_vals=jumps[:]
    ndet=len(jumps)
    n=dat.shape[1]
    x=np.linspace(-1,1,n)
    m1=np.polynomial.legendre.legvander(x,poly_order)
    m2=np.polynomial.legendre.legvander(x,cm_order-1)
    for i in range(cm_order):
        m2[:,i]=m2[:,i]*cm
    mat=np.append(m1,m2,axis=1)
    npp=mat.shape[1]

    dat_dejump=dat.copy()
    for i in range(ndet):
        if not(jumps[i] is None):
            njump=len(jumps[i])
            segs=np.append(jumps[i],n)
            print('working on detector ',i,' who has ', len(jumps[i]),' jumps with segments ',segs)
            mm=np.zeros([n,npp+njump])
            mm[:,:npp]=mat
            for j in range(njump):
                mm[segs[j]:segs[j+1],j+npp]=1.0
            lhs=np.dot(mm.transpose(),mm)
            #print lhs
            rhs=np.dot(mm.transpose(),dat[i,:].transpose())
            lhs_inv=np.linalg.inv(lhs)
            fitp=np.dot(lhs_inv,rhs)
            jump_vals[i]=fitp[npp:]
            jump_pred=np.dot(mm[:,npp:],fitp[npp:])
            dat_dejump[i,:]=dat_dejump[i,:]-jump_pred


    return dat_dejump
            

    #for i in range(ndet):
def gapfill_eig(dat,cuts,tod=None,thresh=5.0, niter_eig=3, niter_inner=3, insert_cuts=False):
    ndat=dat.shape[1]
    cuts_empty=cuts.copy() #use this to clear out cut samples
    cuts_empty.clear() 
    cuts_cur=cuts.copy()
    cuts_cur.clear()
    for eig_ctr in range(niter_eig):
        tmp=dat.copy()
        cuts_cur.map2tod(tod,tmp,do_add=False)
        mycov=np.dot(tmp,tmp.T)
        ee,vv=np.linalg.eig(mycov)
        mask=ee>thresh*thresh*np.median(ee)
        neig=np.sum(mask)
        print('working with ' + repr(neig) + ' eigenvectors.')
        ee=ee[mask]
        vv=vv[:,mask]
        uu=np.dot(vv.T,tmp)
        lhs=np.dot(uu,uu.T)
        lhs_inv=np.linalg.inv(lhs)
        for iter_ctr in range(niter_inner):
            #in this inner loop, we fit the data 
            rhs=np.dot(tmp,uu.T)
            fitp=np.dot(lhs_inv,rhs.T)
            pred=np.dot(fitp.T,uu)
            cuts_cur.tod2map(tod,pred,do_add=False)
            cuts_cur.map2tod(tod,tmp,do_add=False)
    if insert_cuts:
        cuts_cur.map2tod(dat)
    return cuts_cur
        

def __gapfill_eig_poly(dat,cuts,tod=None,npoly=2, thresh=5.0, niter_eig=3, niter_inner=3):
    assert(1==0) #this code is not yet working.  regular gapfill_eig should work since the polys could
                 #be described by SVD, so SVD modes should look like polys iff they would have been important
    ndat=dat.shape[1]
    if npoly>0:
        xvec=np.linspace(-1,1,ndat)
        polymat=np.polynomial.legendre.legvander(x,npoly-1)
    old_coeffs=None
    cuts_cur=cuts.copy()    
    cuts_cur.clear()
    cuts_empty.cuts.copy()
    cuts_empty.clear()
    for eig_ctr in range(niter_eig):
        tmp=dat.copy()
        cuts_cur.map2tod(tod,tmp,do_add=False) #insert current best-guess solution for the cuts
        if npoly>1:  #if we're fitting polynomials as well as eigenmodes, subtract them off before re-estimating the covariance
            if not(old_coeffs is None):
                tmp=tmp-np.dot(polymat,old_coeffs[neig:,:]).T
        mycov=np.dot(tmp,tmp.T)
        mycov=0.5*(mycov+mycov.T)
        ee,vv=np.linalg.eig(mycov)
        mode_map=ee>thresh*thresh*np.median(ee)
        neig=mode_map.sum()
        mat=np.zeros([ndat,neig+npoly])
        eigs=vv[:,mode_map]
        ts_vecs=np.dot(eigs.T,tmp)
        mat[:,:neig]=ts_vecs.T
        if npoly>0:
            mat[:,neig:]=polymat
        lhs=np.dot(mat.T,mat)
        lhs_inv=np.linalg.inv(lhs)
        #now that we have the vectors we expect to describe our data, do a few rounds
        #of fitting amplitudes to timestream models, subtract that off, assign cuts to zero,
        #and restore the model.  
        tmp=dat.copy()
        for inner_ctr in range(niter_inner):
            cuts_cur.map2tod(tod,tmp)
            rhs=np.dot(tmp,mat)
            fitp=np.dot(lhs_inv,rhs.T)
            pred=np.dot(mat,fitp).T
            

def get_type(nbyte):
    if nbyte==8:
        return np.dtype('float64')
    if nbyte==4:
        return np.dtype('float32')
    if nbyte==-4:
        return np.dtype('int32')
    if nbyte==-8:
        return np.dtype('int64')
    if nbyte==1:
        return np.dtype('str')
    print('Unsupported nbyte ' + repr(nbyte) + ' in get_type')
    return None

def read_octave_struct(fname):
    f=open(fname)
    nkey=np.fromfile(f,'int32',1)[0]
    #print 'nkey is ' + repr(nkey)
    dat={}
    for i in range(nkey):
        key=f.readline().strip()
        #print 'key is ' + key
        ndim=np.fromfile(f,'int32',1)[0]
        dims=np.fromfile(f,'int32',ndim)
        dims=np.flipud(dims)
        #print 'Dimensions of ' + key + ' are ' + repr(dims)
        nbyte=np.fromfile(f,'int32',1)[0]
        #print 'nbyte is ' + repr(nbyte)
        dtype=get_type(nbyte)
        tmp=np.fromfile(f,dtype,dims.prod())
        dat[key]=np.reshape(tmp,dims)
    f.close()
    return dat



def nsphere_vol(npp):
    iseven=(npp%2)==0
    if iseven:
        nn=npp/2
        vol=(np.pi**nn)/np.prod(np.arange(1,nn+1))
    else:
        nn=(npp-1)/2
        vol=2**(nn+1)*np.pi**nn/np.prod(np.arange(1,npp+1,2))
    return vol


def _prime_loop(ln,lp,icur,lcur,vals):
    facs=np.arange(lcur,ln+1e-3,lp[0])
    if len(lp)==1:
        nfac=len(facs)
        if (nfac>0):
            vals[icur:(icur+nfac)]=facs
            icur=icur+nfac
            #print 2**vals[:icur]
        else:
            print('bad facs came from ' + repr([2**lcur,2**ln,2**lp[0]]))
        #print icur
        return icur
    else:
        facs=np.arange(lcur,ln,lp[0])
        for fac in facs:
            icur=_prime_loop(ln,lp[1:],icur,fac,vals)
        return icur
    print('I don''t think I should have gotten here.')
    return icur
                             
        

def find_good_fft_lens(n,primes=[2,3,5,7]):
    lmax=np.log(n+0.5)
    npr=len(primes)
    vol=nsphere_vol(npr)

    r=np.log2(n+0.5)
    lp=np.log2(primes)
    npoint_max=(vol/2**npr)*np.prod(r/lp)+30 #add a bit just to make sure we don't act up for small n
    #print 'npoint max is ',npoint max
    npoint_max=np.int(npoint_max)

    #vals=np.zeros(npoint_max,dtype='int')
    vals=np.zeros(npoint_max)
    icur=0
    icur=_prime_loop(r,lp,icur,0.0,vals)
    assert(icur<=npoint_max)
    myvals=np.asarray(np.round(2**vals[:icur]),dtype='int')
    myvals=np.sort(myvals)
    return myvals
    
    

def _linfit_2mat(dat,mat1,mat2):
    np1=mat1.shape[1]
    np2=mat2.shape[1]
    mm=np.append(mat1,mat2,axis=1)
    lhs=np.dot(mm.transpose(),mm)
    rhs=np.dot(mm.transpose(),dat)
    lhs_inv=np.linalg.inv(lhs)
    fitp=np.dot(lhs_inv,rhs)
    fitp1=fitp[0:np1].copy()
    fitp2=fitp[np1:].copy()
    assert(len(fitp2)==np2)
    return fitp1,fitp2

def fit_mat_vecs_poly_nonoise(dat,mat,order,cm_order=None):
    if cm_order is None:
        cm_order=order
    n=dat.shape[1]
    x=np.linspace(-1,1,n)
    polys=np.polynomial.legendre.legvander(x,order).transpose()
    cm_polys=np.polynomial.legendre.legvander(x,cm_order).transpose()
    v1=np.sum(dat,axis=0)
    v2=np.sum(dat*mat,axis=0)
    rhs1=np.dot(cm_polys,v1)
    rhs2=np.dot(polys,v2)
    ndet=dat.shape[0]
    A1=cm_polys*ndet
    vv=np.sum(mat,axis=0)
    A2=polys*np.repeat([vv],order+1,axis=0)
    A=np.append(A1,A2,axis=0)
    rhs=np.append(rhs1,rhs2)
    lhs=np.dot(A,A.transpose())
    fitp=np.dot(np.linalg.inv(lhs),rhs)
    cm_fitp=fitp[:cm_order+1]
    mat_fitp=fitp[cm_order+1:]
    assert(len(mat_fitp)==(order+1))
    cm_pred=np.dot(cm_fitp,cm_polys)
    tmp=np.dot(mat_fitp,polys)
    mat_pred=np.repeat([tmp],ndet,axis=0)*mat
    pred=cm_pred+mat_pred
    return pred,cm_fitp,mat_fitp,polys



def smooth_spectra(spec,fwhm):
    nspec=spec.shape[0]
    n=spec.shape[1]

    x=np.arange(n)
    sig=fwhm/np.sqrt(8*np.log(2))
    to_conv=np.exp(-0.5*(x/sig)**2)
    tot=to_conv[0]+to_conv[-1]+2*to_conv[1:-1].sum() #r2r normalization
    to_conv=to_conv/tot
    to_conv_ft=mkfftw.fft_r2r(to_conv)
    xtrans=mkfftw.fft_r2r(spec)
    for i in range(nspec):
        xtrans[i,:]=xtrans[i,:]*to_conv_ft
    #return mkfftw.fft_r2r(xtrans)/(2*(xtrans.shape[1]-1)),to_conv
    return xtrans,to_conv_ft
def smooth_many_vecs(vecs,fwhm=20):
    n=vecs.shape[1]
    nvec=vecs.shape[0]
    x=np.arange(n)
    sig=fwhm/np.sqrt(8*np.log(2))
    to_conv=np.exp(-0.5*(x/sig)**2)
    tot=to_conv[0]+to_conv[-1]+2*to_conv[1:-1].sum() #r2r normalization
    to_conv=to_conv/tot
    to_conv_ft=mkfftw.fft_r2r(to_conv)
    xtrans=mkfftw.fft_r2r(vecs)
    for i in range(nvec):
        xtrans[i,:]=xtrans[i,:]*to_conv_ft
    back=mkfftw.fft_r2r(xtrans)
    return back/(2*(n-1))
def smooth_vec(vec,fwhm=20):
    n=vec.size
    x=np.arange(n)
    sig=fwhm/np.sqrt(8*np.log(2))
    to_conv=np.exp(-0.5*(x/sig)**2)
    tot=to_conv[0]+to_conv[-1]+2*to_conv[1:-1].sum() #r2r normalization
    to_conv=to_conv/tot
    to_conv_ft=mkfftw.fft_r2r(to_conv)
    xtrans=mkfftw.fft_r2r(vec)
    back=mkfftw.fft_r2r(xtrans*to_conv_ft)
    return back/2.0/(n-1)


def fit_cm_plus_poly(dat,ord=2,cm_ord=1,niter=2,medsub=False,full_out=False):
    n=dat.shape[1]
    ndet=dat.shape[0]
    if medsub:
        med=np.median(dat,axis=1)        
        dat=dat-np.repeat([med],n,axis=0).transpose()
        
        

    xx=np.arange(n)+0.0
    xx=xx-xx.mean()
    xx=xx/xx.max()

    pmat=np.polynomial.legendre.legvander(xx,ord)
    cm_pmat=np.polynomial.legendre.legvander(xx,cm_ord-1)
    calfacs=np.ones(ndet)*1.0
    dd=dat.copy()
    for i in range(1,niter):
        for j in range(ndet):
            dd[j,:]/=calfacs[j]
            
        cm=np.median(dd,axis=0)
        cm_mat=np.zeros(cm_pmat.shape)
        for i in range(cm_mat.shape[1]):
            cm_mat[:,i]=cm_pmat[:,i]*cm
        fitp_p,fitp_cm=_linfit_2mat(dat.transpose(),pmat,cm_mat)
        pred1=np.dot(pmat,fitp_p).transpose()
        pred2=np.dot(cm_mat,fitp_cm).transpose()
        pred=pred1+pred2
        dd=dat-pred1
        
    if full_out:
        return dd,pred2,cm #if requested, return the modelled CM as well
    return dd


def __run_pcg_old(b,x0,tods,mapset,precon):
    Ax=mapset.dot(x0)

    r=b-Ax
    z=precon*r
    p=z.copy()
    k=0
    zr=r.dot(z)
    x=x0.copy()
    for iter in range(25):
        print(iter,zr)
        Ap=mapset.dot(p)
        pAp=p.dot(Ap)
        alpha=zr/pAp

        x_new=x+p*alpha
        r_new=r-Ap*alpha 
        z_new=precon*r_new
        zr_new=r_new.dot(z_new)
        beta=zr_new/zr
        p_new=z_new+p*beta
        
        p=p_new
        z=z_new
        r=r_new
        zr=zr_new
        x=x_new
    return x

def run_pcg(b,x0,tods,precon=None,maxiter=25,outroot='map',save_iters=[-1],save_ind=0,save_tail='.fits'):
    
    t1=time.time()
    Ax=tods.dot(x0)

    try:
        r=b.copy()
        r.axpy(Ax,-1)
    except:
        r=b-Ax
    if not(precon is None):
        #print('applying precon')
        z=precon*r
    else:
        z=r.copy()
    p=z.copy()
    k=0.0

    zr=r.dot(z)
    x=x0.copy()
    t2=time.time()
    for iter in range(maxiter):
        if myrank==0:
            if iter>0:
                print(iter,zr,alpha,t2-t1,t3-t2,t3-t1)
            else:
                print(iter,zr,t2-t1)
        t1=time.time()
        Ap=tods.dot(p)
        t2=time.time()
        pAp=p.dot(Ap)
        alpha=zr/pAp
        #print('alpha,pAp, and zr  are ' + repr(alpha) + '  ' + repr(pAp) + '  ' + repr(zr))
        try:
            x_new=x.copy()
            x_new.axpy(p,alpha)
        except:
            x_new=x+p*alpha

        try:
            r_new=r.copy()
            r_new.axpy(Ap,-alpha)
        except:
            r_new=r-Ap*alpha
        if not(precon is None):
            #print('applying precon')
            z_new=precon*r_new
        else:
            z_new=r_new.copy()
        zr_new=r_new.dot(z_new)
        beta=zr_new/zr
        try:
            p_new=z_new.copy()
            p_new.axpy(p,beta)
        except:
            p_new=z_new+p*beta
        
        p=p_new
        z=z_new
        r=r_new
        zr=zr_new
        x=x_new
        t3=time.time()
        if iter in save_iters:
            if myrank==0:
                x.maps[save_ind].write(outroot+'_'+repr(iter)+save_tail)
    return x

def run_pcg_wprior(b,x0,tods,prior=None,precon=None,maxiter=25,outroot='map',save_iters=[-1],save_ind=0,save_tail='.fits'):
    #least squares equations in the presence of a prior - chi^2 = (d-Am)^T N^-1 (d-Am) + (p-m)^T Q^-1 (p-m)
    #where p is the prior target for parameters, and Q is the variance.  The ensuing equations are
    #(A^T N-1 A + Q^-1)m = A^T N^-1 d + Q^-1 p.  For non-zero p, it is assumed you have done this already and that 
    #b=A^T N^-1 d + Q^-1 p
    #to have a prior then, whenever we call Ax, just a Q^-1 x to Ax.
    t1=time.time()
    Ax=tods.dot(x0)    
    if not(prior is None):
        #print('applying prior')
        prior.apply_prior(x0,Ax) 
    try:
        r=b.copy()
        r.axpy(Ax,-1)
    except:
        r=b-Ax
    if not(precon is None):
        z=precon*r
    else:
        z=r.copy()
    p=z.copy()
    k=0.0

    zr=r.dot(z)
    x=x0.copy()
    t2=time.time()
    for iter in range(maxiter):
        if myrank==0:
            if iter>0:
                print(iter,zr,alpha,t2-t1,t3-t2,t3-t1)
            else:
                print(iter,zr,t2-t1)
            sys.stdout.flush()
        t1=time.time()
        Ap=tods.dot(p)
        if not(prior is None):
            #print('applying prior')
            prior.apply_prior(p,Ap)
        t2=time.time()
        pAp=p.dot(Ap)
        alpha=zr/pAp
        try:
            x_new=x.copy()
            x_new.axpy(p,alpha)
        except:
            x_new=x+p*alpha

        try:
            r_new=r.copy()
            r_new.axpy(Ap,-alpha)
        except:
            r_new=r-Ap*alpha
        if not(precon is None):
            z_new=precon*r_new
        else:
            z_new=r_new.copy()
        zr_new=r_new.dot(z_new)
        beta=zr_new/zr
        try:
            p_new=z_new.copy()
            p_new.axpy(p,beta)
        except:
            p_new=z_new+p*beta
        
        p=p_new
        z=z_new
        r=r_new
        zr=zr_new
        x=x_new
        t3=time.time()
        if iter in save_iters:
            if myrank==0:
                x.maps[save_ind].write(outroot+'_'+repr(iter)+save_tail)

    return x

    

def run_pcg_wprior_old(b,x0,tods,prior,precon=None,maxiter=25):
    t1=time.time()
    Ax=tods.dot(x0)
    #prior.apply_prior(Ax,x0)
    flub=prior.apply_prior(x0.maps[0].map)
    print('means of flub and Ax are ',np.mean(np.abs(Ax.maps[0].map)),np.mean(np.abs(flub)))
    Ax.maps[0].map=Ax.maps[0].map+prior.apply_prior(x0.maps[0].map)
    try:
        r=b.copy()
        r.axpy(Ax,-1)
    except:
        r=b-Ax
    if not(precon is None):
        z=precon*r
    else:
        z=r.copy()
    p=z.copy()
    k=0.0

    zr=r.dot(z)
    x=x0.copy()
    t2=time.time()
    for iter in range(maxiter):
        if myrank==0:
            if iter>0:
                print(iter,zr,alpha,t2-t1,t3-t2,t3-t1)
            else:
                print(iter,zr,t2-t1)
        t1=time.time()
        Ap=tods.dot(p)
        fwee=prior.apply_prior(p.maps[0].map)
        #print 'means are ',np.mean(np.abs(Ap.maps[0].map)),np.mean(np.abs(fwee))
        Ap.maps[0].map=Ap.maps[0].map+fwee
        #prior.apply_prior(Ap,p)
        t2=time.time()
        pAp=p.dot(Ap)
        alpha=zr/pAp
        try:
            x_new=x.copy()
            x_new.axpy(p,alpha)
        except:
            x_new=x+p*alpha

        try:
            r_new=r.copy()
            r_new.axpy(Ap,-alpha)
        except:
            r_new=r-Ap*alpha
        if not(precon is None):
            z_new=precon*r_new
        else:
            z_new=r_new.copy()
        zr_new=r_new.dot(z_new)
        beta=zr_new/zr
        try:
            p_new=z_new.copy()
            p_new.axpy(p,beta)
        except:
            p_new=z_new+p*beta
        
        p=p_new
        z=z_new
        r=r_new
        zr=zr_new
        x=x_new
        t3=time.time()
    return x

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

class null_precon:
    def __init__(self):
        self.isnull=True
    def __add__(self,val):
        return val
    def __mul__(self,val):
        return val

def scaled_airmass_from_el(mat):
    airmass=1/np.cos(mat)
    airmass=airmass-airmass.mean()
    #airmass=airmass/np.std(airmass)
    return airmass

class tsGeneric:
    def __init__(self,tod=None):
        self.fname=tod.info['fname']
    def __mul__(self,to_mul):
        #print('calling mul')
        tt=self.copy()
        tt.params=self.params*to_mul.params
        return tt
    def clear(self):
        self.params[:]=0
    def dot(self,common=None):
        if common is None:
            return np.sum(self.params*self.params)
        else:
            return np.sum(self.params*common.params)
    def axpy(self,common,a):
        self.params=self.params+a*common.params
    def apply_prior(self,x,Ax):
        Ax.params=Ax.params+self.params*x.params
    def copy(self):
        return copy.deepcopy(self)
    def write(self,fname=None):
        pass
class tsVecs(tsGeneric):
    def __init__(self,tod,vecs):
        self.vecs=vecs
        #self.ndet=tod.info['dat_calib'].shape[0]
        self.ndet=tod.get_data_dims()[0]
        self.vecs=vecs
        self.nvec=vecs.shape[0]
        self.params=np.zeros([self.nvec,self.ndet])
    def tod2map(self,tod,mat=None,do_add=True,do_omp=False):
        if mat is None:
            #mat=tod.info['dat_calib']
            mat=tod.get_data()
        if do_add:
            self.params[:]=self.params[:]+np.dot(self.vecs,mat.T)
        else:
            self.params[:]=np.dot(self.vecs,mat.T)
    def map2tod(self,tod,mat=None,do_add=True,do_omp=False):
        if mat is None:
            #mat=tod.info['dat_calib']
            mat=tod.get_data()
        if do_add:
            mat[:]=mat[:]+np.dot(self.params.T,self.vecs)
        else:
            mat[:]=np.dot(self.params.T,self.vecs)    

class tsPoly(tsVecs):
    def __init__(self,tod,order=10):
        self.fname=tod.info['fname']

        #self.ndata=tod.info['dat_calib'].shape[1]
        dims=tod.get_data_dims()
        self.ndata=dims[1]
        self.order=order
        #self.ndet=tod.info['dat_calib'].shape[0]
        self.ndet=dims[0]
        xvec=np.linspace(-1,1,self.ndata)
        self.vecs=(np.polynomial.legendre.legvander(xvec,order).T).copy()
        self.nvec=self.vecs.shape[0]
        self.params=np.zeros([self.nvec,self.ndet])


def partition_interval(start,stop,seg_len=100,round_up=False):
    #print('partitioning ',start,stop,seg_len)
    #make sure we behave correctly if the interval is shorter than the desired segment
    if (stop-start)<=seg_len:
        return np.asarray([start,stop],dtype='int')
    nseg=(stop-start)//seg_len
    if nseg*seg_len<(stop-start):
        if round_up:
            nseg=nseg+1
    seg_len=(stop-start)//nseg
    nextra=(stop-start)-seg_len*nseg
    inds=np.arange(start,stop+1,seg_len)
    if nextra>0:
        vec=np.zeros(len(inds),dtype='int')
        vec[1:nextra+1]=1
        vec=np.cumsum(vec)
        inds=inds+vec
    return inds
    
def split_partitioned_vec(start,stop,breaks=[],seg_len=100):
    if len(breaks)==0:
        return partition_interval(start,stop,seg_len)
    if breaks[0]==start:
        breaks=breaks[1:]
        if len(breaks)==0:
            return partition_interval(start,stop,seg_len)
    if breaks[-1]==stop:
        breaks=breaks[:-1]
        if len(breaks)==0:
            return partition_interval(start,stop,seg_len)
    breaks=np.hstack([start,breaks,stop])
    nseg=len(breaks)-1
    segs=[None]*(nseg)
    for i in range(nseg):
        inds=partition_interval(breaks[i],breaks[i+1],seg_len)
        if i<(nseg-1):
            inds=inds[:-1]
        segs[i]=inds
    segs=np.hstack(segs)
    return segs

#breaks,stop,start=0,seg_len=100)
class tsStripes(tsGeneric):
    def __init__(self,tod,seg_len=500,do_slope=False,tthresh=10):
        dims=tod.get_data_dims()
        tvec=tod.get_tvec()
        dt=np.median(np.diff(tvec))        
        splits=np.where(np.abs(np.diff(tvec))>tthresh*dt)[0]

        dims=tod.get_data_dims()
        inds=split_partitioned_vec(0,dims[1],splits,seg_len)
        
        self.inds=inds
        self.splits=splits
        self.nseg=len(self.inds)-1
        self.params=np.zeros([dims[0],self.nseg])
    def tod2map(self,tod,dat=None,do_add=True,do_omp=False):
        if dat is None:
            print('need dat in tod2map destriper')
            return
        minkasi_nb.tod2map_destriped(dat,self.params,self.inds,do_add)
    def map2tod(self,tod,dat=None,do_add=True,do_omp=False):
        if dat is None:
            print('need dat in map2tod destriper')
            return
        minkasi_nb.map2tod_destriped(dat,self.params,self.inds,do_add)
    def copy(self):
        return copy.deepcopy(self)
    def set_prior_from_corr(self,corrvec,thresh=0.5):
        assert(corrvec.shape[0]==self.params.shape[0])
        n=self.params.shape[1]
        corrvec=corrvec[:,:n].copy()
        corrft=mkfftw.fft_r2r(corrvec)
        if thresh>0:
            for i in range(corrft.shape[0]):
                tt=thresh*np.median(corrft[i,:])
                ind=corrft[i,:]<tt
                corrft[i,ind]=tt
        self.params=1.0/corrft/(2*(n-1))
    def apply_prior(self,x,Ax):
        xft=mkfftw.fft_r2r(x.params)        
        Ax.params=Ax.params+mkfftw.fft_r2r(xft*self.params)
        
class tsStripes_old(tsGeneric):
    def __init__(self,tod,seg_len=100,do_slope=True):
        dims=tod.get_data_dims()
        nseg=dims[1]//seg_len
        if nseg*seg_len<dims[1]:
            nseg=nseg+1
        #this says to connect segments with straight lines as
        #opposed to simple horizontal offsets
        if do_slope:
            nseg=nseg+1
        self.nseg=nseg
        self.seg_len=seg_len
        self.params=np.zeros([dims[0],self.nseg])
        self.do_slope=do_slope
    def tod2map(self,tod,dat=None,do_add=True,do_omp=False):
        if dat is None:
            dat=tod.get_data()
        tmp=np.zeros(self.params.shape)
        if self.do_slope:
            vec=np.arange(self.seg_len)/self.seg_len
            vec2=1-vec
            vv=np.vstack([vec2,vec])
            nseg=dat.shape[1]//self.seg_len
            imax=nseg
            if nseg*self.seg_len<dat.shape[1]:
                have_extra=True
                nextra=dat.shape[1]-imax*self.seg_len
            else:
                have_extra=False
                nseg=nseg-1
                nextra=0
            for i in range(imax):
                #tmp[:,i:i+2]=tmp[:,i:i+2]+dat[:,i*self.seg_len:(i+1)*self.seg_len]@(vv.T)
                tmp[:,i:i+2]=tmp[:,i:i+2]+np.dot(dat[:,i*self.seg_len:(i+1)*self.seg_len],(vv.T))
            if have_extra:
                vec=np.arange(nextra)/nextra
                vec2=1-vec
                vv=np.vstack([vec2,vec])
                #tmp[:,-2:]=tmp[:,-2:]+dat[:,self.seg_len*nseg:]@(vv.T)
                tmp[:,-2:]=tmp[:,-2:]+np.dot(dat[:,self.seg_len*nseg:],(vv.T))

        else:
            nseg=dat.shape[1]//self.seg_len
            if nseg*self.seg_len<dat.shape[1]:
                nseg=nseg+1
            vec=np.zeros(nseg*self.seg_len)
            ndet=dat.shape[0]
            ndat=dat.shape[1]
            for i in range(ndet):
                vec[:ndat]=dat[i,:]
                vv=np.reshape(vec,[nseg,self.seg_len])
                tmp[i,:]=np.sum(vv,axis=1)
        if do_add:
            self.params[:]=self.params[:]+tmp
        else:
            self.params[:]=tmp

    def map2tod(self,tod,dat=None,do_add=True,do_omp=False):
        tmp=tod.get_empty()
        ndet=tmp.shape[0] 
        ndat=tmp.shape[1]
            
        if self.do_slope:
            vec=np.arange(self.seg_len)/self.seg_len
            vec2=1-vec
            vv=np.vstack([vec2,vec])
            nseg=tmp.shape[1]//self.seg_len
            imax=nseg
            if imax*self.seg_len<tmp.shape[1]:
                have_extra=True
                nextra=tmp.shape[1]-imax*self.seg_len
            else:
                have_extra=False
                nseg=nseg-1
                #imax=imax+1
            
            for i in range(imax):
                #tmp[:,i*self.seg_len:(i+1)*self.seg_len]=self.params[:,i:i+2]@vv
                tmp[:,i*self.seg_len:(i+1)*self.seg_len]=np.dot(self.params[:,i:i+2],vv)
            if have_extra:
                vec=np.arange(nextra)/nextra
                vec2=1-vec
                vv=np.vstack([vec2,vec])
                #tmp[:,self.seg_len*nseg:]=self.params[:,-2:]@vv
                tmp[:,self.seg_len*nseg:]=np.dot(self.params[:,-2:],vv)
        else:
            ndet=tmp.shape[0] 
            ndat=tmp.shape[1]
            for i in range(ndet):
                pars=self.params[i,:]
                mymod=np.repeat(pars,self.seg_len)
                tmp[i,:]=mymod[:ndat]
                
        if dat is None:
            dat=tmp
            return dat
        else:
            if do_add:
                dat[:]=dat[:]+tmp
            else:
                dat[:]=tmp

class tsBinnedAz(tsGeneric):
    def __init__(self,ndet=2,lims=[0,2*np.pi],nbin=360):
        self.params=np.zeros([ndet,nbin])
        self.lims=[lims[0],lims[1]]
        self.nbin=nbin
        
    def map2tod(self,tod,dat=None,do_add=True,do_omp=False):
        if dat is None:
            dat=tod.get_data()
        minkasi_nb.map2tod_binned_det(dat,self.params,tod.info['az'],self.lims,self.nbin,do_add)
    def tod2map(self,tod,dat=None,do_add=True,do_omp=False):
        if dat is None:
            dat=tod.get_data()
        minkasi_nb.tod2map_binned_det(dat,self.params,tod.info['az'],self.lims,self.nbin.do_add)
class tsDetAz(tsGeneric):
    def __init__(self,tod,npoly=4):
        if isinstance(tod,tsDetAz): #we're starting a new instance from an old one, e.g. from copy
            self.fname=tod.fname
            self.az=tod.az
            self.azmin=tod.azmin
            self.azmax=tod.azmax
            self.npoly=tod.npoly
            self.ndet=tod.ndet
        else:
            self.fname=tod.info['fname']
            self.az=tod.info['AZ']
            self.azmin=np.min(self.az)
            self.azmax=np.max(self.az)
            self.npoly=npoly
            #self.ndet=tod.info['dat_calib'].shape[0]            
            self.ndet=tod.get_ndet()
        #self.params=np.zeros([self.ndet,self.npoly])
        self.params=np.zeros([self.ndet,self.npoly-1])
    def _get_polys(self):
        polys=np.zeros([self.npoly,len(self.az)])
        polys[0,:]=1.0
        az_scale= (self.az-self.azmin)/(self.azmax-self.azmin)*2.0-1.0
        if self.npoly>1:
            polys[1,:]=az_scale
        for i in range(2,self.npoly):
            polys[i,:]=2*az_scale*polys[i-1,:]-polys[i-2,:]
        polys=polys[1:,:].copy()
        return polys
    def map2tod(self,tod,dat=None,do_add=True,do_omp=False):
        if dat is None:
            #dat=tod.info['dat_calib']
            dat=tod.get_data()
        if do_add:
            dat[:]=dat[:]+np.dot(self.params,self._get_polys())
        else:
            dat[:]=np.dot(self.params,self._get_polys())
    def tod2map(self,tod,dat=None, do_add=True,do_omp=False):
        if dat is None:
            #dat=tod.info['dat_calib']
            dat=tod.get_data()
        if do_add:
            #print("params shape is ",self.params.shape)
            #self.params[:]=self.params[:]+np.dot(self._get_polys(),dat)
            self.params[:]=self.params[:]+np.dot(dat,self._get_polys().T)
        else:
            #self.params[:]=np.dot(self._get_polys(),dat)
            self.params[:]=np.dot(dat,self._get_polys().T)

        
class tsAirmass:
    def __init__(self,tod=None,order=3):
        if tod is None:
            self.sz=np.asarray([0,0],dtype='int')
            self.params=np.zeros(1)
            self.fname=''
            self.order=0
            self.airmass=None
        else:
            #self.sz=tod.info['dat_calib'].shape
            self.sz=tod.get_data_dims()
            self.fname=tod.info['fname']
            self.order=order
            self.params=np.zeros(order)
            if not('apix' in tod.info.keys()):
                #tod.info['apix']=scaled_airmass_from_el(tod.info['elev'])
                self.airmass=scaled_airmass_from_el(tod.info['elev'])
            else:
                self.airmass=tod.info['apix']
    def copy(self,copyMat=False):
        cp=tsAirmass()
        cp.sz=self.sz
        cp.params=self.params.copy()
        cp.fname=self.fname
        cp.order=self.order
        if copyMat:
            cp.airmass=self.airmass.copy()
        else:
            cp.airmass=self.airmass  #since this shouldn't change, use a pointer to not blow up RAM
        return cp
    def clear(self):
        self.params[:]=0.0
    def dot(self,ts):
        return np.sum(self.params*ts.params)
    def axpy(self,ts,a):
        self.params=self.params+a*ts.params
    def _get_current_legmat(self):
        x=np.linspace(-1,1,self.sz[1])
        m1=np.polynomial.legendre.legvander(x,self.order)
        return m1
    def _get_current_model(self):
        x=np.linspace(-1,1,self.sz[1])
        m1=self._get_current_legmat()
        poly=np.dot(m1,self.params)
        mat=np.repeat([poly],self.sz[0],axis=0)
        mat=mat*self.airmass
        return mat

    def tod2map(self,tod,dat,do_add=True,do_omp=False): 
        tmp=np.zeros(self.order)
        for i in range(self.order):
            #tmp[i]=np.sum(tod.info['apix']**(i+1)*dat)
            tmp[i]=np.sum(self.airmass**(i+1)*dat)
        if do_add:
            self.params[:]=self.params[:]+tmp
        else:
            self.params[:]=tmp
        #poly=self._get_current_legmat()
        #vec=np.sum(dat*self.airmass,axis=0)
        #atd=np.dot(vec,poly)
        #if do_add:
        #    self.params[:]=self.params[:]+atd
        #else:
        #    self.params[:]=atd

    def map2tod(self,tod,dat,do_add=True,do_omp=False):
        mat=0.0
        for i in range(self.order):
            #mat=mat+self.params[i]*tod.info['apix']**(i+1)
            mat=mat+self.params[i]*self.airmass**(i+1)

        #mat=self._get_current_model()
        if do_add:
            dat[:]=dat[:]+mat
        else:
            dat[:]=mat
    def __mul__(self,to_mul):
        tt=self.copy()
        tt.params=self.params*to_mul.params
        return tt
    def write(self,fname=None):
        pass

class __tsAirmass_old:
    def __init__(self,tod=None,order=5):
        if tod is None:
            self.sz=np.asarray([0,0],dtype='int')
            self.params=np.zeros(1)
            self.fname=''
            self.order=0
            self.airmass=None
        else:
            #self.sz=tod.info['dat_calib'].shape
            self.sz=tod.get_data_dims()
            self.fname=tod.info['fname']
            self.order=order
            self.params=np.zeros(order+1)
            self.airmass=scaled_airmass_from_el(tod.info['elev'])
    def copy(self,copyMat=False):
        cp=tsAirmass()
        cp.sz=self.sz
        cp.params=self.params.copy()
        cp.fname=self.fname
        cp.order=self.order
        if copyMat:
            cp.airmass=self.airmass.copy()
        else:
            cp.airmass=self.airmass  #since this shouldn't change, use a pointer to not blow up RAM
        return cp
    def clear(self):
        self.params[:]=0.0
    def dot(self,ts):
        return np.sum(self.params*ts.params)
    def axpy(self,ts,a):
        self.params=self.params+a*ts.params
    def _get_current_legmat(self):
        x=np.linspace(-1,1,self.sz[1])
        m1=np.polynomial.legendre.legvander(x,self.order)
        return m1
    def _get_current_model(self):
        x=np.linspace(-1,1,self.sz[1])
        m1=self._get_current_legmat()
        poly=np.dot(m1,self.params)
        mat=np.repeat([poly],self.sz[0],axis=0)
        mat=mat*self.airmass
        return mat

    def tod2map(self,tod,dat,do_add=True,do_omp=False):
        poly=self._get_current_legmat()
        vec=np.sum(dat*self.airmass,axis=0)
        atd=np.dot(vec,poly)
        if do_add:
            self.params[:]=self.params[:]+atd
        else:
            self.params[:]=atd

    def map2tod(self,tod,dat,do_add=True,do_omp=False):
        mat=self._get_current_model()
        if do_add:
            dat[:]=dat[:]+mat
        else:
            dat[:]=mat
        def __mul__(self,to_mul):
            tt=self.copy()
            tt.params=self.params*to_mul.params
    def __mul__(self,to_mul):
        tt=self.copy()
        tt.params=self.params*to_mul.params
        return tt
    def write(self,fname=None):
        pass

class tsCommon:
    def __init__(self,tod=None,*args,**kwargs):
        if tod is None:
            self.sz=np.asarray([0,0],dtype='int')
            self.params=np.zeros(1)
            self.fname=''
        else:
            #self.sz=tod.info['dat_calib'].shape
            self.sz=tod.get_data_dims()
            self.params=np.zeros(self.sz[1])
            self.fname=tod.info['fname']
    def copy(self):
        cp=tsCommon()
        try:
            cp.sz=self.sz.copy() 
        except:#if the size doesn't have a copy function, then it's probably a number you can just assign
            cp.sz=self.sz
            cp.fname=self.fname
            cp.params=self.params.copy()
            return cp
    def clear(self):
        self.params[:]=0.0
    def dot(self,common=None):
        if common is None:
            return np.dot(self.params,self.params)
        else:
            return np.dot(self.params,common.params)
    def axpy(self,common,a):
        self.params=self.params+a*common.params
        
    def tod2map(self,tod,dat,do_add=True,do_omp=False):
        #assert(self.fname==tod.info['fname']
        nm=tod.info['fname']
        if do_add==False:
            self.clear()
        self.params[:]=self.params[:]+np.sum(dat,axis=0)
    def map2tod(self,tod,dat,do_add=True,do_omp=True):
        nm=tod.info['fname']
        dat[:]=dat[:]+np.repeat([self.params],dat.shape[0],axis=0)
    def write(self,fname=None):
        pass
    def __mul__(self,to_mul):
        tt=self.copy()
        tt.params=self.params*to_mul.params
        return tt
class detOffset:
    def __init__(self,tod=None):
        if tod is None:
            self.sz=1
            self.params=np.zeros(1)
            self.fname=''
        else:
            #self.sz=tod.info['dat_calib'].shape[0]
            self.sz=tod.get_ndet()
            self.params=np.zeros(self.sz)
            self.fname=tod.info['fname']
    def copy(self):
        cp=detOffset()
        cp.sz=self.sz
        cp.params=self.params.copy()
        cp.fname=self.fname
        return cp
    def clear(self):
        self.params[:]=0
    def dot(self,other=None):
        if other is None:
            return np.dot(self.params,self.params)
        else:
            return np.dot(self.params,other.params)
    def axpy(self,common,a):
        self.params=self.params+a*common.params
    def tod2map(self,tod,dat,do_add=True,do_omp=False):
        if do_add==False:
            self.clear()
        self.params[:]=self.params[:]+np.sum(dat,axis=1)
    def map2tod(self,tod,dat,do_add=True,do_omp=False):
        if do_add==False:
            dat[:]=0
        dat[:]=dat[:]+np.repeat([self.params],dat.shape[1],axis=0).transpose()
    def write(self,fname=None):
        pass
    def __mul__(self,to_mul):
        tt=self.copy()
        tt.params=self.params*to_mul.params
        return tt

class tsCalib:
    def __init__(self,tod=None,model=None):
        if tod is None:
            self.sz=1
            self.params=np.zeros(1)
            self.fname=''
            self.pred=None
        else:
            #self.sz=tod.info['dat_calib'].shape[0]
            self.sz=tod.get_ndet()
            self.params=np.zeros(self.sz)
            self.pred=model[tod.info['fname']].copy()
            self.fname=tod.info['fname']
    def copy(self):
        cp=tsCalib()
        cp.sz=self.sz
        cp.params=self.params.copy()
        cp.fname=self.fname
        cp.pred=self.pred
        return cp
    def clear(self):
        self.params[:]=0
    def dot(self,other=None):
        if other is None:
            return np.dot(self.params,self.params)
        else:
            return np.dot(self.params,other.params)
    def axpy(self,common,a):
        self.params=self.params+a*common.params
    def tod2map(self,tod,dat,do_add=True,do_omp=False):
        if do_add==False:
            self.clear()
        if self.pred.ndim==1:
            self.params[:]=self.params[:]+np.dot(dat,self.pred)
        else:
            self.params[:]=self.params[:]+np.sum(dat*self.pred,axis=1)
    def map2tod(self,tod,dat,do_add=True,do_omp=False):
        if do_add==False:
            dat[:]=0
        if self.pred.ndim==1:
            dat[:]=dat[:]+np.outer(self.params,self.pred)
        else:
            dat[:]=dat[:]+(self.pred.transpose()*self.params).transpose()
    def write(self,fname=None):
        pass
    def __mul__(self,to_mul):
        tt=self.copy()
        tt.params=self.params*to_mul.params
        return tt
    
    
    
class tsModel:
    def __init__(self,todvec=None,modelclass=None,*args,**kwargs):
        self.data={}
        if todvec is None:
            return
        for tod in todvec.tods:
            nm=tod.info['fname']
            self.data[nm]=modelclass(tod,*args,**kwargs)
    def copy(self):
        new_tsModel=tsModel()
        for nm in self.data.keys():
            new_tsModel.data[nm]=self.data[nm].copy()
        return new_tsModel

    def tod2map(self,tod,dat,do_add=True,do_omp=False):
        nm=tod.info['fname']
        if do_add==False:
            self.clear()
        self.data[nm].tod2map(tod,dat,do_add,do_omp)
    def map2tod(self,tod,dat,do_add=True,do_omp=True):
        nm=tod.info['fname']
        if do_add==False:
            dat[:]=0.0
        self.data[nm].map2tod(tod,dat,do_add,do_omp)

    def apply_prior(self,x,Ax):
        for nm in self.data.keys():
            self.data[nm].apply_prior(x.data[nm],Ax.data[nm])
    def dot(self,tsmodels=None):
        tot=0.0
        for nm in self.data.keys():
            if tsmodels is None:
                tot=tot+self.data[nm].dot(self.data[nm])
            else:
                #if tsmodels.data.has_key(nm):
                if nm in tsmodels.data:
                    tot=tot+self.data[nm].dot(tsmodels.data[nm])
                else:
                    print('error in tsModel.dot - missing key ',nm)
                    assert(1==0)  #pretty sure we want to crash if missing names
        if have_mpi:
            tot=comm.allreduce(tot)
        return tot
    def clear(self):
        for nm in self.data.keys():
            self.data[nm].clear()
    def axpy(self,tsmodel,a):
        for nm in self.data.keys():
            self.data[nm].axpy(tsmodel.data[nm],a)
    def __mul__(self,tsmodel): #this is used in preconditioning - need to fix if ts-based preconditioning is desired        
        tt=self.copy()
        for nm in self.data.keys():
            tt.data[nm]=self.data[nm]*tsmodel.data[nm]
        return tt
        #for nm in tt.data.keys():
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
    def __init__(self,todvec=None,todtags=None,modelclass=None,tag='ts_multi_model',*args,**kwargs):        
        self.data={}
        self.tag=tag
        if not(todtags is None):
            alltags=comm.allgather(todtags)
            alltags=np.hstack(alltags)
            alltags=np.unique(alltags)
            if not(modelclass is None):
                for mytag in alltags:
                    self.data[mytag]=modelclass(*args,**kwargs)
            if not(todvec is None):
                for i,tod in enumerate(todvec.tods):
                    tod.info[tag]=todtags[i]
    def copy(self):
        return copy.deepcopy(self)
    def tod2map(self,tod,dat,do_add=True,do_omp=False):
        self.data[tod.info[self.tag]].tod2map(tod,dat,do_add,do_omp)
    def map2tod(self,tod,dat,do_add=True,do_omp=False):
        self.data[tod.info[self.tag]].map2tod(tod,dat,do_add,do_omp)
    def dot(self,tsmodels=None):
        tot=0.0
        for nm in self.data.keys():
            if tsmodels is None:
                tot=tot+self.data[nm].dot(self.data[nm])
            else:
                if nm in tsmodels.data:
                    tot=tot+self.data[nm].dot(tsmodels.data[nm])
                else:
                    print('error in tsMultiModel.dot - missing key ',nm)
                    assert(1==0)
        return tot
class Mapset:
    def __init__(self):
        self.nmap=0
        self.maps=[]
    def add_map(self,map):
        self.maps.append(map.copy())
        self.nmap=self.nmap+1
    def clear(self):
        for i in range(self.nmap):
            self.maps[i].clear()
    def copy(self):
        new_mapset=Mapset()
        for i in range(self.nmap):
            new_mapset.add_map(self.maps[i].copy())
        return new_mapset
    def dot(self,mapset):
        tot=0.0
        for i in range(self.nmap):
            tot=tot+self.maps[i].dot(mapset.maps[i])
        return tot
    def axpy(self,mapset,a):
        for i in range(self.nmap):
            self.maps[i].axpy(mapset.maps[i],a)
    def __add__(self,mapset):
        mm=self.copy()
        mm.axpy(mapset,1.0)
        return mm

    def __sub__(self,mapset):
        mm=self.copy()
        mm.axpy(mapset,-1.0)
        return mm
    def __mul__(self,mapset):
        #mm=self.copy()
        mm=mapset.copy()
        #return mm
        for i in range(self.nmap):
            #print('callin mul on map ',i)
            mm.maps[i]=self.maps[i]*mapset.maps[i]
        return mm
    def get_caches(self):
        for i in range(self.nmap):
            self.maps[i].get_caches()
    def clear_caches(self):
        for i in range(self.nmap):
            self.maps[i].clear_caches()
    def apply_prior(self,x,Ax):
        for i in range(self.nmap):
            if not(self.maps[i] is None):
                try:                    
                    if self.maps[i].isglobal_prior:
                        #print('applying global prior')
                        self.maps[i].apply_prior(x,Ax)
                    else:
                        self.maps[i].apply_prior(x.maps[i],Ax.maps[i])
                except:
                    #print('going through exception')
                    self.maps[i].apply_prior(x.maps[i],Ax.maps[i])
    def mpi_reduce(self):
        if have_mpi:
            for map in self.maps:
                map.mpi_reduce()
#class Cuts:
#    def __init__(self,tod):
#        self.tag=tod.info['tag']
#        self.ndet=tod.info['dat_calib'].shape[0]
#        self.cuts=[None]*self.ndet
#
#class CutsVec:
#    def __init__(self,todvec):
#        self.ntod=todvec.ntod
#        self.cuts=[None]*self.ntod
#        for tod in todvec.tods:
#            self.cuts[tod.info['tag']]=Cuts(tod)
            
class SkyMap:
    def __init__(self,lims,pixsize=0,proj='CAR',pad=2,primes=None,cosdec=None,nx=None,ny=None,mywcs=None,ref_equ=False):        
        if mywcs is None:
            assert(pixsize!=0) #we had better have a pixel size if we don't have an incoming WCS that contains it
            self.wcs=get_wcs(lims,pixsize,proj,cosdec,ref_equ)            
        else:
            self.wcs=mywcs
            pixsize_use=mywcs.wcs.cdelt[1]*np.pi/180
            #print('pixel size from wcs and requested are ',pixsize_use,pixsize,100*(pixsize_use-pixsize)/pixsize)
            pixsize=pixsize_use
            
        corners=np.zeros([4,2])
        corners[0,:]=[lims[0],lims[2]]
        corners[1,:]=[lims[0],lims[3]]
        corners[2,:]=[lims[1],lims[2]]
        corners[3,:]=[lims[1],lims[3]]
        pix_corners=self.wcs.wcs_world2pix(corners*180/np.pi,1)
        pix_corners=np.round(pix_corners)

        #print pix_corners
        #print type(pix_corners)
        #if pix_corners.min()<0.5:
        if pix_corners.min()<-0.5:
            print('corners seem to have gone negative in SkyMap projection.  not good, you may want to check this.')
        if True: #try a patch to fix the wcs xxx
            if nx is None:
                nx=(pix_corners[:,0].max()+pad)
            if ny is None:
                ny=(pix_corners[:,1].max()+pad)
        else:
            nx=(pix_corners[:,0].max()+pad)
            ny=(pix_corners[:,1].max()+pad)
        #print nx,ny
        nx=int(nx)
        ny=int(ny)
        if not(primes is None):
            lens=find_good_fft_lens(2*(nx+ny),primes)
            #print 'nx and ny initially are ',nx,ny
            nx=lens[lens>=nx].min()
            ny=lens[lens>=ny].min()
            #print 'small prime nx and ny are now ',nx,ny
            self.primes=primes[:]
        else:
            self.primes=None
        self.nx=nx
        self.ny=ny
        self.lims=lims
        self.pixsize=pixsize
        self.map=np.zeros([nx,ny])
        self.proj=proj
        self.pad=pad
        self.caches=None
        self.cosdec=cosdec
    def get_caches(self):
        npix=self.nx*self.ny
        nthread=get_nthread()
        self.caches=np.zeros([nthread,npix])
    def clear_caches(self):
        self.map[:]=np.reshape(np.sum(self.caches,axis=0),self.map.shape)
        self.caches=None
    def copy(self):
        newmap=SkyMap(self.lims,self.pixsize,self.proj,self.pad,self.primes,cosdec=self.cosdec,nx=self.nx,ny=self.ny,mywcs=self.wcs)
        newmap.map[:]=self.map[:]
        return newmap
    def clear(self):
        self.map[:]=0
    def axpy(self,map,a):
        self.map[:]=self.map[:]+a*map.map[:]
    def assign(self,arr):
        assert(arr.shape[0]==self.nx)
        assert(arr.shape[1]==self.ny)
        #self.map[:,:]=arr
        self.map[:]=arr
    def get_pix(self,tod):
        ndet=tod.info['dx'].shape[0]
        nsamp=tod.info['dx'].shape[1]
        nn=ndet*nsamp
        coords=np.zeros([nn,2])
        coords[:,0]=np.reshape(tod.info['dx']*180/np.pi,nn)
        coords[:,1]=np.reshape(tod.info['dy']*180/np.pi,nn)
        #print coords.shape
        pix=self.wcs.wcs_world2pix(coords,1)
        #print pix.shape
        xpix=np.reshape(pix[:,0],[ndet,nsamp])-1  #-1 is to go between unit offset in FITS and zero offset in python
        ypix=np.reshape(pix[:,1],[ndet,nsamp])-1  
        xpix=np.round(xpix)
        ypix=np.round(ypix)
        ipix=np.asarray(xpix*self.ny+ypix,dtype='int32')
        return ipix
    def map2tod(self,tod,dat,do_add=True,do_omp=True):
        map2tod(dat,self.map,tod.info['ipix'],do_add,do_omp)

    def tod2map(self,tod,dat,do_add=True,do_omp=True):
        if do_add==False:
            self.clear()
        if not(self.caches is None):
            tod2map_cached(self.caches,dat,tod.info['ipix'])
        else:
            if do_omp:
                tod2map_omp(self.map,dat,tod.info['ipix'])
            else:
                tod2map_simple(self.map,dat,tod.info['ipix'])

    def r_th_maps(self):
        xvec=np.arange(self.nx)
        xvec=xvec-xvec.mean()        
        yvec=np.arange(self.ny)
        yvec=yvec-yvec.mean()
        ymat,xmat=np.meshgrid(yvec,xvec)
        rmat=np.sqrt(xmat**2+ymat**2)
        th=np.arctan2(xmat,ymat)
        return rmat,th
    def dot(self,map):
        tot=np.sum(self.map*map.map)
        return tot
        
    def write(self,fname='map.fits'):
        header=self.wcs.to_header()
        if True: #try a patch to fix the wcs xxx 
            tmp=self.map.transpose().copy()
            hdu=fits.PrimaryHDU(tmp,header=header)
        else:
            hdu=fits.PrimaryHDU(self.map,header=header)
        try:
            hdu.writeto(fname,overwrite=True)
        except:
            hdu.writeto(fname,clobber=True)
    def __mul__(self,map):
        new_map=map.copy()
        new_map.map[:]=self.map[:]*map.map[:]
        return new_map
    def mpi_reduce(self,chunksize=1e5):
        #chunksize is added since at least on my laptop mpi4py barfs if it
        #tries to reduce an nside=512 healpix map, so need to break it into pieces.
        if have_mpi:
            #print("reducing map")
            if chunksize>0:
                nchunk=(1.0*self.nx*self.ny)/chunksize
                nchunk=np.int(np.ceil(nchunk))
            else:
                nchunk=1
            #print('nchunk is ',nchunk)
            if nchunk==1:
                self.map=comm.allreduce(self.map)
            else:
                inds=np.asarray(np.linspace(0,self.nx*self.ny,nchunk+1),dtype='int')
                if len(self.map.shape)>1:
                    tmp=np.zeros(self.map.size)
                    tmp[:]=np.reshape(self.map,len(tmp))
                else:
                    tmp=self.map

                for i in range(len(inds)-1):
                    tmp[inds[i]:inds[i+1]]=comm.allreduce(tmp[inds[i]:inds[i+1]])
                    #self.map[inds[i]:inds[i+1]]=comm.allreduce(self.map[inds[i]:inds[i+1]])
                    #tmp=np.zeros(inds[i+1]-inds[i])
                    #tmp[:]=self.map[inds[i]:inds[i+1]]
                    #tmp=comm.allreduce(tmp)
                    #self.map[inds[i]:inds[i+1]]=tmp
                if len(self.map.shape)>1:
                    self.map[:]=np.reshape(tmp,self.map.shape)
            
            #print("reduced")
    def invert(self):
        mask=np.abs(self.map)>0
        self.map[mask]=1.0/self.map[mask]

class MapNoiseWhite:
    def __init__(self,ivar_map,isinv=True,nfac=1.0):
        self.ivar=read_fits_map(ivar_map)
        if not(isinv):
            mask=self.ivar>0
            self.ivar[mask]=1.0/self.ivar[mask]
        self.ivar=self.ivar*nfac
    def apply_noise(self,map):
        return map*self.ivar
    
class SkyMapCoarse(SkyMap):
    def __init__(self,map):
        self.nx=map.shape[0]
        try:
            self.ny=map.shape[1]
        except:
            self.ny=1
        self.map=map.copy()
    def get_caches(self):
        return
    def clear_caches(self):
        return
    def copy(self):
        cp=copy.copy(self)
        cp.map=self.map.copy()
        return cp
    def get_pix(self):
        return
    def map2tod(self,*args,**kwargs):
        return
    def tod2map(self,*args,**kwargs):
        return
class SkyMapTwoRes:
    """A pair of maps to serve as a prior for multi-experiment mapping.  This would e.g. be the ACT map that e.g. Mustang should agree
    with on large scales."""
    def __init__(self,map_lowres,lims,osamp=1,smooth_fac=0.0):
        small_wcs,lims_use,map_corner=get_aligned_map_subregion_car(lims,map_lowres,osamp=osamp)
        self.small_lims=lims_use
        self.small_wcs=small_wcs
        self.map=read_fits_map(map_lowres)
        self.osamp=osamp
        self.map_corner=map_corner
        self.beamft=None
        self.mask=None
        self.map_deconvolved=None
        self.noise=None
        self.fine_prior=None
        self.nx_coarse=None
        self.ny_coarse=None
        self.grid_facs=None
        self.isglobal_prior=True
        self.smooth_fac=smooth_fac
    def copy(self):
        return copy.copy(self)
    def get_map_deconvolved(self,map_deconvolved):
        self.map_deconvolved=read_fits_map(map_deconvolved)
    def set_beam_gauss(self,fwhm_pix):
        tmp=0*self.map
        xvec=get_ft_vec(tmp.shape[0])
        yvec=get_ft_vec(tmp.shape[1])
        xx,yy=np.meshgrid(yvec,xvec)
        rsqr=xx**2+yy**2
        sig_pix=fwhm_pix/np.sqrt(8*np.log(2))
        beam=np.exp(-0.5*rsqr/(sig_pix**2))
        beam=beam/np.sum(beam)
        self.beamft=np.fft.rfft2(beam)
    def set_beam_1d(self,prof,pixsize):
        tmp=0*self.map
        xvec=get_ft_vec(tmp.shape[0])
        yvec=get_ft_vec(tmp.shape[1])
        xx,yy=np.meshgrid(yvec,xvec)
        rsqr=xx**2+yy**2
        rr=np.sqrt(rsqr)*pixsize
        beam=np.interp(rr,prof[:,0],prof[:,1])
        beam=beam/np.sum(beam)
        self.beamft=np.fft.rfft2(beam)


    def set_noise_white(self,ivar_map,isinv=True,nfac=1.0):
        self.noise=MapNoiseWhite(ivar_map,isinv,nfac)
    def maps2fine(self,fine,coarse):
        out=fine.copy()
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                out[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)]=coarse[i+self.map_corner[0],j+self.map_corner[1]]
        out[self.mask]=fine[self.mask]
        return out
    def maps2coarse(self,fine,coarse):
        out=coarse.copy()
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                out[i+self.map_corner[0],j+self.map_corner[1]]=(1-self.grid_facs[i,j])*coarse[i+self.map_corner[0],j+self.map_corner[1]]+np.sum(fine[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)])/self.osamp**2
        return out
    def coarse2maps(self,inmap):
        coarse=1.0*inmap
        fine=np.zeros([self.nx_coarse*self.osamp,self.ny_coarse*self.osamp])
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                coarse[i+self.map_corner[0],j+self.map_corner[1]]=(1-self.grid_facs[i,j])*inmap[i+self.map_corner[0],j+self.map_corner[1]]
                fine[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)]=inmap[i+self.map_corner[0],j+self.map_corner[1]]/self.osamp**2
        fine=fine*self.mask
        return coarse,fine
    def set_mask(self,hits,thresh=0):
        self.mask=hits>thresh
        self.fine_prior=0*hits
        self.nx_coarse=np.int(np.round(hits.shape[0]/self.osamp))
        self.ny_coarse=np.int(np.round(hits.shape[1]/self.osamp))
        self.grid_facs=np.zeros([self.nx_coarse,self.ny_coarse])
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                self.grid_facs[i,j]=np.mean(self.mask[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)])
                self.fine_prior[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)]=self.map_deconvolved[self.map_corner[0]+i,self.map_corner[1]+j]
    def apply_Qinv(self,map):
        tmp=self.fine_prior.copy()
        tmp[self.mask]=map[self.mask]
        tmp2=0*self.map_deconvolved.copy()
        for i in range(self.nx_coarse):
            for j in range(self.nx_coarse):
                tmp2[self.map_corner[0]+i,self.map_corner[1]+j]=np.mean(tmp[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)])
        tmp2_conv=np.fft.irfft2(np.fft.rfft2(tmp2)*self.beamft)
        tmp2_conv_filt=self.noise.apply_noise(tmp2_conv)
        tmp2_reconv=np.fft.irfft2(np.fft.rfft2(tmp2_conv_filt)*self.beamft)
        #tmp2_reconv=np.fft.irfft2(np.fft.rfft2(tmp2_conv)*self.beamft)
        #tmp2_reconv=tmp2.copy()
        fac=1.0/self.osamp**2
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                tmp[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)]=fac*tmp2_reconv[i+self.map_corner[0],j+self.map_corner[1]]
        ans=0.0*tmp
        ans[self.mask]=tmp[self.mask]
        return ans
    def apply_H(self,coarse,fine):
        mm=self.maps2coarse(coarse,fine)
        mm=self.beam_convolve(mm)
        return mm
    def apply_HT(self,mm):
        mm=self.beam_convolve(mm)
        coarse,fine=self.coarse2maps(mm)
        return coarse,fine
    def get_rhs(self,mapset):
        #if map is None:
        #    map=self.map
        #map_filt=self.noise.apply_noise(map)
        #map_filt_conv=np.fft.irfft2(np.fft.rfft2(map_filt)*self.beamft)
        #tmp=0.0*self.mask
        #fac=1.0/self.osamp**2
        #for i in range(self.nx_coarse):
        #    for j in range(self.ny_coarse):
        #        tmp[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)]=fac*map_filt_conv[i+self.map_corner[0],j+self.map_corner[1]]

        #ans=0*tmp
        #ans[self.mask]=tmp[self.mask]
        #return ans
        

        coarse_ind=None
        fine_ind=None
        for i in range(mapset.nmap):
            if isinstance(mapset.maps[i],SkyMapCoarse):
                coarse_ind=i
            else:
                if isinstance(mapset.maps[i],SkyMap):
                    fine_ind=i
        if (coarse_ind is None)|(fine_ind is None):
            print("Errror in twolevel prior:  either fine or coarse skymap not found.")
            return

        
        mm=self.noise.apply_noise(self.map)
        if True:
            coarse,fine=self.apply_HT(mm)
            mapset.maps[coarse_ind].map[:]=mapset.maps[coarse_ind].map[:]+coarse
            mapset.maps[fine_ind].map[:]=mapset.maps[fine_ind].map[:]+fine
        else:
            mm=self.beam_convolve(mm)
            coarse,fine=self.coarse2maps(mm)
            i1=self.map_corner[0]
            i2=i1+self.nx_coarse
            j1=self.map_corner[1]
            j2=j1+self.ny_coarse
            coarse[i1:i2,j1:j2]=coarse[i1:i2,j1:j2]*(1-self.grid_facs)
            mapset.maps[coarse_ind].map[:]=mapset.maps[coarse_ind].map[:]+coarse
            mapset.maps[fine_ind].map[self.mask]=mapset.maps[fine_ind].map[self.mask]+fine[self.mask]/self.osamp**2

    def beam_convolve(self,map):
        mapft=np.fft.rfft2(map)
        mapft=mapft*self.beamft
        return np.fft.irfft2(mapft)
    def apply_prior(self,mapset,outmapset):
        coarse_ind=None
        fine_ind=None
        for i in range(mapset.nmap):
            if isinstance(mapset.maps[i],SkyMapCoarse):
                coarse_ind=i
            else:
                if isinstance(mapset.maps[i],SkyMap):
                    fine_ind=i
        if (coarse_ind is None)|(fine_ind is None):
            print("Errror in twolevel prior:  either fine or coarse skymap not found.")
            return
        if True:
            mm=self.apply_H(mapset.maps[fine_ind].map,mapset.maps[coarse_ind].map)
            mm_filt=self.noise.apply_noise(mm)
            coarse,fine=self.apply_HT(mm_filt)
            
        else:
            summed=self.maps2coarse(mapset.maps[fine_ind].map,mapset.maps[coarse_ind].map)
            summed=self.beam_convolve(summed)
            summed=self.noise.apply_noise(summed)
            summed=self.beam_convolve(summed)
            coarse,fine=self.coarse2maps(summed)

        outmapset.maps[fine_ind].map[self.mask]=outmapset.maps[fine_ind].map[self.mask]+fine[self.mask]
        outmapset.maps[coarse_ind].map[:]=outmapset.maps[coarse_ind].map[:]+coarse

        if self.smooth_fac>0:
            summed=self.maps2coarse(mapset.maps[fine_ind].map,mapset.maps[coarse_ind].map)
            summed_smooth=self.beam_convolve(summed)
            delt=summed-summed_smooth
            delt_filt=self.noise.apply_noise(delt)*self.smooth_fac
            delt_filt=delt_filt-self.beam_convolve(delt_filt)
            coarse,fine=self.coarse2maps(delt_filt)
            outmapset.maps[fine_ind].map[self.mask]=outmapset.maps[fine_ind].map[self.mask]+fine[self.mask]
            outmapset.maps[coarse_ind].map[:]=outmapset.maps[coarse_ind].map[:]+coarse
            



    def __bust_apply_prior(self,map,outmap):
        outmap.map[:]=outmap.map[:]+self.apply_Qinv(map.map)
              


def poltag2pols(poltag):
    if poltag=='I':
        return ['I']
    if poltag=='IQU':
        return ['I','Q','U']
    if poltag=='QU':
        return ['Q','U']
    if poltag=='IQU_PRECON':
        return ['I','Q','U','QQ','QU','UU']
    if poltag=='QU_PRECON':
        return ['QQ','UU','QU']

    return None
    
class PolMap:
    def __init__(self,lims,pixsize,poltag='I',proj='CAR',pad=2,primes=None,cosdec=None,nx=None,ny=None,mywcs=None,ref_equ=False):
        pols=poltag2pols(poltag)
        if pols is None:
            print('Unrecognized polarization state ' + poltag + ' in PolMap.__init__')
            return
        npol=len(pols)
        if mywcs is None:
            self.wcs=get_wcs(lims,pixsize,proj,cosdec,ref_equ)
        else:
            self.wcs=mywcs
        corners=np.zeros([4,2])
        corners[0,:]=[lims[0],lims[2]]
        corners[1,:]=[lims[0],lims[3]]
        corners[2,:]=[lims[1],lims[2]]
        corners[3,:]=[lims[1],lims[3]]
        pix_corners=self.wcs.wcs_world2pix(corners*180/np.pi,1)        
        pix_corners=np.round(pix_corners)
        #print pix_corners
        #print type(pix_corners)
        #if pix_corners.min()<0.5:
        if pix_corners.min()<-0.5:
            print('corners seem to have gone negative in SkyMap projection.  not good, you may want to check this.')
        if True: #try a patch to fix the wcs xxx
            if nx is None:
                nx=(pix_corners[:,0].max()+pad)
            if ny is None:
                ny=(pix_corners[:,1].max()+pad)
        else:
            nx=(pix_corners[:,0].max()+pad)
            ny=(pix_corners[:,1].max()+pad)
        #print nx,ny
        nx=int(nx)
        ny=int(ny)
        if not(primes is None):
            lens=find_good_fft_lens(2*(nx+ny),primes)
            #print 'nx and ny initially are ',nx,ny
            nx=lens[lens>=nx].min()
            ny=lens[lens>=ny].min()
            #print 'small prime nx and ny are now ',nx,ny
            self.primes=primes[:]
        else:
            self.primes=None
        self.nx=nx
        self.ny=ny
        self.npol=npol
        self.poltag=poltag
        self.pols=pols
        self.lims=lims
        self.pixsize=pixsize
        if npol>1:
            self.map=np.zeros([nx,ny,npol])
        else:
            self.map=np.zeros([nx,ny])
        self.proj=proj
        self.pad=pad
        self.caches=None
        self.cosdec=cosdec
    def get_caches(self):
        npix=self.nx*self.ny*self.npol
        nthread=get_nthread()
        self.caches=np.zeros([nthread,npix])
    def clear_caches(self):
        self.map[:]=np.reshape(np.sum(self.caches,axis=0),self.map.shape)
        self.caches=None
    def copy(self):
        newmap=PolMap(self.lims,self.pixsize,self.poltag,self.proj,self.pad,self.primes,cosdec=self.cosdec,nx=self.nx,ny=self.ny,mywcs=self.wcs)
        newmap.map[:]=self.map[:]
        return newmap
    def clear(self):
        self.map[:]=0
    def axpy(self,map,a):
        self.map[:]=self.map[:]+a*map.map[:]
    def assign(self,arr):
        assert(arr.shape[0]==self.nx)
        assert(arr.shape[1]==self.ny)
        if self.npol>1:
            assert(arr.shape[2]==self.npol)
        #self.map[:,:]=arr
        self.map[:]=arr
    def set_polstate(self,poltag):
        pols=poltag2pols(poltag)
        if pols is None:
            print('Unrecognized polarization state ' + poltag + ' in PolMap.set_polstate.')
            return
        npol=len(pols)
        self.npol=npol
        self.poltag=poltag
        self.pols=pols
        if npol>1:
            self.map=np.zeros([self.nx,self.ny,npol])
        else:
            self.map=np.zeros([self.nx,self.ny])
    def invert(self,thresh=1e-6):
        #We can use np.linalg.pinv to reasonably efficiently invert a bunch of tiny matrices with an
        #eigenvalue cut.  It's more efficient to do this in C, but it only has to be done once per run
        if self.npol>1: 
            if self.poltag=='QU_PRECON':
                tmp=np.zeros([self.nx*self.ny,2,2])
                tmp[:,0,0]=np.ravel(self.map[:,:,0])
                tmp[:,1,1]=np.ravel(self.map[:,:,1])
                tmp[:,0,1]=np.ravel(self.map[:,:,2])
                tmp[:,1,0]=np.ravel(self.map[:,:,2])
                tmp=np.linalg.pinv(tmp,thresh)
                self.map[:,:,0]=np.reshape(tmp[:,0,0],[self.map.shape[0],self.map.shape[1]])
                self.map[:,:,1]=np.reshape(tmp[:,1,1],[self.map.shape[0],self.map.shape[1]])
                self.map[:,:,2]=np.reshape(tmp[:,0,1],[self.map.shape[0],self.map.shape[1]])
            if self.poltag=='IQU_PRECON':
                #the mapping may seem a bit abstruse here.  The preconditioner matrix has entries
                #  [I   Q   U ]
                #  [Q  QQ  QU ]
                #  [U  QU  UU ]
                #so the unpacking needs to match the ordering in the C file before we can use pinv
                n=self.nx*self.ny
                nx=self.nx
                ny=self.ny
                tmp=np.zeros([self.nx*self.ny,3,3])
                tmp[:,0,0]=np.reshape(self.map[:,:,0],n)
                tmp[:,0,1]=np.reshape(self.map[:,:,1],n)
                tmp[:,1,0]=tmp[:,0,1]
                tmp[:,0,2]=np.reshape(self.map[:,:,2],n)
                tmp[:,2,0]=tmp[:,0,2]
                tmp[:,1,1]=np.reshape(self.map[:,:,3],n)
                tmp[:,1,2]=np.reshape(self.map[:,:,4],n)
                tmp[:,2,1]=tmp[:,1,2]
                tmp[:,2,2]=np.reshape(self.map[:,:,5],n)
                alldets=np.linalg.det(tmp)
                isbad=alldets<thresh*alldets.max()
                ispos=tmp[:,0,0]>0
                inds=isbad&ispos
                vec=tmp[inds,0,0]
                print('determinant range is ' + repr(alldets.max())+ '  ' + repr(alldets.min()))
                tmp=np.linalg.pinv(tmp,thresh)
                if True:
                    print('Warning!  zeroing out bits like this is super janky.  Be warned...')
                    tmp[isbad,:,:]=0
                    inds=isbad&ispos
                    tmp[inds,0,0]=1.0/vec
                alldets=np.linalg.det(tmp)                
                print('determinant range is now ' + repr(alldets.max())+ '  ' + repr(alldets.min()))

                self.map[:,:,0]=np.reshape(tmp[:,0,0],[nx,ny])
                self.map[:,:,1]=np.reshape(tmp[:,0,1],[nx,ny])
                self.map[:,:,2]=np.reshape(tmp[:,0,2],[nx,ny])
                self.map[:,:,3]=np.reshape(tmp[:,1,1],[nx,ny])
                self.map[:,:,4]=np.reshape(tmp[:,1,2],[nx,ny])
                self.map[:,:,5]=np.reshape(tmp[:,2,2],[nx,ny])
                
        else:
            mask=self.map!=0
            self.map[mask]=1.0/self.map[mask]
    def get_pix(self,tod):
        ndet=tod.info['dx'].shape[0]
        nsamp=tod.info['dx'].shape[1]
        nn=ndet*nsamp
        coords=np.zeros([nn,2])
        coords[:,0]=np.reshape(tod.info['dx']*180/np.pi,nn)
        coords[:,1]=np.reshape(tod.info['dy']*180/np.pi,nn)
        #print coords.shape
        pix=self.wcs.wcs_world2pix(coords,1)
        #print pix.shape
        xpix=np.reshape(pix[:,0],[ndet,nsamp])-1  #-1 is to go between unit offset in FITS and zero offset in python
        ypix=np.reshape(pix[:,1],[ndet,nsamp])-1  
        xpix=np.round(xpix)
        ypix=np.round(ypix)
        ipix=np.asarray(xpix*self.ny+ypix,dtype='int32')
        return ipix
    def map2tod(self,tod,dat,do_add=True,do_omp=True):
        if self.npol>1:
            polmap2tod(dat,self.map,self.poltag,tod.info['twogamma_saved'],tod.info['ipix'],do_add,do_omp)
        else:
            map2tod(dat,self.map,tod.info['ipix'],do_add,do_omp)

        
    def tod2map(self,tod,dat,do_add=True,do_omp=True):
        if do_add==False:
            self.clear()
        if self.npol>1:
            tod2polmap(self.map,dat,self.poltag,tod.info['twogamma_saved'],tod.info['ipix'])
            return
        #print("working on nonpolarized bit")

        if not(self.caches is None):
            tod2map_cached(self.caches,dat,tod.info['ipix'])
        else:
            if do_omp:
                tod2map_omp(self.map,dat,tod.info['ipix'])
            else:
                tod2map_simple(self.map,dat,tod.info['ipix'])

    def r_th_maps(self):
        xvec=np.arange(self.nx)
        xvec=xvec-xvec.mean()        
        yvec=np.arange(self.ny)
        yvec=yvec-yvec.mean()
        ymat,xmat=np.meshgrid(yvec,xvec)
        rmat=np.sqrt(xmat**2+ymat**2)
        th=np.arctan2(xmat,ymat)
        return rmat,th
    def dot(self,map):
        tot=np.sum(self.map*map.map)
        return tot
        
    def write(self,fname='map.fits'):
        header=self.wcs.to_header()

        if self.npol>1:
            ind=fname.rfind('.')
            if ind>0:
                if fname[ind+1:]=='fits':
                    head=fname[:ind]
                    tail=fname[ind:]
                else:
                    head=fname
                    tail='.fits'
            else:
                head=fname
                tail='.fits'
            tmp=np.zeros([self.ny,self.nx])
            for i in range(self.npol):
                tmp[:]=np.squeeze(self.map[:,:,i]).T
                hdu=fits.PrimaryHDU(tmp,header=header)
                try:
                    hdu.writeto(head+'_'+self.pols[i]+tail,overwrite=True)
                except:
                    hdu.writeto(head+'_'+self.pols[i]+tail,clobber=True)
            return

        if True: #try a patch to fix the wcs xxx 
            tmp=self.map.transpose().copy()
            hdu=fits.PrimaryHDU(tmp,header=header)
        else:
            hdu=fits.PrimaryHDU(self.map,header=header)
        try:
            hdu.writeto(fname,overwrite=True)
        except:
            hdu.writeto(fname,clobber=True)
    def __mul__(self,map):
        if self.npol==1:
            new_map=self.copy()
            new_map.map[:]=self.map[:]*map.map[:]
            return new_map
        else:
            assert(map.poltag+'_PRECON'==self.poltag)
            new_map=map.copy()
            if self.poltag=='QU_PRECON':
                new_map.map[:,:,0]=self.map[:,:,0]*map.map[:,:,0]+self.map[:,:,2]*map.map[:,:,1]
                new_map.map[:,:,1]=self.map[:,:,2]*map.map[:,:,0]+self.map[:,:,1]*map.map[:,:,1]
                return new_map
            if self.poltag=='IQU_PRECON':
                #the indices are set such that the preconditioner matrix [I Q U; Q QQ QU; U QU UU] match the C code.  
                #once we've inverted, the output should be the product of that matrix times [I Q U]
                new_map.map[:,:,0]=self.map[:,:,0]*map.map[:,:,0]+self.map[:,:,1]*map.map[:,:,1]+self.map[:,:,2]*map.map[:,:,2]
                new_map.map[:,:,1]=self.map[:,:,1]*map.map[:,:,0]+self.map[:,:,3]*map.map[:,:,1]+self.map[:,:,4]*map.map[:,:,2]
                new_map.map[:,:,2]=self.map[:,:,2]*map.map[:,:,0]+self.map[:,:,4]*map.map[:,:,1]+self.map[:,:,5]*map.map[:,:,2]
                return new_map

            print('unrecognized tag in PolMap.__mul__:  ' + repr(self.poltag))
            assert(1==0)
    def mpi_reduce(self,chunksize=1e5):
        #chunksize is added since at least on my laptop mpi4py barfs if it
        #tries to reduce an nside=512 healpix map, so need to break it into pieces.
        if have_mpi:
            #print("reducing map")
            if chunksize>0:
                nchunk=(1.0*self.nx*self.ny*self.npol)/chunksize
                nchunk=np.int(np.ceil(nchunk))
            else:
                nchunk=1
            #print('nchunk is ',nchunk)
            if nchunk==1:
                self.map=comm.allreduce(self.map)
            else:
                inds=np.asarray(np.linspace(0,self.nx*self.ny*self.npol,nchunk+1),dtype='int')
                if len(self.map.shape)>1:
                    tmp=np.zeros(self.map.size)
                    tmp[:]=np.reshape(self.map,len(tmp))
                else:
                    tmp=self.map

                for i in range(len(inds)-1):
                    tmp[inds[i]:inds[i+1]]=comm.allreduce(tmp[inds[i]:inds[i+1]])
                    #self.map[inds[i]:inds[i+1]]=comm.allreduce(self.map[inds[i]:inds[i+1]])
                    #tmp=np.zeros(inds[i+1]-inds[i])
                    #tmp[:]=self.map[inds[i]:inds[i+1]]
                    #tmp=comm.allreduce(tmp)
                    #self.map[inds[i]:inds[i+1]]=tmp
                if len(self.map.shape)>1:
                    self.map[:]=np.reshape(tmp,self.map.shape)
            
            #print("reduced")
class HealMap(SkyMap):
    def __init__(self,proj='RING',nside=512):
        if not(have_healpy):
            printf("Healpix map requested, but healpy not found.")
            return
        self.proj=proj
        self.nside=nside
        self.nx=healpy.nside2npix(self.nside)
        self.ny=1
        self.caches=None
        self.map=np.zeros([self.nx,self.ny])
    def copy(self):
        newmap=HealMap(self.proj,self.nside)
        newmap.map[:]=self.map[:]
        return newmap
    def get_pix(self,tod):
        ipix=healpy.ang2pix(self.nside,np.pi/2-tod.info['dy'],tod.info['dx'],self.proj=='NEST')
        return ipix
    def write(self,fname='map.fits',overwrite=True):
        if self.map.shape[1]<=1:
            healpy.write_map(fname,self.map[:,0],nest=(self.proj=='NEST'),overwrite=overwrite)        
    

class HealPolMap(PolMap):
    def __init__(self,poltag='I',proj='RING',nside=512):
        if not(have_healpy):
            printf("Healpix map requested, but healpy not found.")
            return
        pols=poltag2pols(poltag)
        if pols is None:
            print('Unrecognized polarization state ' + poltag + ' in PolMap.__init__')
            return
        npol=len(pols)
        self.proj=proj
        self.nside=nside
        self.nx=healpy.nside2npix(self.nside)
        self.ny=1
        self.npol=npol
        self.poltag=poltag
        self.pols=pols
        self.caches=None
        if self.npol>1:
            self.map=np.zeros([self.nx,self.ny,self.npol])
        else:
            self.map=np.zeros([self.nx,self.ny])
    def copy(self):
        newmap=HealPolMap(self.poltag,self.proj,self.nside)
        newmap.map[:]=self.map[:]
        return newmap
    def get_pix(self,tod):
        ipix=healpy.ang2pix(self.nside,np.pi/2-tod.info['dy'],tod.info['dx'],self.proj=='NEST')
        return ipix
    def write(self,fname='map.fits',overwrite=True):
        if self.map.shape[1]<=1:
            if self.npol==1:
                healpy.write_map(fname,self.map[:,0],nest=(self.proj=='NEST'),overwrite=overwrite)        
            else:
                    ind=fname.rfind('.')
                    if ind>0:
                        if fname[ind+1:]=='fits':
                            head=fname[:ind]
                            tail=fname[ind:]
                        else:
                            head=fname
                            tail='.fits'
                    else:
                        head=fname
                        tail='.fits'
                    #tmp=np.zeros([self.ny,self.nx])
                    tmp=np.zeros(self.nx)
                    for i in range(self.npol):
                        tmp[:]=np.squeeze(self.map[:,:,i]).T
                        #print('tmp shape is ',tmp.shape)
                        fname=head+'_'+self.pols[i]+tail
                        healpy.write_map(fname,tmp,nest=(self.proj=='NEST'),overwrite=overwrite)
                        #healpy.write_map(fname,tmp[:,0],nest=(self.proj=='NEST'),overwrite=overwrite)
    

class Cuts:
    def __init__(self,tod,do_add=True):
        #if class(tod)==Cuts: #for use in copy
        if isinstance(tod,Cuts):
            self.map=tod.map.copy()
            self.bad_inds=tod.bad_inds.copy()
            self.namps=tod.nsamp
            self.do_add=tod.do_add
            return
        bad_inds=np.where(tod.info['bad_samples'])
        #dims=tod.info['dat_calib'].shape
        dims=tod.get_data_dims()
        bad_inds=np.ravel_multi_index(bad_inds,dims)
        self.nsamp=len(bad_inds)
        self.inds=bad_inds
        self.map=np.zeros(self.nsamp)
        self.do_add=do_add
    def clear(self):
        self.map[:]=0
    def axpy(self,cuts,a):
        self.map[:]=self.map[:]+a*cuts.map[:]
    def map2tod(self,tod,dat):
        dd=np.ravel(dat)
        if self.do_add:
            dd[self.inds]=self.map
        else:
            dd[self.inds]+=self.map
    def tod2map(self,tod,dat):
        dd=np.ravel(dat)
        self.map[:]=dd[self.inds]
    def dot(self,cuts):
        tot=np.dot(self.map,cuts.map)
        return tot
    def copy(self):
        return Cuts(self)
class CutsCompact:
    def __init__(self,tod):
        if isinstance(tod,CutsCompact):
            self.ndet=tod.ndet
            self.nseg=tod.nseg
            self.istart=tod.istart
            self.istop=tod.istop
        else:
            #ndet=tod.info['dat_calib'].shape[0]
            ndet=tod.get_ndet()
            self.ndet=ndet
            self.nseg=np.zeros(ndet,dtype='int')
            self.istart=[None]*ndet
            self.istop=[None]*ndet
            #self.imax=tod.info['dat_calib'].shape[1]
            self.imax=tod.get_ndata()

        self.imap=None
        self.map=None
        
    def copy(self,deep=True):
        copy=CutsCompact(self)
        if deep:
            if not(self.imap is None):
                copy.imap=self.imap.copy()
            if not(self.map is None):
                copy.map=self.map.copy()
        else:
            copy.imap=self.imap
            copy.map=self.map
        return copy
    def add_cut(self,det,istart,istop):
        if istart>=self.imax:
            #this is asking to add a cut past the end of the data.
            return
        if istop>self.imax: #don't have a cut run past the end of the timestream
            istop=self.imax
            
        self.nseg[det]=self.nseg[det]+1
        if self.istart[det] is None:
            self.istart[det]=[istart]
        else:
            self.istart[det].append(istart)
        if self.istop[det] is None:
            self.istop[det]=[istop]
        else:
            self.istop[det].append(istop)
    def get_imap(self):
        ncut=0
        for det in range(self.ndet):
            for i in range(self.nseg[det]):
                ncut=ncut+(self.istop[det][i]-self.istart[det][i])
        print('ncut is ' + repr(ncut))
        self.imap=np.zeros(ncut,dtype='int64')
        icur=0
        for det in range(self.ndet):
            for i in range(self.nseg[det]):
                istart=det*self.imax+self.istart[det][i]
                istop=det*self.imax+self.istop[det][i]
                nn=istop-istart
                self.imap[icur:icur+nn]=np.arange(istart,istop)
                icur=icur+nn
        self.map=np.zeros(len(self.imap))
    def cuts_from_array(self,cutmat):
        for det in range(cutmat.shape[0]):
            nseg,istart,istop=segs_from_vec(cutmat[det,:])
            self.nseg[det]=nseg
            self.istart[det]=istart
            self.istop[det]=istop
    def merge_cuts(self):
        tmp=np.ones(self.imax+2,dtype='bool')
        for det in range(self.ndet):
            if self.nseg[det]>1:  #if we only have one segment, don't have to worry about strange overlaps
                tmp[:]=True
                for i in range(self.nseg[det]):
                    tmp[(self.istart[det][i]+1):(self.istop[det][i]+1)]=False
                nseg,istart,istop=segs_from_vec(tmp,pad=False)
                self.nseg[det]=nseg
                self.istart[det]=istart
                self.istop[det]=istop
    
    def tod2map(self,tod,mat=None,do_add=True,do_omp=False):
        if mat is None:
            #mat=tod.info['dat_calib']
            mat=tod.get_data()
        tod2cuts_c(self.map.ctypes.data,mat.ctypes.data,self.imap.ctypes.data,len(self.imap),do_add)

    def map2tod(self,tod,mat=None,do_add=True,do_omp=False):
        if mat is None:
            #mat=tod.info['dat_calib']
            mat=tod.get_data()
        #print('first element is ' + repr(mat[0,self.imap[0]]))
        cuts2tod_c(mat.ctypes.data,self.map.ctypes.data,self.imap.ctypes.data,len(self.imap),do_add)
        #print('first element is now ' + repr(mat[0,self.imap[0]]))
        #return mat
    def clear(self):
        if not(self.map is None):
            self.map[:]=0
    def dot(self,other=None):
        if self.map is None:
            return None
        if other is None:
            return np.dot(self.map,self.map)
        else:
            if other.map is None:
                return None
            return np.dot(self.map,other.map)
    def axpy(self,common,a):
        self.map=self.map+a*common.map
    def write(self,fname=None):
        pass
    def apply_prior(self,x,Ax):
        Ax.map=Ax.map+self.map*x.map
    def __mul__(self,to_mul):
        tt=self.copy()
        tt.map=self.map*to_mul.map
        return tt

#this class is pointless, as you can get the same functionality with the tsModel class, which will be 
#consistent with other timestream model classes.  
#class CutsVecs:
#    def __init__(self,todvec,do_add=True):
#        #if class(todvec)==CutsVecs: #for use in copy
#        if isinstance(todvec,CutsVecs):
#            self.cuts=[None]*todvec.ntod
#            self.ntod=todvec.ntod
#            for i in range(todvec.ntod):
#                self.cuts[i]=todvec.cuts[i].copy()
#            return
#        #if class(todvec)!=TodVec:
#        if not(isinstance(todvec,TodVec)):
#            print('error in CutsVecs init, must pass in a todvec class.')
#            return None
#        self.cuts=[None]*todvec.ntod
#        self.ntod=todvec.ntod
#        for i in range(todvec.ntod):
#            tod=todvec.tods[i]
#            if tod.info['tag']!=i:
#                print('warning, tag mismatch in CutsVecs.__init__')
#                print('continuing, but you should be careful...')            
#            if 'bad_samples' in tod.info:
#                self.cuts[i]=Cuts(tod,do_add)
#            elif 'mask' in tod.info:
#                self.cuts[i]=CutsCompact(tod)
#                self.cuts[i].cuts_from_array(tod.info['mask'])
#                self.cuts[i].get_imap()
#    def copy(self):
#        return CutsVecs(self)
#    def clear(self):
#        for cuts in self.cuts:
#            cuts.clear()
#    def axpy(self,cutsvec,a):
#        assert(self.ntod==cutsvec.ntod)
#        for i in range(ntod):
#            self.cuts[i].axpy(cutsvec.cuts[i],a)
#    def map2tod(self,todvec):
#        assert(self.ntod==todvec.ntod)
#        for i in range(self.ntod):
#            self.cuts[i].map2tod(todvec.tods[i])
#    def tod2map(self,todvec,dat):
#        assert(self.ntod==todvec.ntod)
#        assert(self.ntod==dat.ntod)
#        for i in range(self.ntod):
#            self.cuts[i].tod2map(todvec.tods[i],dat.tods[i])
#    def dot(self,cutsvec):
#        tot=0.0
#        assert(self.ntod==cutsvec.ntod)
#        for i in range(self.ntod):
#            tot+=self.cuts[i].dot(cutsvec.cuts[i])
#        return tot
        
                                 
            
class SkyMapCar:
    def __init__(self,lims,pixsize):
        try:
            self.lims=lims.copy()
        except:
            self.lims=lims[:]
        self.pixsize=pixsize
        self.cosdec=np.cos(0.5*(lims[2]+lims[3]))
        nx=np.int(np.ceil((lims[1]-lims[0])/pixsize*self.cosdec))
        ny=np.int(np.ceil((lims[3]-lims[2])/pixsize))
        self.nx=nx
        self.ny=ny
        self.npix=nx*ny
        self.map=np.zeros([nx,ny])
    def copy(self):
        mycopy=SkyMapCar(self.lims,self.pixsize)
        mycopy.map[:]=self.map[:]
        return mycopy
    def clear(self):
        self.map[:,:]=0

    def axpy(self,map,a):
        self.map[:]=self.map[:]+a*map.map[:]
        
    def assign(self,arr):
        assert(arr.shape[0]==self.nx)
        assert(arr.shape[1]==self.ny)
        self.map[:,:]=arr
    def get_pix(self,tod):
        xpix=np.round((tod.info['dx']-self.lims[0])*self.cosdec/self.pixsize)
        ypix=np.round((tod.info['dy']-self.lims[2])/self.pixsize)
        #ipix=np.asarray(ypix*self.nx+xpix,dtype='int32')
        ipix=np.asarray(xpix*self.ny+ypix,dtype='int32')
        return ipix
    def map2tod(self,tod,dat,do_add=True,do_omp=True):
        map2tod(dat,self.map,tod.info['ipix'],do_add,do_omp)

    def tod2map(self,tod,dat,do_add=True,do_omp=True):
        if do_add==False:
            self.clear()
        if do_omp:
            tod2map_omp(self.map,dat,tod.info['ipix'])
        else:
            tod2map_simple(self.map,dat,tod.info['ipix'])

    def r_th_maps(self):
        xvec=np.arange(self.nx)
        xvec=xvec-xvec.mean()        
        yvec=np.arange(self.ny)
        yvec=yvec-yvec.mean()
        ymat,xmat=np.meshgrid(yvec,xvec)
        rmat=np.sqrt(xmat**2+ymat**2)
        th=np.arctan2(xmat,ymat)
        return rmat,th
    def dot(self,map):
        tot=np.sum(self.map*map.map)
        return tot
def find_bad_skew_kurt(dat,skew_thresh=6.0,kurt_thresh=5.0):
    ndet=dat.shape[0]
    isgood=np.ones(ndet,dtype='bool')
    skew=np.mean(dat**3,axis=1)
    mystd=np.std(dat,axis=1)
    skew=skew/mystd**1.5
    mykurt=np.mean(dat**4,axis=1)
    kurt=mykurt/mystd**4-3
    
    isgood[np.abs(skew)>skew_thresh*np.median(np.abs(skew))]=False
    isgood[np.abs(kurt)>kurt_thresh*np.median(np.abs(kurt))]=False
    


    return skew,kurt,isgood

def timestreams_from_gauss(ra,dec,fwhm,tod,pred=None):
    if pred is None:
        #pred=np.zeros(tod.info['dat_calib'].shape)
        pred=tod.get_empty(True)
    #n=tod.info['dat_calib'].size
    n=np.product(tod.get_data_dims())
    assert(pred.size==n)
    npar_src=4 #x,y,sig,amp
    dx=tod.info['dx']
    dy=tod.info['dy']
    pp=np.zeros(npar_src)
    pp[0]=ra
    pp[1]=dec
    pp[2]=fwhm/np.sqrt(8*np.log(2))*np.pi/180/3600 
    pp[3]=1
    fill_gauss_src_c(pp.ctypes.data,dx.ctypes.data,dy.ctypes.data,pred.ctypes.data,n)    
    return pred

def timestreams_from_isobeta_c(params,tod,pred=None):
    if pred is None:
        #pred=np.zeros(tod.info['dat_calib'].shape)
        pred=tod.get_empty(True)
    #n=tod.info['dat_calib'].size
    n=np.product(tod.get_data_dims())
    assert(pred.size==n)
    dx=tod.info['dx']
    dy=tod.info['dy']
    fill_isobeta_c(params.ctypes.data,dx.ctypes.data,dy.ctypes.data,pred.ctypes.data,n)

    npar_beta=5 #x,y,theta,beta,amp
    npar_src=4 #x,y,sig,amp
    nsrc=(params.size-npar_beta)//npar_src
    for i in range(nsrc):
        pp=np.zeros(npar_src)
        ioff=i*npar_src+npar_beta
        pp[:]=params[ioff:(ioff+npar_src)]
        fill_gauss_src_c(pp.ctypes.data,dx.ctypes.data,dy.ctypes.data,pred.ctypes.data,n)


    return pred

def derivs_from_elliptical_isobeta(params,tod,*args,**kwargs):
    npar=len(params)
    assert(npar==7)
    pred=tod.get_empty()
    dims=np.hstack([npar,pred.shape])
    derivs=np.empty(dims)

    dx=tod.info['dx']
    dy=tod.info['dy']
    minkasi_nb.fill_elliptical_isobeta_derivs(params,dx,dy,pred,derivs)
    return derivs,pred

def derivs_from_elliptical_gauss(params,tod,*args,**kwargs):
    npar=len(params)
    assert(npar==6)
    pred=tod.get_empty()
    dims=np.hstack([npar,pred.shape])
    derivs=np.empty(dims)

    dx=tod.info['dx']
    dy=tod.info['dy']
    minkasi_nb.fill_elliptical_gauss_derivs(params,dx,dy,pred,derivs)
    return derivs,pred


def derivs_from_isobeta_c(params,tod,*args,**kwargs):
    npar=5;
    #n=tod.info['dat_calib'].size
    dims=tod.get_data_dims()
    n=np.product(dims)
    #sz_deriv=np.append(npar,tod.info['dat_calib'].shape)
    sz_deriv=np.append(npar,dims)
    #pred=np.zeros(tod.info['dat_calib'].shape)
    pred=tod.get_empty(True)
    derivs=np.zeros(sz_deriv)

    dx=tod.info['dx']
    dy=tod.info['dy']
    fill_isobeta_derivs_c(params.ctypes.data,dx.ctypes.data,dy.ctypes.data,pred.ctypes.data,derivs.ctypes.data,n)

    return derivs,pred

def derivs_from_gauss_c(params,tod,*args,**kwargs):
    npar=4
    #n=tod.info['dat_calib'].size
    n=tod.get_nsamp()
    #sz_deriv=np.append(npar,tod.info['dat_calib'].shape)
    sz_deriv=np.append(npar,tod.get_data_dims())
    
    #pred=np.zeros(tod.info['dat_calib'].shape)
    pred=tod.get_empty(True)

    derivs=np.zeros(sz_deriv)

    dx=tod.info['dx']
    dy=tod.info['dy']
    fill_gauss_derivs_c(params.ctypes.data,dx.ctypes.data,dy.ctypes.data,pred.ctypes.data,derivs.ctypes.data,n)

    return derivs,pred

def timestreams_from_isobeta(params,tod):
    npar_beta=5 #x,y,theta,beta,amp
    npar_src=4 #x,y,sig,amp
    nsrc=(params.size-npar_beta)//npar_src
    assert(params.size==nsrc*npar_src+npar_beta)
    x0=params[0]
    y0=params[1]
    theta=params[2]
    beta=params[3]
    amp=params[4]
    cosdec=np.cos(y0)


    dx=(tod.info['dx']-x0)*cosdec
    dy=tod.info['dy']-y0
    rsqr=dx*dx+dy*dy
    rsqr=rsqr/theta**2
    #print rsqr.max()
    pred=amp*(1+rsqr)**(0.5-1.5*beta)
    for i in range(nsrc):
        src_x=params[i*npar_src+npar_beta+0]
        src_y=params[i*npar_src+npar_beta+1]
        src_sig=params[i*npar_src+npar_beta+2]
        src_amp=params[i*npar_src+npar_beta+3]
        
        dx=tod.info['dx']-src_x
        dy=tod.info['dy']-src_y
        rsqr=( (dx*np.cos(src_y))**2+dy**2)
        pred=pred+src_amp*np.exp(-0.5*rsqr/src_sig**2)

    return pred

    

def isobeta_src_chisq(params,tods):
    chisq=0.0
    for tod in tods.tods:
        pred=timestreams_from_isobeta_c(params,tod)
        #chisq=chisq+tod.timestream_chisq(tod.info['dat_calib']-pred)
        chisq=chisq+tod.timestream_chisq(tod.get_data()-pred)
        
    return chisq
    npar_beta=5 #x,y,theta,beta,amp
    npar_src=4 #x,y,sig,amp
    nsrc=(params.size-npar_beta)//npar_src
    assert(params.size==nsrc*npar_src+npar_beta)
    x0=params[0]
    y0=params[1]
    theta=params[2]
    beta=params[3]
    amp=params[4]
    cosdec=np.cos(y0)
    chisq=0.0
    for tod in tods.tods:
        dx=tod.info['dx']-x0
        dy=tod.info['dy']-y0
        rsqr=(dx*cosdec)**2+dy**2
        pred=amp*(1+rsqr/theta**2)**(0.5-1.5*beta)
        for i in range(nsrc):
            src_x=params[i*npar_src+npar_beta+0]
            src_y=params[i*npar_src+npar_beta+1]
            src_sig=params[i*npar_src+npar_beta+2]
            src_amp=params[i*npar_src+npar_beta+3]

            dx=tod.info['dx']-src_x
            dy=tod.info['dy']-src_y
            rsqr=( (dx*np.cos(src_y))**2+dy**2)
            pred=pred+src_amp*np.exp(-0.5*rsqr/src_sig**2)
        #chisq=chisq+tod.timestream_chisq(tod.info['dat_calib']-pred)
        chisq=chisq+tod.timestream_chisq(tod.get_data()-pred)
    return chisq



class NoiseBinnedDet:
    def __init__(self,dat,dt,freqs=None,scale_facs=None):
        ndet=dat.shape[0]
        ndata=dat.shape[1]
        nn=2*(ndata-1)
        dnu=1/(nn*dt)
        bins=np.asarray(freqs/dnu,dtype='int')
        bins=bins[bins<ndata]
        bins=np.hstack([bins,ndata])
        if bins[0]>0:
            bins=np.hstack([0,bins])
        if bins[0]<0:
            bins[0]=0
        self.bins=bins
        nbin=len(bins)-1
        self.nbin=nbin
        det_ps=np.zeros([ndet,nbin])
        datft=mkfftw.fft_r2r(dat)
        for i in range(nbin):
            det_ps[:,i]=1.0/np.mean(datft[:,bins[i]:bins[i+1]]**2,axis=1)
        self.det_ps=det_ps
        self.ndata=ndata
        self.ndet=ndet
        self.nn=nn
    def apply_noise(self,dat):
        datft=mkfftw.fft_r2r(dat)
        for i in range(self.nbin):
            #datft[:,self.bins[i]:self.bins[i+1]]=datft[:,self.bins[i]:self.bins[i+1]]*np.outer(self.det_ps[:,i],self.bins[i+1]-self.bins[i])
            datft[:,self.bins[i]:self.bins[i+1]]=datft[:,self.bins[i]:self.bins[i+1]]*np.outer(self.det_ps[:,i],np.ones(self.bins[i+1]-self.bins[i]))
        dd=mkfftw.fft_r2r(datft)
        dd[:,0]=0.5*dd[:,0]
        dd[:,-1]=0.5*dd[:,-1]
        return dd

class NoiseWhite:
    def __init__(self,dat):
        #this is the ratio between the median absolute
        #deviation of the diff and sigma
        fac=scipy.special.erfinv(0.5)*2
        sigs=np.median(np.abs(np.diff(dat,axis=1)),axis=1)/fac
        self.sigs=sigs
        self.weights=1/sigs**2
    def apply_noise(self,dat):
        assert(dat.shape[0]==len(self.weights))
        ndet=dat.shape[0]
        for i in range(ndet):
            dat[i,:]=dat[i,:]*self.weights[i]
        return dat
class NoiseBinnedEig:
    def __init__(self,dat,dt,freqs=None,scale_facs=None,thresh=5.0):

        ndet=dat.shape[0]
        ndata=dat.shape[1]
        nn=2*(ndata-1)

        mycov=np.dot(dat,dat.T)
        mycov=0.5*(mycov+mycov.T)
        ee,vv=np.linalg.eig(mycov)
        mask=ee>thresh*thresh*np.median(ee)
        vecs=vv[:,mask]
        ts=np.dot(vecs.T,dat)
        resid=dat-np.dot(vv[:,mask],ts)
        dnu=1/(nn*dt)
        print('dnu is ' + repr(dnu))
        bins=np.asarray(freqs/dnu,dtype='int')
        bins=bins[bins<ndata]
        bins=np.hstack([bins,ndata])
        if bins[0]>0:
            bins=np.hstack([0,bins])
        if bins[0]<0:
            bins[0]=0
        self.bins=bins
        nbin=len(bins)-1
        self.nbin=nbin

        nmode=ts.shape[0]
        det_ps=np.zeros([ndet,nbin])
        mode_ps=np.zeros([nmode,nbin])
        residft=mkfftw.fft_r2r(resid)
        modeft=mkfftw.fft_r2r(ts)
        
        for i in range(nbin):
            det_ps[:,i]=1.0/np.mean(residft[:,bins[i]:bins[i+1]]**2,axis=1)
            mode_ps[:,i]=1.0/np.mean(modeft[:,bins[i]:bins[i+1]]**2,axis=1)
        self.modes=vecs.copy()
        if not(np.all(np.isfinite(det_ps))):
            print("warning - have non-finite numbers in noise model.  This should not be unexpected.")
            det_ps[~np.isfinite(det_ps)]=0.0
        self.det_ps=det_ps
        self.mode_ps=mode_ps
        self.ndata=ndata
        self.ndet=ndet
        self.nn=nn
    def apply_noise(self,dat):
        assert(dat.shape[0]==self.ndet)
        assert(dat.shape[1]==self.ndata)
        datft=mkfftw.fft_r2r(dat)
        for i in range(self.nbin):
            n=self.bins[i+1]-self.bins[i]
            #print('bins are ',self.bins[i],self.bins[i+1],n,datft.shape[1])
            tmp=self.modes*np.outer(self.det_ps[:,i],np.ones(self.modes.shape[1]))
            mat=np.dot(self.modes.T,tmp)
            mat=mat+np.diag(self.mode_ps[:,i])
            mat_inv=np.linalg.inv(mat)
            Ax=datft[:,self.bins[i]:self.bins[i+1]]*np.outer(self.det_ps[:,i],np.ones(n))
            tmp=np.dot(self.modes.T,Ax)
            tmp=np.dot(mat_inv,tmp)
            tmp=np.dot(self.modes,tmp)
            tmp=Ax-tmp*np.outer(self.det_ps[:,i],np.ones(n))
            datft[:,self.bins[i]:self.bins[i+1]]=tmp
            #print(tmp.shape,mat.shape)
        dd=mkfftw.fft_r2r(datft)
        dd[:,0]=0.5*dd[:,0]
        dd[:,-1]=0.5*dd[:,-1]
        return dd
class NoiseCMWhite:
    def __init__(self,dat):
        u,s,v=np.linalg.svd(dat,0)
        self.ndet=len(s)
        ind=np.argmax(s)
        self.v=np.zeros(self.ndet)
        self.v[:]=u[:,ind]
        pred=np.outer(self.v*s[ind],v[ind,:])
        dat_clean=dat-pred
        myvar=np.std(dat_clean,1)**2
        self.mywt=1.0/myvar
    def apply_noise(self,dat):
        mat=np.dot(self.v,np.diag(self.mywt))
        lhs=np.dot(self.v,mat.T)
        rhs=np.dot(mat,dat)
        if isinstance(lhs,np.ndarray):
            cm=np.dot(np.linalg.inv(lhs),rhs)
        else:
            cm=rhs/lhs
        dd=dat-np.outer(self.v,cm)
        tmp=np.repeat([self.mywt],len(cm),axis=0).T
        dd=dd*tmp
        return dd
    def get_det_weights(self):
        return self.mywt.copy()

class NoiseSmoothedSVD:
    def __init__(self,dat_use,fwhm=50,prewhiten=False,fit_powlaw=False):
        if prewhiten:
            noisevec=np.median(np.abs(np.diff(dat_use,axis=1)),axis=1)
            dat_use=dat_use/(np.repeat([noisevec],dat_use.shape[1],axis=0).transpose())
        u,s,v=np.linalg.svd(dat_use,0)
        #print(u.shape,s.shape,v.shape)
        print('got svd')
        ndet=s.size
        n=dat_use.shape[1]
        self.v=np.zeros([ndet,ndet])
        self.v[:]=u.transpose()
        dat_rot=np.dot(self.v,dat_use)
        if fit_powlaw:
            spec_smooth=0*dat_rot
            for ind in range(ndet):
                fitp,datsqr,C=fit_ts_ps(dat_rot[ind,:]);
                spec_smooth[ind,1:]=C
        else:
            dat_trans=mkfftw.fft_r2r(dat_rot)
            spec_smooth=smooth_many_vecs(dat_trans**2,fwhm)
        spec_smooth[:,1:]=1.0/spec_smooth[:,1:]
        spec_smooth[:,0]=0
        if prewhiten:
            self.noisevec=noisevec.copy()
        else:
            self.noisevec=None
        self.mywt=spec_smooth
    def apply_noise(self,dat):
        if not(self.noisevec is None):
            noisemat=np.repeat([self.noisevec],dat.shape[1],axis=0).transpose()
            dat=dat/noisemat
        dat_rot=np.dot(self.v,dat)
        datft=mkfftw.fft_r2r(dat_rot)
        nn=datft.shape[1]
        datft=datft*self.mywt[:,:nn]
        dat_rot=mkfftw.fft_r2r(datft)
        dat=np.dot(self.v.T,dat_rot)
        dat[:,0]=0.5*dat[:,0]
        dat[:,-1]=0.5*dat[:,-1]
        if not(self.noisevec is None):
            #noisemat=np.repeat([self.noisevec],dat.shape[1],axis=0).transpose()
            dat=dat/noisemat        
        return dat
    def get_det_weights(self):
        """Find the per-detector weights for use in making actual noise maps."""
        mode_wt=np.sum(self.mywt,axis=1)
        tmp=np.dot(self.v.T,np.dot(np.diag(mode_wt),self.v))
        return np.diag(tmp).copy()*2.0

class Tod:
    def __init__(self,info):
        self.info=info.copy()
        self.jumps=None
        self.cuts=None
        self.noise=None
        self.noise_delayed=False
    def lims(self):
        xmin=self.info['dx'].min()
        xmax=self.info['dx'].max()
        ymin=self.info['dy'].min()
        ymax=self.info['dy'].max()
        return xmin,xmax,ymin,ymax
    def set_apix(self):
        '''calculates dxel normalized to +-1 from elevation'''
        #TBD pass in and calculate scan center's elevation vs time
        elev=np.mean(self.info['elev'],axis=0)
        x=np.arange(elev.shape[0])/elev.shape[0] 
        a=np.polyfit(x,elev,2)
        ndet=self.info['elev'].shape[0]
        track_elev,xel=np.meshgrid(a[2]+a[1]*x+a[0]*x**2,np.ones(ndet))
        delev=self.info['elev'] - track_elev
        ml=np.max(np.abs(delev))
        self.info['apix']=delev/ml
    def get_ndet(self):
        return self.info['dat_calib'].shape[0]
    def get_ndata(self):
        return self.info['dat_calib'].shape[1]
    def get_nsamp(self):
        #get total number of timestream samples, not samples per detector
        #return np.product(self.info['dat_calib'].shape)
        return self.get_ndet()*self.get_ndata()

    def get_data_dims(self):
        return (self.get_ndet(),self.get_ndata())
        #dims=self.info['dat_calib'].shape
        #if len(dims)==1:
        #    dims=np.asarray([1,dims[0]],dtype='int')
        #return dims
        #return self.info['dat_calib'].shape

    def get_data(self):
        return self.info['dat_calib']
    def get_tvec(self):
        return self.info['ctime']

    def get_radec(self):
        return self.info['dx'],self.info['dy']
    def get_empty(self,clear=False):
        if 'dtype' in self.info.keys():
            dtype=self.info['dtype']
        elif 'dat_calib' in self.info.keys():
            dtype=self.info['dat_calib'].dtype
        else:            
            dtype='float'
        if clear:
            #return np.zeros(self.info['dat_calib'].shape,dtype=self.info['dat_calib'].dtype)            
            return np.zeros([self.get_ndet(),self.get_ndata()],dtype=dtype)
        else:
            #return np.empty(self.info['dat_calib'].shape,dtype=self.info['dat_calib'].dtype)
            return np.empty([self.get_ndet(),self.get_ndata()],dtype=dtype)
    def set_tag(self,tag):
        self.info['tag']=tag
    def set_pix(self,map):
        ipix=map.get_pix(self)
        self.info['ipix']=ipix
    def copy(self,copy_info=False):
        if copy_info:
            myinfo=self.info.copy()
            for key in myinfo.keys():
                try:
                    myinfo[key]=self.info[key].copy()
                except:
                    pass
            tod=Tod(myinfo)
        else:
            tod=Tod(self.info)
        if not(self.jumps is None):
            try:
                tod.jumps=self.jumps.copy()
            except:
                tod.jumps=self.jumps[:]
        if not(self.cuts is None):
            try:
                tod.cuts=self.cuts.copy()
            except:
                tod.cuts=self.cuts[:]
            tod.cuts=self.cuts[:]
        tod.noise=self.noise
            
        return tod
    def set_noise(self,modelclass=NoiseSmoothedSVD,dat=None,delayed=False,*args,**kwargs):
        if delayed:
            self.noise_args=copy.deepcopy(args)
            self.noise_kwargs=copy.deepcopy(kwargs)
            self.noise_delayed=True
            self.noise_modelclass=modelclass
        else:
            self.noise_delayed=False
            if dat is None:
                dat=self.info['dat_calib']
            self.noise=modelclass(dat,*args,**kwargs)
    def get_det_weights(self):
        if self.noise is None:
            print("noise model not set in get_det_weights.")
            return None
        try:
            return self.noise.get_det_weights()
        except:
            print("noise model does not support detector weights in get_det_weights.")
            return None
    def set_noise_white_masked(self):
        self.info['noise']='white_masked'
        self.info['mywt']=np.ones(self.info['dat_calib'].shape[0])
    def apply_noise_white_masked(self,dat=None):
        if dat is None:
            dat=self.info['dat_calib']
        dd=self.info['mask']*dat*np.repeat([self.info['mywt']],self.info['dat_calib'].shape[1],axis=0).transpose()
        return dd
    def set_noise_cm_white(self):
        print('deprecated usage - please switch to tod.set_noise(minkasi.NoiseCMWhite)')
        self.set_noise(NoiseCMWhite)
        return

        u,s,v=np.linalg.svd(self.info['dat_calib'],0)
        ndet=len(s)
        ind=np.argmax(s)
        mode=np.zeros(ndet)
        #mode[:]=u[:,0]  #bug fixes pointed out by Fernando Zago.  19 Nov 2019
        #pred=np.outer(mode,v[0,:])
        mode[:]=u[:,ind]
        pred=np.outer(mode*s[ind],v[ind,:])

        dat_clean=self.info['dat_calib']-pred
        myvar=np.std(dat_clean,1)**2
        self.info['v']=mode
        self.info['mywt']=1.0/myvar
        self.info['noise']='cm_white'
        
    def apply_noise_cm_white(self,dat=None):
        print("I'm not sure how you got here (tod.apply_noise_cm_white), but you should not have been able to.  Please complain to someone.")
        if dat is None:
            dat=self.info['dat_calib']

        mat=np.dot(self.info['v'],np.diag(self.info['mywt']))
        lhs=np.dot(self.info['v'],mat.transpose())
        rhs=np.dot(mat,dat)
        #if len(lhs)>1:
        if isinstance(lhs,np.ndarray):
            cm=np.dot(np.linalg.inv(lhs),rhs)
        else:
            cm=rhs/lhs
        dd=dat-np.outer(self.info['v'],cm)
        tmp=np.repeat([self.info['mywt']],len(cm),axis=0).transpose()
        dd=dd*tmp
        return dd
    def set_noise_binned_eig(self,dat=None,freqs=None,scale_facs=None,thresh=5.0):
        if dat is None:
            dat=self.info['dat_calib']
        mycov=np.dot(dat,dat.T)
        mycov=0.5*(mycov+mycov.T)
        ee,vv=np.linalg.eig(mycov)
        mask=ee>thresh*thresh*np.median(ee)
        vecs=vv[:,mask]
        ts=np.dot(vecs.T,dat)
        resid=dat-np.dot(vv[:,mask],ts)
        
        return resid
    def set_noise_smoothed_svd(self,fwhm=50,func=None,pars=None,prewhiten=False,fit_powlaw=False):
        '''If func comes in as not empty, assume we can call func(pars,tod) to get a predicted model for the tod that
        we subtract off before estimating the noise.'''

        
        print('deprecated usage - please switch to tod.set_noise(minkasi.NoiseSmoothedSVD)')

        if func is None:
            self.set_noise(NoiseSmoothedSVD,self.info['dat_calib'])
        else:
            dat_use=func(pars,self)
            dat_use=self.info['dat_calib']-dat_use
            self.set_noise(NoiseSmoothedSVD,dat_use)
        return


        if func is None:
            dat_use=self.info['dat_calib']
        else:
            dat_use=func(pars,self)
            dat_use=self.info['dat_calib']-dat_use
            #u,s,v=numpy.linalg.svd(self.info['dat_calib']-tmp,0)
        if prewhiten:
            noisevec=np.median(np.abs(np.diff(dat_use,axis=1)),axis=1)
            dat_use=dat_use/(np.repeat([noisevec],dat_use.shape[1],axis=0).transpose())
        u,s,v=np.linalg.svd(dat_use,0)
        print('got svd')
        ndet=s.size
        n=self.info['dat_calib'].shape[1]
        self.info['v']=np.zeros([ndet,ndet])
        self.info['v'][:]=u.transpose()
        dat_rot=np.dot(self.info['v'],self.info['dat_calib'])
        if fit_powlaw:
            spec_smooth=0*dat_rot
            for ind in range(ndet):
                fitp,datsqr,C=fit_ts_ps(dat_rot[ind,:]);
                spec_smooth[ind,1:]=C
        else:
            dat_trans=mkfftw.fft_r2r(dat_rot)
            spec_smooth=smooth_many_vecs(dat_trans**2,fwhm)
        spec_smooth[:,1:]=1.0/spec_smooth[:,1:]
        spec_smooth[:,0]=0
        if prewhiten:
            self.info['noisevec']=noisevec.copy()
        self.info['mywt']=spec_smooth
        self.info['noise']='smoothed_svd'
        #return dat_rot
        
    def apply_noise(self,dat=None):
        if dat is None:
            #dat=self.info['dat_calib']
            dat=self.get_data().copy() #the .copy() is here so you don't
                                       #overwrite data stored in the TOD.
        if self.noise_delayed:
            self.noise=self.noise_modelclass(dat,*(self.noise_args), **(self.noise_kwargs))
            self.noise_delayed=False
        try:
            return self.noise.apply_noise(dat)
        except:
            print("unable to use class-based noised, falling back onto hardwired.")
            
        if self.info['noise']=='cm_white':
            #print 'calling cm_white'
            return self.apply_noise_cm_white(dat)
        if self.info['noise']=='white_masked':
            return self.apply_noise_white_masked(dat)
        #if self.info.has_key('noisevec'):
        if 'noisevec' in self.info:
            noisemat=np.repeat([self.info['noisevec']],dat.shape[1],axis=0).transpose()
            dat=dat/noisemat
        dat_rot=np.dot(self.info['v'],dat)

        datft=mkfftw.fft_r2r(dat_rot)
        nn=datft.shape[1]
        datft=datft*self.info['mywt'][:,0:nn]
        dat_rot=mkfftw.fft_r2r(datft)
        dat=np.dot(self.info['v'].transpose(),dat_rot)
        #if self.info.has_key('noisevec'):
        if 'noisevec' in self.info:
            dat=dat/noisemat
        dat[:,0]=0.5*dat[:,0]
        dat[:,-1]=0.5*dat[:,-1]

        return dat
    def mapset2tod(self,mapset,dat=None):
        if dat is None:
            #dat=0*self.info['dat_calib']
            dat=self.get_empty(True)
        for map in mapset.maps:
            map.map2tod(self,dat)
        return dat
    def tod2mapset(self,mapset,dat=None):                     
        if dat is None:
            #dat=self.info['dat_calib']
            dat=self.get_data()
        for map in mapset.maps:
            map.tod2map(self,dat)
    def dot(self,mapset,mapset_out):
        #tmp=0.0*self.info['dat_calib']
        #for map in mapset.maps:
        #    map.map2tod(self,tmp)
        tmp=self.mapset2tod(mapset)
        tmp=self.apply_noise(tmp)
        self.tod2mapset(mapset_out,tmp)
        #for map in mapset_out.maps:
        #    map.tod2map(self,tmp)
    def set_jumps(self,jumps):
        self.jumps=jumps
    def cut_detectors(self,isgood):
        #cut all detectors not in boolean array isgood
        isbad=np.asarray(1-isgood,dtype='bool')
        bad_inds=np.where(isbad)
        bad_inds=np.fliplr(bad_inds)
        bad_inds=bad_inds[0]
        print(bad_inds)
        nkeep=np.sum(isgood)
        for key in self.info.keys():
            if isinstance(self.info[key],np.ndarray):
                self.info[key]=slice_with_copy(self.info[key],isgood)
        if not(self.jumps is None):
            for i in bad_inds:
                print('i in bad_inds is ',i)
                del(self.jumps[i])
        if not(self.cuts is None):
            for i in bad_inds:
                del(self.cuts[i])
                
    def timestream_chisq(self,dat=None):
        if dat is None:
            dat=self.info['dat_calib']
        dat_filt=self.apply_noise(dat)
        chisq=np.sum(dat_filt*dat)
        return chisq
    def prior_from_skymap(self,skymap):
        """stuff.
        prior_from_skymap(self,skymap):
        Given e.g. the gradient of a map that has been zeroed under some threshold,
        return a CutsCompact object that can be used as a prior for solving for per-sample deviations
        due to strong map gradients.  This is to reduce X's around bright sources.  The input map
        should be a SkyMap that is non-zero where one wishes to solve for the per-sample deviations, and 
        the non-zero values should be the standard deviations expected in those pixel.  The returned CutsCompact 
        object will have the weight (i.e. 1/input squared) in its map.        
        """
        tmp=np.zeros(self.info['dat_calib'].shape)
        skymap.map2tod(self,tmp)
        mask=(tmp==0)
        prior=CutsCompact(self)
        prior.cuts_from_array(mask)
        prior.get_imap()
        prior.tod2map(self,tmp)
        prior.map=1.0/prior.map**2
        return prior

def slice_with_copy(arr,ind):
    if isinstance(arr,np.ndarray):
        myshape=arr.shape

        if len(myshape)==1:
            ans=np.zeros(ind.sum(),dtype=arr.dtype)
            print(ans.shape)
            print(ind.sum())
            ans[:]=arr[ind]
        else:   
            mydims=np.append(np.sum(ind),myshape[1:])
            print(mydims,mydims.dtype)
            ans=np.zeros(mydims,dtype=arr.dtype)
            ans[:,:]=arr[ind,:].copy()
        return ans
    return None #should not get here
class TodVec:
    def __init__(self):
        self.tods=[]
        self.ntod=0
    def add_tod(self,tod):

        self.tods.append(tod.copy())
        self.tods[-1].set_tag(self.ntod)
        self.ntod=self.ntod+1
    def lims(self):
        if self.ntod==0:
            return None
        xmin,xmax,ymin,ymax=self.tods[0].lims()
        for i in range(1,self.ntod):
            x1,x2,y1,y2=self.tods[i].lims()
            xmin=min(x1,xmin)
            xmax=max(x2,xmax)
            ymin=min(y1,ymin)
            ymax=max(y2,ymax)
        if have_mpi:
            print('before reduction lims are ',[xmin,xmax,ymin,ymax])
            xmin=comm.allreduce(xmin,op=MPI.MIN)
            xmax=comm.allreduce(xmax,op=MPI.MAX)
            ymin=comm.allreduce(ymin,op=MPI.MIN)
            ymax=comm.allreduce(ymax,op=MPI.MAX)
            print('after reduction lims are ',[xmin,xmax,ymin,ymax])
        return [xmin,xmax,ymin,ymax]
    def set_pix(self,map):
        for tod in self.tods:
            #ipix=map.get_pix(tod)
            #tod.info['ipix']=ipix
            tod.set_pix(map)
    def set_apix(self):
        for tod in self.tods:
            tod.set_apix()
    def dot_cached(self,mapset,mapset2=None):
        nthread=get_nthread()
        mapset2.get_caches()
        for i in range(self.ntod):
            tod=self.tods[i]
            tod.dot(mapset,mapset2)
        mapset2.clear_caches()
        if have_mpi:
            mapset2.mpi_reduce()

        return mapset2

    def dot(self,mapset,mapset2=None,report_times=False,cache_maps=False):
        if mapset2 is None:
            mapset2=mapset.copy()
            mapset2.clear()

        if cache_maps:
            mapset2=self.dot_cached(mapset,mapset2)
            return mapset2
            

        times=np.zeros(self.ntod)
        #for tod in self.tods:
        for i in range(self.ntod):
            tod=self.tods[i]
            t1=time.time()
            tod.dot(mapset,mapset2)
            t2=time.time()
            times[i]=t2-t1
        if have_mpi:
            mapset2.mpi_reduce()
        if report_times:
            return mapset2,times
        else:
            return mapset2
    def make_rhs(self,mapset,do_clear=False):
        if do_clear:
            mapset.clear()
        for tod in self.tods:
            dat_filt=tod.apply_noise()
            for map in mapset.maps:
                map.tod2map(tod,dat_filt)
        
        if have_mpi:
            mapset.mpi_reduce()

def read_tod_from_fits_cbass(fname,dopol=False,lat=37.2314,lon=-118.2941,v34=True,nm20=False):
    f=pyfits.open(fname)
    raw=f[1].data
    ra=raw['RA']
    dec=raw['DEC']
    flag=raw['FLAG']
    I=0.5*(raw['I1']+raw['I2'])


    mjd=raw['MJD']
    tvec=(mjd-2455977.5+2400000.5)*86400+1329696000
    #(mjd-2455977.5)*86400+1329696000;
    dt=np.median(np.diff(tvec))

    dat={}
    dat['dx']=np.reshape(np.asarray(ra,dtype='float64'),[1,len(ra)])
    dat['dy']=np.reshape(np.asarray(dec,dtype='float64'),[1,len(dec)])
    dat['dt']=dt
    dat['ctime']=tvec
    if dopol:
        dat['dx']=np.vstack([dat['dx'],dat['dx']])
        dat['dy']=np.vstack([dat['dy'],dat['dy']])
        Q=0.5*(raw['Q1']+raw['Q2'])
        U=0.5*(raw['U1']+raw['U2'])
        dat['dat_calib']=np.zeros([2,len(Q)])
        if v34:  #We believe this is the correct sign convention for V34
            dat['dat_calib'][0,:]=-U
            dat['dat_calib'][1,:]=Q
        else:
            dat['dat_calib'][0,:]=Q
            dat['dat_calib'][1,:]=U            
        az=raw['AZ']
        el=raw['EL']
        dat['az']=az
        dat['el']=el
        
        #dat['AZ']=az
        #dat['EL']=el
        #dat['ctime']=tvec
        dat['mask']=np.zeros([2,len(Q)],dtype='int8')
        dat['mask'][0,:]=1-raw['FLAG']
        dat['mask'][1,:]=1-raw['FLAG']
        if have_qp:
            Q = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True,num_threads=4)
            #q_bore = Q.azel2bore(dat['AZ'], dat['EL'], 0*dat['AZ'], 0*dat['AZ'], lon*np.pi/180, lat*np.pi/180, dat['ctime'])
            q_bore = Q.azel2bore(az,el, 0*az, 0*az, lon, lat, dat['ctime'])
            q_off = Q.det_offset(0.0,0.0,0.0)
            #ra, dec, sin2psi, cos2psi = Q.bore2radec(q_off, ctime, q_bore)
            ra, dec, sin2psi, cos2psi = Q.bore2radec(q_off, tvec, q_bore)
            tmp=np.arctan2(sin2psi,cos2psi) 
            tmp=tmp-np.pi/2 #this seems to be needed to get these coordinates to line up with 
                            #the expected, in IAU convention I believe.  JLS Nov 12 2020
            #dat['twogamma_saved']=np.arctan2(sin2psi,cos2psi)
            dat['twogamma_saved']=np.vstack([tmp,tmp+np.pi/2])
            #print('pointing rms is ',np.std(ra*np.pi/180-dat['dx']),np.std(dec*np.pi/180-dat['dy']))
            dat['ra']=ra*np.pi/180
            dat['dec']=dec*np.pi/180
    else:
        dat['dat_calib']=np.zeros([1,len(I)])
        dat['dat_calib'][:]=I
        dat['mask']=np.zeros([1,len(I)],dtype='int8')
        dat['mask'][:]=1-raw['FLAG']


    dat['pixid']=[0]
    dat['fname']=fname

    if nm20:
        try:
        #kludget to read in bonus cuts, which should be in f[3]
            raw=f[3].data 
            dat['nm20_start']=raw['START']
            dat['nm20_stop']=raw['END']
            #nm20=0*dat['flag']
            print(dat.keys())
            nm20=0*dat['mask']
            start=dat['nm20_start']
            stop=dat['nm20_stop']
            for i in range(len(start)):
                nm20[:,start[i]:stop[i]]=1
                #nm20[:,start[i]:stop[i]]=0
            dat['mask']=dat['mask']*nm20
        except:
            print('missing nm20 for ',fname)

    f.close()
    return dat

def read_tod_from_fits(fname,hdu=1,branch=None):
    f=pyfits.open(fname)
    raw=f[hdu].data
    #print 'sum of cut elements is ',np.sum(raw['UFNU']<9e5)
    try : #read in calinfo (per-scan beam volumes etc) if present
        calinfo={'calinfo':True}
        kwds=('scan','bunit','azimuth','elevatio','beameff','apereff','antgain','gainunc','bmaj','bmin','bpa','parang','beamvol','beamvunc')#for now just hardwired ones we want
        for kwd in kwds:
            calinfo[kwd]=f[hdu].header[kwd]
    except KeyError : 
        print('WARNING - calinfo information not found in fits file header - to track JytoK etc you may need to reprocess the fits files using mustangidl > revision 932') 
        calinfo['calinfo']=False

    pixid=raw['PIXID']
    dets=np.unique(pixid)
    ndet=len(dets)
    nsamp=len(pixid)/len(dets)
    if True:
        ff=180/np.pi
        xmin=raw['DX'].min()*ff
        xmax=raw['DX'].max()*ff
        ymin=raw['DY'].min()*ff
        ymax=raw['DY'].max()*ff
        print('nsamp and ndet are ',ndet,nsamp,len(pixid),' on ',fname, 'with lims ',xmin,xmax,ymin,ymax)
    else:
        print('nsamp and ndet are ',ndet,nsamp,len(pixid),' on ',fname)
    #print raw.names
    dat={}
    #this bit of odd gymnastics is because a straightforward reshape doesn't seem to leave the data in
    #memory-contiguous order, which causes problems down the road
    #also, float32 is a bit on the edge for pointing, so cast to float64
    dx=raw['DX']
    if not(branch is None):
        bb=branch*np.pi/180.0
        dx[dx>bb]=dx[dx>bb]-2*np.pi
    #dat['dx']=np.zeros([ndet,nsamp],dtype=type(dx[0]))
    ndet=np.int(ndet)
    nsamp=np.int(nsamp)
    dat['dx']=np.zeros([ndet,nsamp],dtype='float64')
    dat['dx'][:]=np.reshape(dx,[ndet,nsamp])[:]
    dy=raw['DY']
    #dat['dy']=np.zeros([ndet,nsamp],dtype=type(dy[0]))
    dat['dy']=np.zeros([ndet,nsamp],dtype='float64')
    dat['dy'][:]=np.reshape(dy,[ndet,nsamp])[:]
    if 'ELEV' in raw.names:
        elev=raw['ELEV']*np.pi/180
        dat['elev']=np.zeros([ndet,nsamp],dtype='float64')
        dat['elev'][:]=np.reshape(elev,[ndet,nsamp])[:]

    tt=np.reshape(raw['TIME'],[ndet,nsamp])
    tt=tt[0,:]
    dt=np.median(np.diff(tt))
    dat['dt']=dt
    pixid=np.reshape(pixid,[ndet,nsamp])
    pixid=pixid[:,0]
    dat['pixid']=pixid
    dat_calib=raw['FNU']
    #print 'shapes are ',raw['FNU'].shape,raw['UFNU'].shape,np.mean(raw['UFNU']>9e5)
    #dat_calib[raw['UFNU']>9e5]=0.0

    #dat['dat_calib']=np.zeros([ndet,nsamp],dtype=type(dat_calib[0]))
    dat['dat_calib']=np.zeros([ndet,nsamp],dtype='float64') #go to double because why not
    dat_calib=np.reshape(dat_calib,[ndet,nsamp])

    dat['dat_calib'][:]=dat_calib[:]
    if np.sum(raw['UFNU']>9e5)>0:
        dat['mask']=np.reshape(raw['UFNU']<9e5,dat['dat_calib'].shape)
        dat['mask_sum']=np.sum(dat['mask'],axis=0)
    #print 'cut frac is now ',np.mean(dat_calib==0)
    #print 'cut frac is now ',np.mean(dat['dat_calib']==0),dat['dat_calib'][0,0]
    dat['fname']=fname
    dat['calinfo']=calinfo
    f.close()
    return dat


def downsample_array_r2r(arr,fac):

    n=arr.shape[1]
    nn=int(n/fac)
    arr_ft=mkfftw.fft_r2r(arr)
    arr_ft=arr_ft[:,0:nn].copy()
    arr=mkfftw.fft_r2r(arr_ft)/(2*(n-1))
    return arr

def downsample_vec_r2r(vec,fac):

    n=len(vec)
    nn=int(n/fac)
    vec_ft=mkfftw.fft_r2r(vec)
    vec_ft=vec_ft[0:nn].copy()
    vec=mkfftw.fft_r2r(vec_ft)/(2*(n-1))
    return vec

def downsample_tod(dat,fac=10):
    ndata=dat['dat_calib'].shape[1]
    keys=dat.keys()
    for key in dat.keys():
        try:
            if len(dat[key].shape)==1:
                #print('working on downsampling ' + key)
                #print('shape is ' + repr(dat[key].shape[0])+'  '+repr(n))
                if len(dat[key]):
                    #print('working on downsampling ' + key)
                    dat[key]=downsample_vec_r2r(dat[key],fac)
            else:
                if dat[key].shape[1]==ndata:
                #print 'downsampling ' + key
                    dat[key]=downsample_array_r2r(dat[key],fac)
        except:
            #print 'not downsampling ' + key
            pass
    

def truncate_tod(dat,primes=[2,3,5,7,11]):
    n=dat['dat_calib'].shape[1]
    lens=find_good_fft_lens(n-1,primes)
    n_new=lens.max()+1
    if n_new<n:
        print('truncating from ',n,' to ',n_new)
        for key in dat.keys():
            try:
                #print('working on key ' + key)
                if len(dat[key].shape)==1:
                    if dat[key].shape[0]==n:
                        dat[key]=dat[key][:n_new].copy()
                else:
                    if dat[key].shape[1]==n:
                        dat[key]=dat[key][:,0:n_new].copy()
            except:
                #print('skipping key ' + key)
                pass




def todvec_from_files_octave(fnames):
    todvec=TodVec()
    for fname in fnames:
        info=read_octave_struct(fname)
        tod=Tod(info)
        todvec.add_tod(tod)
    return todvec
        
def make_hits(todvec,map,do_weights=False):
    hits=map.copy()
    try:
        if map.npol>1:
            hits.set_polstate(map.poltag+'_PRECON')
    except:
        pass
    hits.clear()
    for tod in todvec.tods:
        if do_weights:
            try:
                weights=tod.get_det_weights()
                #sz=tod.info['dat_calib'].shape
                sz=tod.get_data_dims()
                tmp=np.outer(weights,np.ones(sz[1]))
                #tmp=np.outer(weights,np.ones(tod.info['dat_calb'].shape[1]))
            except:
                print("error in making weight map.  Detector weights requested, but do not appear to be present.  Do you have a noise model?")
                             
        else:
            #tmp=np.ones(tod.info['dat_calib'].shape)
            tmp=np.ones(tod.get_data_dims())
        #if tod.info.has_key('mask'):
        if 'mask' in tod.info:
            tmp=tmp*tod.info['mask']
        hits.tod2map(tod,tmp)
    if have_mpi:
        print('reducing hits')
        tot=hits.map.sum()
        print('starting with total hitcount ' + repr(tot))
        hits.mpi_reduce()
        tot=hits.map.sum()
        print('ending with total hitcount ' + repr(tot))
    return hits


def decimate(vec,nrep=1):
    for i in range(nrep):
        if len(vec)%2:
            vec=vec[:-1]
        vec=0.5*(vec[0::2]+vec[1::2])
    return vec
def plot_ps(vec,downsamp=0):
    vecft=mkfftw.fft_r2r(vec)
    
def get_wcs(lims,pixsize,proj='CAR',cosdec=None,ref_equ=False):
    w=wcs.WCS(naxis=2)    
    dec=0.5*(lims[2]+lims[3])
    if cosdec is None:
        cosdec=np.cos(dec)
    if proj=='CAR':
        #CAR in FITS seems to already correct for cosin(dec), which has me confused, but whatever...
        cosdec=1.0
        if ref_equ:
            w.wcs.crval=[0.0,0.0]
            #this seems to be a needed hack if you want the position sent
            #in for the corner to actually be the corner.
            w.wcs.crpix=[lims[1]/pixsize+1,-lims[2]/pixsize+1]
            #w.wcs.crpix=[lims[1]/pixsize,-lims[2]/pixsize]
            #print 'crpix is ',w.wcs.crpix
        else:
            w.wcs.crpix=[1.0,1.0]
            w.wcs.crval=[lims[1]*180/np.pi,lims[2]*180/np.pi]
        w.wcs.cdelt=[-pixsize/cosdec*180/np.pi,pixsize*180/np.pi]
        w.wcs.ctype=['RA---CAR','DEC--CAR']
        return w
    print('unknown projection type ',proj,' in get_wcs.')
    return None



def get_aligned_map_subregion_car(lims,fname=None,big_wcs=None,osamp=1):
    """Get a wcs for a subregion of a map, with optionally finer pixellization.  
    Designed for use in e.g. combining ACT maps and Mustang data.  Input arguments
    are RA/Dec limits for the subregion (which will be tweaked as-needed) and either a 
    WCS structure or the name of a FITS file containing the WCS info the sub-region
    will be aligned with."""
    
    if big_wcs is None:
        if fname is None:
            print("Error in get_aligned_map_subregion_car.  Must send in either a file or a WCS.")
        big_wcs=wcs.WCS(fname)
    ll=np.asarray(lims)
    ll=ll*180/np.pi 
    
    #get the ra/dec limits in big pixel coordinates
    corner1=big_wcs.wcs_world2pix(ll[0],ll[2],0)
    corner2=big_wcs.wcs_world2pix(ll[1],ll[3],0)

    #get the pixel edges for the corners.  FITS works in
    #pixel centers, so edges are a half-pixel off
    corner1[0]=np.ceil(corner1[0])+0.5
    corner1[1]=np.floor(corner1[1])-0.5
    corner2[0]=np.floor(corner2[0])-0.5
    corner2[1]=np.ceil(corner2[1])+0.5
    
    corner1_radec=big_wcs.wcs_pix2world(corner1[0],corner1[1],0)
    corner2_radec=big_wcs.wcs_pix2world(corner2[0],corner2[1],0)

    dra=(corner1_radec[0]-corner2_radec[0])/(corner1[0]-corner2[0])
    ddec=(corner1_radec[1]-corner2_radec[1])/(corner1[1]-corner2[1])
    assert(np.abs(dra/ddec)-1<1e-5)  #we are not currently smart enough to deal with rectangular pixels
    
    lims_use=np.asarray([corner1_radec[0],corner2_radec[0],corner1_radec[1],corner2_radec[1]])
    pixsize=ddec/osamp
    lims_use=lims_use+np.asarray([0.5,-0.5,0.5,-0.5])*pixsize
    
    small_wcs=get_wcs(lims_use*np.pi/180,pixsize*np.pi/180,ref_equ=True)
    imin=np.int(np.round(corner2[0]+0.5))
    jmin=np.int(np.round(corner1[1]+0.5))
    map_corner=np.asarray([imin,jmin],dtype='int')
    lims_use=lims_use*np.pi/180

    return small_wcs,lims_use,map_corner






def fit_linear_ps_uncorr(dat,vecs,tol=1e-3,guess=None,max_iter=15):
    if guess is None:
        lhs=np.dot(vecs,vecs.transpose())
        rhs=np.dot(vecs,dat**2)
        guess=np.dot(np.linalg.inv(lhs),rhs) 
        guess=0.5*guess #scale down since we're less likely to run into convergence issues if we start low
        #print guess
    fitp=guess.copy()
    converged=False
    npp=len(fitp)
    iter=0
    
    grad_tr=np.zeros(npp)
    grad_chi=np.zeros(npp)
    curve=np.zeros([npp,npp])
    datsqr=dat*dat
    while (converged==False):
        iter=iter+1
        C=np.dot(fitp,vecs)
        Cinv=1.0/C
        for i in range(npp):
            grad_chi[i]=0.5*np.sum(datsqr*vecs[i,:]*Cinv*Cinv)
            grad_tr[i]=-0.5*np.sum(vecs[i,:]*Cinv)
            for j in range(i,npp):
                #curve[i,j]=-0.5*np.sum(datsqr*Cinv*Cinv*Cinv*vecs[i,:]*vecs[j,:]) #data-only curvature
                #curve[i,j]=-0.5*np.sum(Cinv*Cinv*vecs[i,:]*vecs[j,:]) #Fisher curvature
                curve[i,j]=0.5*np.sum(Cinv*Cinv*vecs[i,:]*vecs[j,:])-np.sum(datsqr*Cinv*Cinv*Cinv*vecs[i,:]*vecs[j,:]) #exact
                curve[j,i]=curve[i,j]
        grad=grad_chi+grad_tr
        curve_inv=np.linalg.inv(curve)
        errs=np.diag(curve_inv)
        dp=np.dot(grad,curve_inv)
        fitp=fitp-dp
        frac_shift=dp/errs
        #print dp,errs,frac_shift
        if np.max(np.abs(frac_shift))<tol:
            print('successful convergence after ',iter,' iterations with error estimate ',np.max(np.abs(frac_shift)))
            converged=True
            print(C[0],C[-1])
        if iter==max_iter:
            print('not converging after ',iter,' iterations in fit_linear_ps_uncorr with current convergence parameter ',np.max(np.abs(frac_shift)))
            converged=True
            
    return fitp

def get_curve_deriv_powspec(fitp,nu_scale,lognu,datsqr,vecs):
    vec=nu_scale**fitp[2]
    C=fitp[0]+fitp[1]*vec
    Cinv=1.0/C
    vecs[1,:]=vec
    vecs[2,:]=fitp[1]*lognu*vec
    grad_chi=0.5*np.dot(vecs,datsqr*Cinv*Cinv)
    grad_tr=-0.5*np.dot(vecs,Cinv)
    grad=grad_chi+grad_tr
    np=len(grad_chi)
    curve=np.zeros([np,np])
    for i in range(np):
        for j in range(i,np):
            curve[i,j]=0.5*np.sum(Cinv*Cinv*vecs[i,:]*vecs[j,:])-np.sum(datsqr*Cinv*Cinv*Cinv*vecs[i,:]*vecs[j,:])
            curve[j,i]=curve[i,j]
    like=-0.5*sum(datsqr*Cinv)-0.5*sum(np.log(C))
    return like,grad,curve,C

def fit_ts_ps(dat,dt=1.0,ind=-1.5,nu_min=0.0,nu_max=np.inf,scale_fac=1.0,tol=0.01):

    datft=mkfftw.fft_r2r(dat)
    n=len(datft)

    dnu=0.5/(len(dat)*dt) #coefficient should reflect the type of fft you did...
    nu=dnu*np.arange(n)
    isgood=(nu>nu_min)&(nu<nu_max)
    datft=datft[isgood]
    nu=nu[isgood]
    n=len(nu)
    vecs=np.zeros([2,n])
    vecs[0,:]=1.0 #white noise
    vecs[1,:]=nu**ind
    guess=fit_linear_ps_uncorr(datft,vecs)
    pred=np.dot(guess,vecs)
    #pred=guess[0]*vecs[0]+guess[1]*vecs[1]
    #return pred

    rat=vecs[1,:]*guess[1]/(vecs[0,:]*guess[0])
    #print 'rat lims are ',rat.max(),rat.min()
    my_ind=np.max(np.where(rat>1)[0])
    nu_ref=np.sqrt(nu[my_ind]*nu[0]) #WAG as to a sensible frequency pivot point

    #nu_ref=0.2*nu[my_ind] #WAG as to a sensible frequency pivot point
    #print 'knee is roughly at ',nu[my_ind],nu_ref

    #model = guess[1]*nu^ind+guess[0]
    #      = guess[1]*(nu/nu_ref*nu_ref)^ind+guess[0]
    #      = guess[1]*(nu_ref)^in*(nu/nu_ref)^ind+guess[0]

    nu_scale=nu/nu_ref
    guess_scale=guess.copy()
    guess_scale[1]=guess[1]*(nu_ref**ind)
    #print 'guess is ',guess
    #print 'guess_scale is ',guess_scale
    C_scale=guess_scale[0]+guess_scale[1]*(nu_scale**ind)
    

    fitp=np.zeros(3)
    fitp[0:2]=guess_scale
    fitp[2]=ind

    npp=3
    vecs=np.zeros([npp,n])
    vecs[0,:]=1.0
    lognu=np.log(nu_scale)
    curve=np.zeros([npp,npp])
    grad_chi=np.zeros(npp)
    grad_tr=np.zeros(npp)
    datsqr=datft**2
    #for robustness, start with downscaling 1/f part
    fitp[1]=0.5*fitp[1]
    like,grad,curve,C=get_curve_deriv_powspec(fitp,nu_scale,lognu,datsqr,vecs)
    lamda=0.0
    #print 'starting likelihood is',like
    for iter in range(50):
        tmp=curve+lamda*np.diag(np.diag(curve))
        curve_inv=np.linalg.inv(tmp)
        dp=np.dot(grad,curve_inv)
        trial_fitp=fitp-dp
        errs=np.sqrt(-np.diag(curve_inv))
        frac=dp/errs
        new_like,new_grad,new_curve,C=get_curve_deriv_powspec(trial_fitp,nu_scale,lognu,datsqr,vecs)

        if (new_like>like):
        #if True:
            like=new_like
            grad=new_grad
            curve=new_curve
            fitp=trial_fitp
            lamda=update_lamda(lamda,True)
        else:
            lamda=update_lamda(lamda,False)
        if (lamda==0)&(np.max(np.abs(frac))<tol):
            converged=True
        else:
            converged=False
        if False:
            vec=nu_scale**fitp[2]
            C=fitp[0]+fitp[1]*vec
            Cinv=1.0/C
            vecs[1,:]=vec
            vecs[2,:]=fitp[1]*lognu*vec
            like=-0.5*np.sum(datsqr*Cinv)-0.5*np.sum(np.log(C))
            for i in range(np):
                grad_chi[i]=0.5*np.sum(datsqr*vecs[i,:]*Cinv*Cinv)
                grad_tr[i]=-0.5*np.sum(vecs[i,:]*Cinv)
                for j in range(i,np):
                    curve[i,j]=0.5*np.sum(Cinv*Cinv*vecs[i,:]*vecs[j,:])-np.sum(datsqr*Cinv*Cinv*Cinv*vecs[i,:]*vecs[j,:])
                    curve[j,i]=curve[i,j]
            grad=grad_chi+grad_tr
            curve_inv=np.linalg.inv(curve)
            errs=np.diag(curve_inv)
            dp=np.dot(grad,curve_inv)
            fitp=fitp-dp*scale_fac
            frac_shift=dp/errs

        #print fitp,errs,frac_shift,np.mean(np.abs(new_grad-grad))
        #print fitp,grad,frac,lamda
        if converged:
            print('converged after ',iter,' iterations')
            break



    #C=np.dot(guess,vecs)
    print('mean diff is ',np.mean(np.abs(C_scale-C)))
    #return datft,vecs,nu,C
    return fitp,datsqr,C
    
def get_derivs_tod_isosrc(pars,tod,niso=None):
    np_src=4
    np_iso=5
    #nsrc=(len(pars)-np_iso)/np_src
    npp=len(pars)
    if niso is None:
        niso=(npp%np_src)/(np_iso-np_src)
    nsrc=(npp-niso*np_iso)/np_src
    #print nsrc,niso
    

    fitp_iso=np.zeros(np_iso)
    fitp_iso[:]=pars[:np_iso]
    #print 'fitp_iso is ',fitp_iso
    derivs_iso,f_iso=derivs_from_isobeta_c(fitp_iso,tod)

    #nn=tod.info['dat_calib'].size
    nn=tod.get_nsamp()
    derivs=np.reshape(derivs_iso,[np_iso,nn])
    pred=f_iso

    for ii in range(nsrc):
        fitp_src=np.zeros(np_src)
        istart=np_iso+ii*np_src
        fitp_src[:]=pars[istart:istart+np_src]
        derivs_src,f_src=derivs_from_gauss_c(fitp_src,tod)
        pred=pred+f_src
        derivs_src_tmp=np.reshape(derivs_src,[np_src,nn])
        derivs=np.append(derivs,derivs_src_tmp,axis=0)
    return pred,derivs

def get_curve_deriv_tod_manygauss(pars,tod,return_vecs=False):
    npp=4
    nsrc=len(pars)//npp
    fitp_gauss=np.zeros(npp)
    #dat=tod.info['dat_calib']
    dat=tod.get_data()
    big_derivs=np.zeros([npp*nsrc,dat.shape[0],dat.shape[1]])
    pred=0
    curve=np.zeros([npp*nsrc,npp*nsrc])
    deriv=np.zeros([npp*nsrc])
    for i in range(nsrc):
        fitp_gauss[:]=pars[i*npp:(i+1)*npp]
        derivs,src_pred=derivs_from_gauss_c(fitp_gauss,tod)
        pred=pred+src_pred
        big_derivs[i*npp:(i+1)*npp,:,:]=derivs
    delt=dat-pred
    delt_filt=tod.apply_noise(delt)
    chisq=0.5*np.sum(delt[:,0]*delt_filt[:,0])
    chisq=chisq+0.5*np.sum(delt[:,-1]*delt_filt[:,-1])
    chisq=chisq+np.sum(delt[:,1:-1]*delt_filt[:,1:-1])
    for i in range(npp*nsrc):
        deriv_filt=tod.apply_noise(big_derivs[i,:,:])
        for j in range(i,npp*nsrc):
            curve[i,j]=curve[i,j]+0.5*np.sum(deriv_filt[:,0]*big_derivs[j,:,0])
            curve[i,j]=curve[i,j]+0.5*np.sum(deriv_filt[:,-1]*big_derivs[j,:,-1])
            curve[i,j]=curve[i,j]+np.sum(deriv_filt[:,1:-1]*big_derivs[j,:,1:-1])
            curve[j,i]=curve[i,j]
            #print i,j,curve[i,j]
        deriv[i]=deriv[i]+0.5*np.sum(deriv_filt[:,0]*delt[:,0])
        deriv[i]=deriv[i]+0.5*np.sum(deriv_filt[:,-1]*delt[:,-1])
        deriv[i]=deriv[i]+np.sum(deriv_filt[:,1:-1]*delt[:,1:-1])
    return curve,deriv,chisq
def get_curve_deriv_tod_isosrc(pars,tod,return_vecs=False):
    np_src=4
    np_iso=5
    nsrc=(len(pars)-np_iso)/np_src
    #print 'nsrc is ',nsrc
    fitp_iso=np.zeros(np_iso)
    fitp_iso[:]=pars[:np_iso]
    #print 'fitp_iso is ',fitp_iso
    derivs_iso,f_iso=derivs_from_isobeta_c(fitp_iso,tod)
    derivs_iso_filt=0*derivs_iso
    #tmp=0*tod.info['dat_calib']
    tmp=tod.get_empty(True)
    #nn=tod.info['dat_calib'].size
    nn=tod.get_nsamp
    for i in range(np_iso):
        tmp[:,:]=derivs_iso[i,:,:]
        derivs_iso_filt[i,:,:]=tod.apply_noise(tmp)
    derivs=np.reshape(derivs_iso,[np_iso,nn])
    derivs_filt=np.reshape(derivs_iso_filt,[np_iso,nn])
    pred=f_iso

    for ii in range(nsrc):
        fitp_src=np.zeros(np_src)
        istart=np_iso+ii*np_src
        fitp_src[:]=pars[istart:istart+np_src]
        #print 'fitp_src is ',fitp_src
        derivs_src,f_src=derivs_from_gauss_c(fitp_src,tod)
        pred=pred+f_src
        derivs_src_filt=0*derivs_src
        for i in range(np_src):
            tmp[:,:]=derivs_src[i,:,:]
            derivs_src_filt[i,:,:]=tod.apply_noise(tmp)
        derivs_src_tmp=np.reshape(derivs_src,[np_src,nn])
        derivs=np.append(derivs,derivs_src_tmp,axis=0)
        derivs_src_tmp=np.reshape(derivs_src_filt,[np_src,nn])
        derivs_filt=np.append(derivs_filt,derivs_src_tmp,axis=0)

    #delt_filt=tod.apply_noise(tod.info['dat_calib']-pred)
    delt_filt=tod.apply_noise(tod.get_data()-pred)
    delt_filt=np.reshape(delt_filt,nn)

    #dvec=np.reshape(tod.info['dat_calib'],nn)
    dvec=np.ravel(tod.get_data())
    #predvec=np.reshape(pred,nn)
    predvec=np.ravel(pred)
    delt=dvec-predvec
    



    grad=np.dot(derivs_filt,delt)
    grad2=np.dot(derivs,delt_filt)
    curve=np.dot(derivs_filt,derivs.transpose())
    #return pred
    if return_vecs:
        return grad,grad2,curve,derivs,derivs_filt,delt,delt_filt
    else:
        return grad,grad2,curve

    

def get_timestream_chisq_from_func(func,pars,tods):
    chisq=0.0
    for tod in tods.tods:
        pred,derivs=func(pars,tod)
        #delt=tod.info['dat_calib']-pred
        delt=tod.get_data()-pred
        delt_filt=tod.apply_noise(delt)
        delt_filt[:,0]=delt_filt[:,0]*0.5
        delt_filt[:,-1]=delt_filt[:,-1]*0.5
        chisq=chisq+np.sum(delt*delt_filt)
    return chisq

def get_timestream_chisq_curve_deriv_from_func(func,pars,tods,rotmat=None):
    chisq=0.0
    grad=0.0
    curve=0.0
    #print 'inside func, len(tods) is ',len(tods.tods),len(pars)
    for tod in tods.tods:
        #print 'type of tod is ',type(tod)
        pred,derivs=func(pars,tod)
        if not(rotmat is None):
            derivs=np.dot(rotmat.transpose(),derivs)
        derivs_filt=0*derivs
        #sz=tod.info['dat_calib'].shape
        sz=tod.get_data_dims()
        tmp=np.zeros(sz)
        npp=derivs.shape[0]
        nn=derivs.shape[1]
        #delt=tod.info['dat_calib']-pred
        delt=tod.get_data()-pred
        delt_filt=tod.apply_noise(delt)
        #delt_filt[:,1:-1]=delt_filt[:,1:-1]*2
#<<<<<<< HEAD
#        delt_filt[:,0]=delt_filt[:,0]*0.5
#        delt_filt[:,-1]=delt_filt[:,-1]*0.5
        for i in range(npp):
            tmp[:,:]=np.reshape(derivs[i,:],sz)
            tmp_filt=tod.apply_noise(tmp)
            #tmp_filt[:,1:-1]=tmp_filt[:,1:-1]*2
            tmp_filt[:,0]=tmp_filt[:,0]*0.5
            tmp_filt[:,-1]=tmp_filt[:,-1]*0.5
            derivs_filt[i,:]=np.reshape(tmp_filt,nn)
        delt=np.reshape(delt,nn)
        delt_filt=np.reshape(delt_filt,nn)
        grad1=np.dot(derivs,delt_filt)
        grad2=np.dot(derivs_filt,delt)
        #print 'grad error is ',np.mean(np.abs((grad1-grad2)/(0.5*(np.abs(grad1)+np.abs(grad2)))))
#=======#
#        #delt_filt[:,0]=delt_filt[:,0]*0.5
#        #delt_filt[:,-1]=delt_filt[:,-1]*0.5
#        for i in range(np):
#            tmp[:,:]=np.reshape(derivs[i,:],sz)
#            tmp_filt=tod.apply_noise(tmp)
#            #tmp_filt[:,1:-1]=tmp_filt[:,1:-1]*2
#            #tmp_filt[:,0]=tmp_filt[:,0]*0.5
#            #tmp_filt[:,-1]=tmp_filt[:,-1]*0.5
#            derivs_filt[i,:]=numpy.reshape(tmp_filt,nn)
#        delt=numpy.reshape(delt,nn)
#        delt_filt=numpy.reshape(delt_filt,nn)
#        grad1=numpy.dot(derivs,delt_filt)
#        grad2=numpy.dot(derivs_filt,delt)
#        #print 'grad error is ',numpy.mean(numpy.abs((grad1-grad2)/(0.5*(numpy.abs(grad1)+numpy.abs(grad2)))))
#>>>>>>> bbb21b998a2d0152d5228ea7eb318946309da37b
        grad=grad+0.5*(grad1+grad2)
        curve=curve+np.dot(derivs,derivs_filt.transpose())
        chisq=chisq+np.dot(delt,delt_filt)
    curve=0.5*(curve+curve.transpose())
    return chisq,grad,curve

def get_ts_derivs_many_funcs(tod,pars,npar_fun,funcs,func_args=None,*args,**kwargs):
    #ndet=tod.info['dat_calib'].shape[0]
    #ndat=tod.info['dat_calib'].shape[1]
    ndet=tod.get_ndet()
    ndat=tod.get_ndata()
    npar=np.sum(np.asarray(npar_fun),dtype='int')
    #vals=np.zeros([ndet,ndat])
    
    pred=0
    derivs=np.zeros([npar,ndet,ndat])
    icur=0
    for i in range(len(funcs)):
        tmp=pars[icur:icur+npar_fun[i]].copy()
        myderivs,mypred=funcs[i](tmp,tod,*args,**kwargs)
        pred=pred+mypred
        derivs[icur:icur+npar_fun[i],:,:]=myderivs
        icur=icur+npar_fun[i]
    return derivs,pred
    #derivs,pred=funcs[i](pars,tod)
def get_ts_curve_derivs_many_funcs(todvec,pars,npar_fun,funcs,driver=get_ts_derivs_many_funcs,*args,**kwargs):
    curve=0
    grad=0
    chisq=0
    for tod in todvec.tods:
        derivs,pred=driver(tod,pars,npar_fun,funcs,*args,**kwargs)
        npar=derivs.shape[0]
        ndet=derivs.shape[1]
        ndat=derivs.shape[2]

        #pred_filt=tod.apply_noise(pred)
        derivs_filt=np.empty(derivs.shape)
        for i in range(npar):
            derivs_filt[i,:,:]=tod.apply_noise(derivs[i,:,:])        

        derivs=np.reshape(derivs,[npar,ndet*ndat])
        derivs_filt=np.reshape(derivs_filt,[npar,ndet*ndat])
        #delt=tod.info['dat_calib']-pred
        delt=tod.get_data()-pred
        delt_filt=tod.apply_noise(delt)
        chisq=chisq+np.sum(delt*delt_filt)
        delt=np.reshape(delt,ndet*ndat)
        #delt_filt=np.reshape(delt_filt,[1,ndet*ndat])
        grad=grad+np.dot(derivs_filt,delt.T)
        #grad2=grad2+np.dot(derivs,delt_filt.T)
        curve=curve+np.dot(derivs_filt,derivs.T)
    if have_mpi:
        chisq=comm.allreduce(chisq)
        grad=comm.allreduce(grad)
        curve=comm.allreduce(curve)
    return chisq,grad,curve



def update_lamda(lamda,success):
    if success:
        if lamda<0.2:
            return 0
        else:
            return lamda/np.sqrt(2)
    else:
        if lamda==0.0:
            return 1.0
        else:
            return 2.0*lamda
        
def invscale(mat,do_invsafe=False):
    vec=1/np.sqrt(np.diag(mat))
    mm=np.outer(vec,vec)
    mat=mm*mat
    #ee,vv=np.linalg.eig(mat)
    #print 'rcond is ',ee.max()/ee.min(),vv[:,np.argmin(ee)]
    if do_invsafe:
        return mm*invsafe(mat)
    else:
        return mm*np.linalg.inv(mat)

def _par_step(grad,curve,to_fit,lamda,return_full=False):
    curve_use=curve+lamda*np.diag(np.diag(curve))
    if to_fit is None:
        step=np.dot(invscale(curve_use,True),grad)
        errs=np.sqrt(np.diag(invscale(curve_use,True)))
    else:
        curve_use=curve_use[to_fit,:]
        curve_use=curve_use[:,to_fit]
        grad_use=grad[to_fit]
        step=np.dot(invscale(curve_use),grad_use)
        step_use=np.zeros(len(to_fit))
        step_use[to_fit]=step
        errs_tmp=np.sqrt(np.diag(invscale(curve_use,True)))
        errs=np.zeros(len(to_fit))
        errs[to_fit]=errs_tmp
        step=step_use
    #print('step shape ',step.shape,step)
    if return_full:
        return step,errs
    else:
        return step

def fit_timestreams_with_derivs_manyfun(funcs,pars,npar_fun,tods,to_fit=None,to_scale=None,tol=1e-2,chitol=1e-4,maxiter=10,scale_facs=None,driver=get_ts_derivs_many_funcs):    
    lamda=0
    t1=time.time()
    chisq,grad,curve=get_ts_curve_derivs_many_funcs(tods,pars,npar_fun,funcs,driver=driver)
    t2=time.time()
    if myrank==0:
        print('starting chisq is ',chisq,' with ',t2-t1,' seconds to get curvature')
    for iter in range(maxiter):
        pars_new=pars+_par_step(grad,curve,to_fit,lamda)
        chisq_new,grad_new,curve_new=get_ts_curve_derivs_many_funcs(tods,pars_new,npar_fun,funcs,driver=driver)
        if chisq_new<chisq:
            if myrank==0:
                print('accepting with delta_chisq ',chisq_new-chisq,' and lamda ',lamda,pars_new.shape)
                print(repr(pars_new))
            pars=pars_new
            curve=curve_new
            grad=grad_new
            lamda=update_lamda(lamda,True)
            if (chisq-chisq_new<chitol)&(lamda==0):
                step,errs=_par_step(grad,curve,to_fit,lamda,True)
                return pars,chisq_new,curve_new,errs
            else:
                chisq=chisq_new
        else:
            if myrank==0:
                print('rejecting with delta_chisq ',chisq_new-chisq,' and lamda ',lamda)
            lamda=update_lamda(lamda,False)
        sys.stdout.flush()
    if myrank==0:
        print("fit_timestreams_with_derivs_manyfun failed to converge after ",maxiter," iterations.")    
    step,errs=_par_step(grad,curve,to_fit,lamda,True)
    return pars,chisq,curve,errs
        
def fit_timestreams_with_derivs(func,pars,tods,to_fit=None,to_scale=None,tol=1e-2,chitol=1e-4,maxiter=10,scale_facs=None):
    if not(to_fit is None):
        #print 'working on creating rotmat'
        to_fit=np.asarray(to_fit,dtype='int64')
        inds=np.unique(to_fit)
        nfloat=np.sum(to_fit==1)
        ncovary=np.sum(inds>1)
        nfit=nfloat+ncovary
        rotmat=np.zeros([len(pars),nfit])
        
        solo_inds=np.where(to_fit==1)[0]
        icur=0
        for ind in solo_inds:
            rotmat[ind,icur]=1.0
            icur=icur+1
        if ncovary>0:
            group_inds=inds[inds>1]
            for ind in group_inds:
                ii=np.where(to_fit==ind)[0]
                rotmat[ii,icur]=1.0
                icur=icur+1
    else:
        rotmat=None
        
    iter=0
    converged=False
    pp=pars.copy()
    lamda=0.0
    chi_ref,grad,curve=get_timestream_chisq_curve_deriv_from_func(func,pp,tods,rotmat)
    chi_cur=chi_ref
    iter=0
    while (converged==False) and (iter<maxiter):
        iter=iter+1
        curve_tmp=curve+lamda*np.diag(np.diag(curve))
        #curve_inv=np.linalg.inv(curve_tmp)
        curve_inv=invscale(curve_tmp)
        shifts=np.dot(curve_inv,grad)
        if not(rotmat is None):
            shifts_use=np.dot(rotmat,shifts)
        else:
            shifts_use=shifts
        pp_tmp=pp+shifts_use
        chi_new=get_timestream_chisq_from_func(func,pp_tmp,tods)
        if chi_new<=chi_cur+chitol: #add in a bit of extra tolerance in chi^2 in case we're bopping about the minimum
            success=True
        else:
            success=False
        if success:
            pp=pp_tmp
            chi_cur=chi_new
            chi_tmp,grad,curve=get_timestream_chisq_curve_deriv_from_func(func,pp,tods,rotmat)
        lamda=update_lamda(lamda,success)
        if (lamda==0)&success:
            errs=np.sqrt(np.diag(curve_inv))
            conv_fac=np.max(np.abs(shifts/errs))
            if (conv_fac<tol):
                print('we have converged')
                converged=True
        else:
            conv_fac=None
        to_print=np.asarray([3600*180.0/np.pi,3600*180.0/np.pi,3600*180.0/np.pi,1.0,1.0,3600*180.0/np.pi,3600*180.0/np.pi,3600*180.0/np.pi*np.sqrt(8*np.log(2)),1.0])*(pp-pars)
        print('iter',iter,' max_shift is ',conv_fac,' with lamda ',lamda,chi_ref-chi_cur,chi_ref-chi_new)
    return pp,chi_cur
def _fit_timestreams_with_derivs_old(func,pars,tods,to_fit=None,to_scale=None,tol=1e-2,maxiter=10,scale_facs=None):
    '''Fit a model to timestreams.  func should return the model and the derivatives evaluated at 
    the parameter values in pars.  to_fit says which parameters to float.  0 to fix, 1 to float, and anything
    larger than 1 is expected to vary together (e.g. shifting a TOD pointing mode you could put in a 2 for all RA offsets 
    and a 3 for all dec offsets.).  to_scale will normalize
    by the input value, so one can do things like keep relative fluxes locked together.'''
    

    if not(to_fit is None):
        #print 'working on creating rotmat'
        to_fit=np.asarray(to_fit,dtype='int64')
        inds=np.unique(to_fit)
        nfloat=np.sum(to_fit==1)
        ncovary=np.sum(inds>1)
        nfit=nfloat+ncovary
        rotmat=np.zeros([len(pars),nfit])

        solo_inds=np.where(to_fit==1)[0]
        icur=0
        for ind in solo_inds:
            rotmat[ind,icur]=1.0
            icur=icur+1
        if ncovary>0:
            group_inds=inds[inds>1]
            for ind in group_inds:
                ii=np.where(to_fit==ind)[0]
                rotmat[ii,icur]=1.0
                icur=icur+1

        
    iter=0
    converged=False
    pp=pars.copy()
    while (converged==False) and (iter<maxiter):
        curve=0.0
        grad=0.0
        chisq=0.0
        
        for tod in tods.tods:
            #sz=tod.info['dat_calib'].shape
            sz=tod.get_data_dims()
            pred,derivs=func(pp,tod)
            if not (to_fit is None):
                derivs=np.dot(rotmat.transpose(),derivs)
            derivs_filt=0*derivs
            tmp=np.zeros(sz)
            npp=derivs.shape[0]
            nn=derivs.shape[1]
            #delt=tod.info['dat_calib']-pred
            delt=tod.get_data()-pred
            delt_filt=tod.apply_noise(delt)
            for i in range(npp):
                tmp[:,:]=np.reshape(derivs[i,:],sz)
                tmp_filt=tod.apply_noise(tmp)
                derivs_filt[i,:]=np.reshape(tmp_filt,nn)
            delt=np.reshape(delt,nn)
            delt_filt=np.reshape(delt_filt,nn)
            grad1=np.dot(derivs,delt_filt)
            grad2=np.dot(derivs_filt,delt)
            grad=grad+0.5*(grad1+grad2)
            curve=curve+np.dot(derivs,derivs_filt.transpose())
            chisq=chisq+np.dot(delt,delt_filt)
        if iter==0:
            chi_ref=chisq
        curve=0.5*(curve+curve.transpose())
        curve=curve+2.0*np.diag(np.diag(curve)) #double the diagonal for testing purposes
        curve_inv=np.linalg.inv(curve)
        errs=np.sqrt(np.diag(curve_inv))
        shifts=np.dot(curve_inv,grad)
        #print errs,shifts
        conv_fac=np.max(np.abs(shifts/errs))
        if conv_fac<tol:
            print('We have converged.')
            converged=True
        if not (to_fit is None):
            shifts=np.dot(rotmat,shifts)
        if not(scale_facs is None):
            if iter<len(scale_facs):
                print('rescaling shift by ',scale_facs[iter])
                shifts=shifts*scale_facs[iter]
        to_print=np.asarray([3600*180.0/np.pi,3600*180.0/np.pi,3600*180.0/np.pi,1.0,1.0,3600*180.0/np.pi,3600*180.0/np.pi,3600*180.0/np.pi*np.sqrt(8*np.log(2)),1.0])*(pp-pars)
        print('iter ',iter,' max shift is ',conv_fac,' with chisq improvement ',chi_ref-chisq,to_print) #converged,pp,shifts
        pp=pp+shifts

        iter=iter+1
    return pp,chisq


def split_dict(mydict,vec,thresh):
    #split a dictionary into sub-dictionaries wherever a gap in vec is larger than thresh.
    #useful for e.g. splitting TODs where there's a large time gap due to cuts.
    inds=np.where(np.diff(vec)>thresh)[0]
    #print(inds,len(inds))
    if len(inds)==0:
        return [mydict]
    ndict=len(inds)+1
    inds=np.hstack([[0],inds+1,[len(vec)]])
    #print(inds)

    out=[None]*ndict
    for i in range(ndict):
        out[i]={}
    for key in mydict.keys():
        tmp=mydict[key]
        for i in range(ndict):
            out[i][key]=tmp
        try:
            dims=tmp.shape
            ndim=len(dims)
            if ndim==1:
                if dims[0]==len(vec):
                    for i in range(ndict):
                        out[i][key]=tmp[inds[i]:inds[i+1]].copy()
            if ndim==2:
                if dims[1]==len(vec):
                    for i in range(ndict):
                        out[i][key]=tmp[:,inds[i]:inds[i+1]].copy()
                elif dims[0]==len(vec):
                    for i in range(ndict):
                        out[i][key]=tmp[inds[i]:inds[i+1],:].copy()
        except:
            continue
            #print('copying ',key,' unchanged')
            #don't need below as it's already copied by default
            #for i in range(ndict):
            #    out[i][key]=mydict[key]

    return out

def mask_dict(mydict,mask):
    for key in mydict.keys():
        tmp=mydict[key]
        try:
            dims=tmp.shape
            ndim=len(dims)
            if ndim==1:
                if dims[0]==len(mask):
                    tmp=tmp[mask]
                    mydict[key]=tmp
            if ndim==2:
                if dims[0]==len(mask):
                    tmp=tmp[mask,:]
                if dims[1]==len(mask):
                    tmp=tmp[:,mask]
                mydict[key]=tmp
            if ndim==3:
                if dims[0]==len(mask):
                    tmp=tmp[mask,:,:]
                if dims[1]==len(mask):
                    tmp=tmp[:,mask,:]
                if dims[2]==len(maks):
                    tmp=tmp[:,:,mask]
                mydict[key]=tmp
        except:
            continue
