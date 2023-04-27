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

print('importing mpi4py')

try:
    import mpi4py.rc
    mpi4py.rc.threads = False
    from mpi4py import MPI
    print('mpi4py imported')
    comm=MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nproc=comm.Get_size()
    print('nproc:, ', nproc)
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

tod2map_atomic_c=mylib.tod2map_atomic
tod2map_atomic_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

#tod2map_everyone_c=mylib.tod2map_everyone
#tod2map_everyone_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]

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

outer_c=mylib.outer_block
outer_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]



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

def tod2map_everyone(map,dat,ipix,edges):
    assert(len(edges)==get_nthread()+1)
    tod2map_everyone_c(map.ctypes.data,dat.ctypes.data,dat.shape[0],dat.shape[1],ipix.ctypes.data,map.size,edges.ctypes.data,len(edges))

def tod2map_omp(map,dat,ipix,atomic=False):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    if not(ipix.dtype=='int32'):
        print("Warning - ipix is not int32 in tod2map_omp.  this is likely to produce garbage results.")
    if atomic:
        tod2map_atomic_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data,map.size)
    else:
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

def run_pcg(b,x0,tods,precon=None,maxiter=25,outroot='map',save_iters=[-1],save_ind=0,save_tail='.fits',plot_iters=[],plot_info=None,plot_ind=0):
    
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
    nsamp=tods.get_nsamp()
    tloop=time.time()
    for iter in range(maxiter):
        if myrank==0:
            if iter>0:
                print(iter,zr,alpha,t2-t1,t3-t2,t3-t1,nsamp/(t2-t1)/1e6)
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
        if iter in plot_iters:
            print('plotting on iteration ',iter)
            x.maps[plot_ind].plot(plot_info)

    tave=(time.time()-tloop)/maxiter
    print('average time per iteration was ',tave,' with effective throughput ',nsamp/tave/1e6,' Msamp/s')
    if iter in plot_iters:
        print('plotting on iteration ',iter)
        x.maps[plot_ind].plot(plot_info)
    else:
        print('skipping plotting on iter ',iter)
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

class tsNotch(tsGeneric):
    def __init__(self,tod,numin,numax):
        self.fname=tod.info['fname']
        tvec=tod.get_tvec()
        dt=tvec[-1]-tvec[0]
        bw=numax-numin
        dnu=1/dt
        nfreq=np.int(np.ceil(2*bw/dnu)) #factor of 2 is to account for partial waves
        ndet=tod.get_ndet()
        self.freqs=np.linspace(numin,numax,nfreq)
        self.nfreq=nfreq
        self.params=np.zeros([2*nfreq,ndet])
    def get_vecs(self,tvec):
        tvec=tvec-tvec[0]
        vecs=np.zeros([self.nfreq*2,len(tvec)])
        for i in range(self.nfreq):
            vecs[2*i,:]=np.cos(tvec*self.freqs[i])
            vecs[2*i+1,:]=np.sin(tvec*self.freqs[i])
        return vecs
        
    def map2tod(self,tod,mat=None,do_add=True,do_omp=False):
        tvec=tod.get_tvec()
        vecs=self.get_vecs(tvec)
        pred=self.params.T@vecs
        if mat is None:
            mat=tod.get_data()
        if do_add:
            mat[:]=mat[:]+pred
        else:
            mat[:]=pred
    def tod2map(self,tod,mat=None,do_add=True,do_omp=False):
        tvec=tod.get_tvec()
        vecs=self.get_vecs(tvec)
        if mat is None:
            mat=tod.get_data()
        #tmp=mat@(vecs.T)
        tmp=vecs@mat.T
        if do_add:
            self.params[:]=self.params[:]+tmp
        else:
            self.params[:]=tmp


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
    def __init__(self,tod,lims=[0,2*np.pi],nbin=360):
        #print('nbin is',nbin)
        ndet=tod.get_ndet()
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
        minkasi_nb.tod2map_binned_det(dat,self.params,tod.info['az'],self.lims,self.nbin,do_add)

class tsBinnedAzShared(tsGeneric):
#"""class to have az shared amongst TODs (say, if you think the ground is constant for a while)"""
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
        print('nbin is',self.nbin)
        print(self.params.dtype)
        minkasi_nb.tod2map_binned_det(dat,self.params,tod.info['az'],self.lims,self.nbin,do_add)
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
    def __init__(self,lims,pixsize=0,proj='CAR',pad=2,primes=None,cosdec=None,nx=None,ny=None,mywcs=None,tag='ipix',purge_pixellization=False,ref_equ=False):
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
        self.tag=tag
        self.purge_pixellization=purge_pixellization
        self.caches=None
        self.cosdec=cosdec
        self.tod2map_method=None
    def get_caches(self):
        npix=self.nx*self.ny
        nthread=get_nthread()
        self.caches=np.zeros([nthread,npix])
    def clear_caches(self):
        self.map[:]=np.reshape(np.sum(self.caches,axis=0),self.map.shape)
        self.caches=None
    def set_tod2map(self,method=None,todvec=None):
        """Select which method of tod2map to use.  options include simple (1 proc), omp (everyone makes a map copy), everyone (everyone loops through
           all the data but assigns only to their own piece), atomic (no map copy, accumulate via atomic adds), and cached (every thread has a sticky
           copy of the map)."""
        if method is None:
            if nproc==1:
                self.tod2map_method=self.tod2map_simple
            else:
                self.tod2map_method=self.tod2map_omp
            return
        if method=='omp':
            self.tod2map_method=self.tod2map_omp
            return
        if method=='simple':
            self.tod2map_method=self.tod2map_simple
        if method=='everyone':
            if todvec is None:
                print('need tods when setting to everyone so we can know which pieces belong to which threads')
            for tod in todvec.tods:
                ipix=self.get_pix(tod,False)
                ipix=ipix.copy()
                ipix=np.ravel(ipix)
                ipix.sort()
                inds=len(ipix)*np.arange(nproc+1)//nproc
                inds=np.asarray(inds,dtype='int32')
                tod.save_pixellization(self.tag+'_edges',inds)
                self.tod2map_method=self.tod2map_everyone
        if method=='cached':
            self.get_caches()
            self.tod2map_method=self.tod2map_cached
        if method=='atomic':
            self.tod2map_method=self.todmap_atomic
    def tod2map_atomic(self,tod,dat):
        ipix=self.get_pix(tod)
        tod2map_omp(self.map,dat,ipix,True)
    def todmap_omp(self,tod,dat):
        ipix=self.get_pix(tod)
        tod2map_omp(self.map,dat,ipix,False)
    def tod2map_simple(self,tod,dat):
        ipix=self.get_pix(tod)
        tod2map_simple(self.map,dat,ipix)
    #def tod2map_cached(self.map,dat,ipix):
    #    ipix=self.get_pix(tod)
    #    tod2map_cached(map,dat,ipix)

    def copy(self):
        if False:
            newmap=SkyMap(self.lims,self.pixsize,self.proj,self.pad,self.primes,cosdec=self.cosdec,nx=self.nx,ny=self.ny,mywcs=self.wcs,tag=self.tag)
            newmap.map[:]=self.map[:]
            return newmap
        else:
            return copy.deepcopy(self)
    def clear(self):
        self.map[:]=0
    def axpy(self,map,a):
        self.map[:]=self.map[:]+a*map.map[:]
    def assign(self,arr):
        assert(arr.shape[0]==self.nx)
        assert(arr.shape[1]==self.ny)
        #self.map[:,:]=arr
        self.map[:]=arr
    def pix_from_radec(self,ra,dec):
        ndet=ra.shape[0]
        nsamp=ra.shape[1]
        nn=ndet*nsamp
        coords=np.zeros([nn,2])
        #coords[:,0]=np.reshape(tod.info['dx']*180/np.pi,nn)
        #coords[:,1]=np.reshape(tod.info['dy']*180/np.pi,nn)
        coords[:,0]=np.reshape(ra*180/np.pi,nn)
        coords[:,1]=np.reshape(dec*180/np.pi,nn)


        #print coords.shape
        pix=self.wcs.wcs_world2pix(coords,1)
        #print pix.shape
        xpix=np.reshape(pix[:,0],[ndet,nsamp])-1  #-1 is to go between unit offset in FITS and zero offset in python
        ypix=np.reshape(pix[:,1],[ndet,nsamp])-1  
        xpix=np.round(xpix)
        ypix=np.round(ypix)
        ipix=np.asarray(xpix*self.ny+ypix,dtype='int32')
        return ipix

    def get_pix(self,tod,savepix=True):
        if not(self.tag is None):
            ipix=tod.get_saved_pix(self.tag)
            if not(ipix is None):
                return ipix
        ra,dec=tod.get_radec()
        #ndet=tod.info['dx'].shape[0]
        #nsamp=tod.info['dx'].shape[1]
        if False:
            ndet=ra.shape[0]
            nsamp=ra.shape[1]
            nn=ndet*nsamp
            coords=np.zeros([nn,2])
        #coords[:,0]=np.reshape(tod.info['dx']*180/np.pi,nn)
        #coords[:,1]=np.reshape(tod.info['dy']*180/np.pi,nn)
            coords[:,0]=np.reshape(ra*180/np.pi,nn)
            coords[:,1]=np.reshape(dec*180/np.pi,nn)


        #print coords.shape
            pix=self.wcs.wcs_world2pix(coords,1)
        #print pix.shape
            xpix=np.reshape(pix[:,0],[ndet,nsamp])-1  #-1 is to go between unit offset in FITS and zero offset in python
            ypix=np.reshape(pix[:,1],[ndet,nsamp])-1  
            xpix=np.round(xpix)
            ypix=np.round(ypix)
            ipix=np.asarray(xpix*self.ny+ypix,dtype='int32')
        else:
            ipix=self.pix_from_radec(ra,dec)
        if savepix:
            if not(self.tag is None):
                tod.save_pixellization(self.tag,ipix)
        return ipix
    def map2tod(self,tod,dat,do_add=True,do_omp=True):
        ipix=self.get_pix(tod)
        #map2tod(dat,self.map,tod.info['ipix'],do_add,do_omp)
        map2tod(dat,self.map,ipix,do_add,do_omp)

    def tod2map(self,tod,dat=None,do_add=True,do_omp=True):        
        if dat is None:
            dat=tod.get_data()
        if do_add==False:
            self.clear()
        ipix=self.get_pix(tod)
        
        if not(self.caches is None):
            #tod2map_cached(self.caches,dat,tod.info['ipix'])
            tod2map_cached(self.caches,dat,ipix)
        else:
            if do_omp:
                #tod2map_omp(self.map,dat,tod.info['ipix'])
                tod2map_omp(self.map,dat,ipix)
            else:
                #tod2map_simple(self.map,dat,tod.info['ipix'])
                tod2map_simple(self.map,dat,ipix)
        if self.purge_pixellization:
            tod.clear_saved_pix(self.tag)

    def tod2map_old(self,tod,dat=None,do_add=True,do_omp=True):
        if dat is None:
            dat=tod.get_data()
        if do_add==False:
            self.clear()
        ipix=self.get_pix(tod)
        if not(self.caches is None):
            #tod2map_cached(self.caches,dat,tod.info['ipix'])
            tod2map_cached(self.caches,dat,ipix)
        else:
            if do_omp:
                #tod2map_omp(self.map,dat,tod.info['ipix'])
                tod2map_omp(self.map,dat,ipix)
            else:
                #tod2map_simple(self.map,dat,tod.info['ipix'])
                tod2map_simple(self.map,dat,ipix)
        if self.purge_pixellization:
            tod.clear_saved_pix(self.tag)

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
        
    def plot(self,plot_info=None):
        vmin=self.map.min()
        vmax=self.map.max()
        clf=True
        pause=True
        pause_len=0.001
        if not(plot_info is None):
            if 'vmin' in plot_info.keys():
                vmin=plot_info['vmin']
            if 'vmax' in plot_info.keys():
                vmax=plot_info['vmax']
            if 'clf' in plot_info.keys():
                clf=plot_info['clf']
            if 'pause' in plot_info.keys():
                pause=plot_info['pause']
            if pause_len in plot_info.keys():
                pause_len=plot_info['pause_len']
        from matplotlib import pyplot as plt
        if clf:
            plt.clf()
        plt.imshow(self.map,vmin=vmin,vmax=vmax)
        if pause:
            plt.pause(pause_len)

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
                inds
