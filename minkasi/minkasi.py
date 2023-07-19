import os
import numpy as np
import ctypes
import time
from . import mkfftw
#import pyfits
from astropy.io import fits as pyfits
import astropy
from astropy import wcs
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo #choose your cosmology here
import scipy
import copy
import sys
from numba import jit

try:
    import healpy
    have_healpy=True
except:
    have_healpy=False
try: 
    import numba as nb
    from . import minkasi_nb
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


try:
    mylib=ctypes.cdll.LoadLibrary("libminkasi.so")
except OSError:
    mylib=ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libminkasi.so"))

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

    return to_return   


def set_nthread(nthread):
    set_nthread_c(nthread)

def get_nthread():
    nthread=np.zeros([1,1],dtype='int32')
    get_nthread_c(nthread.ctypes.data)
    return nthread[0,0]


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
    ix=int(map.nx/2)
    iy=int(map.ny/2)
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
    int_max=(vol/2**npr)*np.prod(r/lp)+30 #add a bit just to make sure we don't act up for small n
    #print 'int max is ',int max
    int_max=int(int_max)

    #vals=np.zeros(int_max,dtype='int')
    vals=np.zeros(int_max)
    icur=0
    icur=_prime_loop(r,lp,icur,0.0,vals)
    assert(icur<=int_max)
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
