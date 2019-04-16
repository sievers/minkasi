import numpy as np
import ctypes
import time
import pyfftw
import pyfits
import astropy
from astropy import wcs
from astropy.io import fits

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




def report_mpi():
    if have_mpi:
        print 'myrank is ',myrank,' out of ',nproc
    else:
        print 'mpi not found'

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
    tod2map_simple_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data)

def tod2map_omp(map,dat,ipix):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    tod2map_omp_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data,map.size)

def tod2map_cached(map,dat,ipix):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    tod2map_cached_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data,map.shape[1])
    

def map2tod(dat,map,ipix,do_add=False,do_omp=True):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    if do_omp:
        map2tod_omp_c(dat.ctypes.data, map.ctypes.data, ndet, ndata, ipix.ctypes.data, do_add)
    else:
        map2tod_simple_c(dat.ctypes.data,map.ctypes.data,ndet,ndata,ipix.ctypes.data,do_add)
    

def set_nthread(nthread):
    set_nthread_c(nthread)

def get_nthread():
    nthread=np.zeros([1,1],dtype='int32')
    get_nthread_c(nthread.ctypes.data)
    return nthread[0,0]


def cut_blacklist(tod_names,blacklist):
    mydict={}
    for nm in tod_names:
        tt=nm.split('/')[-1]
        mydict[tt]=nm
    ncut=0
    for nm in blacklist:
        tt=nm.split('/')[-1]
        if mydict.has_key(tt):
            ncut=ncut+1
            del(mydict[tt])
    if ncut>0:
        print 'deleted ',ncut,' bad files.'
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

def make_rings(edges,cent,map,pixsize=2.0,fwhm=10.0,amps=None):
    xvec=np.arange(map.nx)
    yvec=np.arange(map.ny)
    xvec[map.nx/2:]=xvec[map.nx/2:]-map.nx
    yvec[map.ny/2:]=yvec[map.ny/2:]-map.ny

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
        print 'beam_area is ',beam_area*1e9,' nsr'
    nring=len(edges)-1
    rings=np.zeros([nring,map.nx,map.ny])
    mypix=map.wcs.wcs_world2pix(cent[0],cent[1],1)
    print 'mypix is ',mypix

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

    print dat_filt.shape
    dat_filt[:,0:pad*width]=0
    dat_filt[:,-pad*width:]=0
    det_thresh=thresh*np.median(np.abs(dat_filt),axis=1)
    dat_dejump=dat.copy()
    jumps=[None]*ndet
    print 'have filtered data, now searching for jumps'
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
            
            
            print 'found jump on detector ',i,' at sample ',ind
            if np.abs(val2/val)>rat:
                print 'I think this is a spike due to ratio ',np.abs(val2/val)
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
            print 'working on detector ',i,' who has ', len(jumps[i]),' jumps with segments ',segs
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
    print 'Unsupported nbyte ' + repr(nbyte) + ' in get_type'
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
            print 'bad facs came from ' + repr([2**lcur,2**ln,2**lp[0]])
        #print icur
        return icur
    else:
        facs=np.arange(lcur,ln,lp[0])
        for fac in facs:
            icur=_prime_loop(ln,lp[1:],icur,fac,vals)
        return icur
    print 'I don''t think I should have gotten here.'
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
    to_conv_ft=pyfftw.fft_r2r(to_conv)
    xtrans=pyfftw.fft_r2r(spec)
    for i in range(nspec):
        xtrans[i,:]=xtrans[i,:]*to_conv_ft
    #return pyfftw.fft_r2r(xtrans)/(2*(xtrans.shape[1]-1)),to_conv
    return xtrans,to_conv_ft
def smooth_many_vecs(vecs,fwhm=20):
    n=vecs.shape[1]
    nvec=vecs.shape[0]
    x=np.arange(n)
    sig=fwhm/np.sqrt(8*np.log(2))
    to_conv=np.exp(-0.5*(x/sig)**2)
    tot=to_conv[0]+to_conv[-1]+2*to_conv[1:-1].sum() #r2r normalization
    to_conv=to_conv/tot
    to_conv_ft=pyfftw.fft_r2r(to_conv)
    xtrans=pyfftw.fft_r2r(vecs)
    for i in range(nvec):
        xtrans[i,:]=xtrans[i,:]*to_conv_ft
    back=pyfftw.fft_r2r(xtrans)
    return back/(2*(n-1))
def smooth_vec(vec,fwhm=20):
    n=vec.size
    x=np.arange(n)
    sig=fwhm/np.sqrt(8*np.log(2))
    to_conv=np.exp(-0.5*(x/sig)**2)
    tot=to_conv[0]+to_conv[-1]+2*to_conv[1:-1].sum() #r2r normalization
    to_conv=to_conv/tot
    to_conv_ft=pyfftw.fft_r2r(to_conv)
    xtrans=pyfftw.fft_r2r(vec)
    back=pyfftw.fft_r2r(xtrans*to_conv_ft)
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
        print iter,zr
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

def run_pcg(b,x0,tods,precon=None,maxiter=25):
    t1=time.time()
    Ax=tods.dot(x0)

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
                print iter,zr,alpha,t2-t1,t3-t2,t3-t1
            else:
                print iter,zr,t2-t1
        t1=time.time()
        Ap=tods.dot(p)
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

def run_pcg_wprior(b,x0,tods,prior,precon=None,maxiter=25):
    t1=time.time()
    Ax=tods.dot(x0)
    #prior.apply_prior(Ax,x0)
    flub=prior.apply_prior(x0.maps[0].map)
    print 'means of flub and Ax are ',np.mean(np.abs(Ax.maps[0].map)),np.mean(np.abs(flub))
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
                print iter,zr,alpha,t2-t1,t3-t2,t3-t1
            else:
                print iter,zr,t2-t1
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
        dat=tod['dat_calib']
    dat_rot=np.dot(tod['v'],dat)
    datft=pyfftw.fft_r2r(dat_rot)
    nn=datft.shape[1]
    datft=datft*tod['mywt'][:,0:nn]
    dat_rot=pyfftw.fft_r2r(datft)
    dat=np.dot(tod['v'].transpose(),dat_rot)
    return dat



class null_precon:
    def __init__(self):
        self.isnull=True
    def __add__(self,val):
        return val
    def __mul__(self,val):
        return val

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
        mm=self.copy()
        for i in range(self.nmap):
            mm.maps[i]=self.maps[i]*mapset.maps[i]
        return mm
    def get_caches(self):
        for i in range(self.nmap):
            self.maps[i].get_caches()
    def clear_caches(self):
        for i in range(self.nmap):
            self.maps[i].clear_caches()

    def mpi_reduce(self):
        if have_mpi:
            for map in self.maps:
                map.mpi_reduce()
class Cuts:
    def __init__(self,tod):
        self.tag=tod.info['tag']
        self.ndet=tod.info['dat_calib'].shape[0]
        self.cuts=[None]*self.ndet

class CutsVec:
    def __init__(self,todvec):
        self.ntod=todvec.ntod
        self.cuts=[None]*self.ntod
        for tod in todvec.tods:
            self.cuts[tod.info['tag']]=Cuts(tod)
            
class SkyMap:
    def __init__(self,lims,pixsize,proj='CAR',pad=2,primes=None,cosdec=None,nx=None,ny=None,mywcs=None,ref_equ=False):
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
            print 'corners seem to have gone negative in SkyMap projection.  not good, you may want to check this.'
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
        self.map[:,:]=arr
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
        new_map=self.copy()
        new_map.map[:]=self.map[:]*map.map[:]
        return new_map
    def mpi_reduce(self):
        if have_mpi:
            self.map=comm.allreduce(self.map)
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
        dims=tod.info['dat_calib'].shape
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
    
class CutsVecs:
    def __init__(self,todvec,do_add=True):
        #if class(todvec)==CutsVecs: #for use in copy
        if isinstance(todvec,CutsVecs):
            self.cuts=[None]*todvec.ntod
            self.ntod=todvec.ntod
            for i in range(todvec.ntod):
                self.cuts[i]=todvec.cuts[i].copy()
            return
        #if class(todvec)!=TodVec:
        if not(isinstance(todvec,TodVec)):
            print 'error in CutsVecs init, must pass in a todvec class.'
            return None
        self.cuts=[None]*todvec.ntod
        for i in range(todvec.ntod):
            tod=todvec.tods[i]
            if tod.info['tag']!=i:
                print 'warning, tag mismatch in CutsVecs.__init__'
                print 'continuing, but you should be careful...'
            if iskey(tod.info['bad_samples']):
                self.cuts[i]=Cuts(tod,do_add)
    def copy(self):
        return CutsVecs(self)
    def clear(self):
        for cuts in self.cuts:
            cuts.clear()
    def axpy(self,cutsvec,a):
        assert(self.ntod==cutsvec.ntod)
        for i in range(ntod):
            self.cuts[i].axpy(cutsvec.cuts[i],a)
    def map2tod(self,todvec):
        assert(self.ntod==todvec.ntod)
        for i in range(self.ntod):
            self.cuts[i].map2tod(todvec.tods[i])
    def tod2map(self,todvec,dat):
        assert(self.ntod==todvec.ntod)
        assert(self.ntod==dat.ntod)
        for i in range(self.ntod):
            self.cuts[i].tod2map(todvec.tods[i],dat.tods[i])
    def dot(self,cutsvec):
        tot=0.0
        assert(self.ntod==cutsvec.ntod)
        for i in range(self.ntod):
            tot+=self.cuts[i].dot(cutsvec.cuts[i])
        return tot
        
                                 
            
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
        pred=np.zeros(tod.info['dat_calib'].shape)
    n=tod.info['dat_calib'].size
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
        pred=np.zeros(tod.info['dat_calib'].shape)
    n=tod.info['dat_calib'].size
    assert(pred.size==n)
    dx=tod.info['dx']
    dy=tod.info['dy']
    fill_isobeta_c(params.ctypes.data,dx.ctypes.data,dy.ctypes.data,pred.ctypes.data,n)

    npar_beta=5 #x,y,theta,beta,amp
    npar_src=4 #x,y,sig,amp
    nsrc=(params.size-npar_beta)/npar_src
    for i in range(nsrc):
        pp=np.zeros(npar_src)
        ioff=i*npar_src+npar_beta
        pp[:]=params[ioff:(ioff+npar_src)]
        fill_gauss_src_c(pp.ctypes.data,dx.ctypes.data,dy.ctypes.data,pred.ctypes.data,n)


    return pred

def derivs_from_isobeta_c(params,tod):
    npar=5;
    n=tod.info['dat_calib'].size
    sz_deriv=np.append(npar,tod.info['dat_calib'].shape)
    
    pred=np.zeros(tod.info['dat_calib'].shape)
    derivs=np.zeros(sz_deriv)

    dx=tod.info['dx']
    dy=tod.info['dy']
    fill_isobeta_derivs_c(params.ctypes.data,dx.ctypes.data,dy.ctypes.data,pred.ctypes.data,derivs.ctypes.data,n)

    return derivs,pred

def derivs_from_gauss_c(params,tod):
    npar=4
    n=tod.info['dat_calib'].size
    sz_deriv=np.append(npar,tod.info['dat_calib'].shape)
    
    pred=np.zeros(tod.info['dat_calib'].shape)
    derivs=np.zeros(sz_deriv)

    dx=tod.info['dx']
    dy=tod.info['dy']
    fill_gauss_derivs_c(params.ctypes.data,dx.ctypes.data,dy.ctypes.data,pred.ctypes.data,derivs.ctypes.data,n)

    return derivs,pred

def timestreams_from_isobeta(params,tod):
    npar_beta=5 #x,y,theta,beta,amp
    npar_src=4 #x,y,sig,amp
    nsrc=(params.size-npar_beta)/npar_src
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
        chisq=chisq+tod.timestream_chisq(tod.info['dat_calib']-pred)

    return chisq
    npar_beta=5 #x,y,theta,beta,amp
    npar_src=4 #x,y,sig,amp
    nsrc=(params.size-npar_beta)/npar_src
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
        chisq=chisq+tod.timestream_chisq(tod.info['dat_calib']-pred)
    return chisq

class Tod:
    def __init__(self,info):
        self.info=info.copy()
        self.jumps=None
        self.cuts=None
    def lims(self):
        xmin=self.info['dx'].min()
        xmax=self.info['dx'].max()
        ymin=self.info['dy'].min()
        ymax=self.info['dy'].max()
        return xmin,xmax,ymin,ymax
    def set_tag(self,tag):
        self.info['tag']=tag
    def copy(self):
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
            
        return tod
    def set_noise_cm_white(self):
        u,s,v=np.linalg.svd(self.info['dat_calib'],0)
        ndet=len(s)
        ind=np.argmax(s)
        mode=np.zeros(ndet)
        mode[:]=u[:,0]
        pred=np.outer(mode,v[0,:])        
        dat_clean=self.info['dat_calib']-pred
        myvar=np.std(dat_clean,1)**2
        self.info['v']=mode
        self.info['mywt']=1.0/myvar
        self.info['noise']='cm_white'
        
    def apply_noise_cm_white(self,dat=None):
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
        
    def set_noise_smoothed_svd(self,fwhm=50,func=None,pars=None):
        '''If func comes in as not empy, assume we can call func(pars,tod) to get a predicted model for the tod that
        we subtract off before estimating the noise.'''
        if func is None:
            u,s,v=np.linalg.svd(self.info['dat_calib'],0)
        else:
            tmp=func(pars,self)
            u,s,v=np.linalg.svd(self.info['dat_calib']-tmp,0)
        print 'got svd'
        ndet=s.size
        n=self.info['dat_calib'].shape[1]
        self.info['v']=np.zeros([ndet,ndet])
        self.info['v'][:]=u.transpose()
        dat_rot=np.dot(self.info['v'],self.info['dat_calib'])
        dat_trans=pyfftw.fft_r2r(dat_rot)
        spec_smooth=smooth_many_vecs(dat_trans**2,fwhm)
        self.info['mywt']=1.0/spec_smooth
        self.info['noise']='smoothed_svd'
        #return dat_rot
        
    def apply_noise(self,dat=None):
        if dat is None:
            dat=self.info['dat_calib']
        if self.info['noise']=='cm_white':
            return self.apply_noise_cm_white(dat)
        dat_rot=np.dot(self.info['v'],dat)
        datft=pyfftw.fft_r2r(dat_rot)
        nn=datft.shape[1]
        datft=datft*self.info['mywt'][:,0:nn]
        dat_rot=pyfftw.fft_r2r(datft)
        dat=np.dot(self.info['v'].transpose(),dat_rot)
        return dat
    def dot(self,mapset,mapset_out):
        tmp=0.0*self.info['dat_calib']
        for map in mapset.maps:
            map.map2tod(self,tmp)
        tmp=self.apply_noise(tmp)
        for map in mapset_out.maps:
            map.tod2map(self,tmp)
    def set_jumps(self,jumps):
        self.jumps=jumps
    def cut_detectors(self,isgood):
        #cut all detectors not in boolean array isgood
        isbad=np.asarray(1-isgood,dtype='bool')
        bad_inds=np.where(isbad)
        bad_inds=np.fliplr(bad_inds)
        bad_inds=bad_inds[0]
        print bad_inds
        nkeep=np.sum(isgood)
        for key in self.info.keys():
            if isinstance(self.info[key],np.ndarray):
                self.info[key]=slice_with_copy(self.info[key],isgood)
        if not(self.jumps is None):
            for i in bad_inds:
                print 'i in bad_inds is ',i
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

def slice_with_copy(arr,ind):
    if isinstance(arr,np.ndarray):
        myshape=arr.shape

        if len(myshape)==1:
            ans=np.zeros(ind.sum(),dtype=arr.dtype)
            print ans.shape
            print ind.sum()
            ans[:]=arr[ind]
        else:   
            mydims=np.append(np.sum(ind),myshape[1:])
            print mydims,mydims.dtype
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
            print 'before reduction lims are ',[xmin,xmax,ymin,ymax]
            xmin=comm.allreduce(xmin,op=MPI.MIN)
            xmax=comm.allreduce(xmax,op=MPI.MAX)
            ymin=comm.allreduce(ymin,op=MPI.MIN)
            ymax=comm.allreduce(ymax,op=MPI.MAX)
            print 'after reduction lims are ',[xmin,xmax,ymin,ymax]
        return [xmin,xmax,ymin,ymax]
    def set_pix(self,map):
        for tod in self.tods:
            ipix=map.get_pix(tod)
            tod.info['ipix']=ipix
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

    def dot(self,mapset,mapset2=None,report_times=False,cache_maps=True):
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
def read_tod_from_fits(fname,hdu=1,branch=None):
    f=pyfits.open(fname)
    raw=f[hdu].data
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
        print 'nsamp and ndet are ',ndet,nsamp,len(pixid),' on ',fname, 'with lims ',xmin,xmax,ymin,ymax
    else:
        print 'nsamp and ndet are ',ndet,nsamp,len(pixid),' on ',fname
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
    #dat['dat_calib']=np.zeros([ndet,nsamp],dtype=type(dat_calib[0]))
    dat['dat_calib']=np.zeros([ndet,nsamp],dtype='float64') #go to double because why not
    dat_calib=np.reshape(dat_calib,[ndet,nsamp])
    dat['dat_calib'][:]=dat_calib[:]
    dat['fname']=fname
    f.close()
    return dat


def downsample_array_r2r(arr,fac):

    n=arr.shape[1]
    nn=int(n/fac)
    arr_ft=pyfftw.fft_r2r(arr)
    arr_ft=arr_ft[:,0:nn].copy()
    arr=pyfftw.fft_r2r(arr_ft)/(2*(n-1))
    return arr

def downsample_tod(dat,fac=10):
    ndata=dat['dat_calib'].shape[1]
    keys=dat.keys()
    for key in dat.keys():
        try:
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
        print 'truncating from ',n,' to ',n_new
        for key in dat.keys():
            try:
                if dat[key].shape[1]==n:
                    dat[key]=dat[key][:,0:n_new].copy()
            except:
                #print 'skipping key ' + key
                pass


def todvec_from_files_octave(fnames):
    todvec=TodVec()
    for fname in fnames:
        info=read_octave_struct(fname)
        tod=Tod(info)
        todvec.add_tod(tod)
    return todvec
        
def make_hits(todvec,map):
    hits=map.copy()
    hits.clear()
    for tod in todvec.tods:
        tmp=np.ones(tod.info['dat_calib'].shape)
        hits.tod2map(tod,tmp)
    if have_mpi:
        hits.mpi_reduce()
    return hits


def decimate(vec,nrep=1):
    for i in range(nrep):
        if len(vec)%2:
            vec=vec[:-1]
        vec=0.5*(vec[0::2]+vec[1::2])
    return vec
def plot_ps(vec,downsamp=0):
    vecft=pyfftw.fft_r2r(vec)
    
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
            w.wcs.crpix=[lims[1]/pixsize,-lims[2]/pixsize]
            #print 'crpix is ',w.wcs.crpix
        else:
            w.wcs.crpix=[1.0,1.0]
            w.wcs.crval=[lims[1]*180/np.pi,lims[2]*180/np.pi]
        w.wcs.cdelt=[-pixsize/cosdec*180/np.pi,pixsize*180/np.pi]
        w.wcs.ctype=['RA---CAR','DEC--CAR']
        return w
    print 'unknown projection type ',proj,' in get_wcs.'
    return None




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
            print 'successful convergence after ',iter,' iterations with error estimate ',np.max(np.abs(frac_shift))
            converged=True
        if iter==max_iter:
            print 'not converging after ',iter,' iterations in fit_linear_ps_uncorr with current convergence parameter ',np.max(np.abs(frac_shift))
            converged=True

    return fitp
def fit_ts_ps(dat,dt=1.0,ind=-2.0,nu_min=0.0,nu_max=np.inf):
    datft=pyfftw.fft_r2r(dat)
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

    rat=vecs[1,:]*guess[1]/(vecs[0,:]*guess[0])
    my_ind=np.max(np.where(rat>1)[0])
    nu_ref=np.sqrt(nu[my_ind]*nu[0]) #WAG as to a sensible frequency pivot point
    #nu_ref=0.2*nu[my_ind] #WAG as to a sensible frequency pivot point
    print 'knee is roughly at ',nu[my_ind],nu_ref

    #model = guess[1]*nu^ind+guess[0]
    #      = guess[1]*(nu/nu_ref*nu_ref)^ind+guess[0]
    #      = guess[1]*(nu_ref)^in*(nu/nu_ref)^ind+guess[0]

    nu_scale=nu/nu_ref
    guess_scale=guess.copy()
    guess_scale[1]=guess[1]*(nu_ref**ind)
    print 'guess is ',guess
    print 'guess_scale is ',guess_scale
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
    for iter in range(20):
        vec=nu_scale**fitp[2]
        C=fitp[0]+fitp[1]*vec
        Cinv=1.0/C
        vecs[1,:]=vec
        vecs[2,:]=fitp[1]*lognu*vec
        for i in range(npp):
            grad_chi[i]=0.5*np.sum(datsqr*vecs[i,:]*Cinv*Cinv)
            grad_tr[i]=-0.5*np.sum(vecs[i,:]*Cinv)
            for j in range(i,npp):
                curve[i,j]=0.5*np.sum(Cinv*Cinv*vecs[i,:]*vecs[j,:])-np.sum(datsqr*Cinv*Cinv*Cinv*vecs[i,:]*vecs[j,:])
                curve[j,i]=curve[i,j]
        grad=grad_chi+grad_tr
        curve_inv=np.linalg.inv(curve)
        errs=np.diag(curve_inv)
        dp=np.dot(grad,curve_inv)
        fitp=fitp-dp
        frac_shift=dp/errs
        print fitp,errs,frac_shift


    #C=np.dot(guess,vecs)
    print 'mean diff is ',np.mean(np.abs(C_scale-C))
    #return datft,vecs,nu,C
    return fitp,nu_ref,C
    
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

    nn=tod.info['dat_calib'].size
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
    nsrc=len(pars)/npp
    fitp_gauss=np.zeros(npp)
    dat=tod.info['dat_calib']
    big_derivs=np.zeros([np*nsrc,dat.shape[0],dat.shape[1]])
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
    tmp=0*tod.info['dat_calib']
    nn=tod.info['dat_calib'].size
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

    delt_filt=tod.apply_noise(tod.info['dat_calib']-pred)
    delt_filt=np.reshape(delt_filt,nn)

    dvec=np.reshape(tod.info['dat_calib'],nn)
    predvec=np.reshape(pred,nn)
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
        delt=tod.info['dat_calib']-pred
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
        sz=tod.info['dat_calib'].shape
        tmp=np.zeros(sz)
        npp=derivs.shape[0]
        nn=derivs.shape[1]
        delt=tod.info['dat_calib']-pred
        delt_filt=tod.apply_noise(delt)
        #delt_filt[:,1:-1]=delt_filt[:,1:-1]*2
        delt_filt[:,0]=delt_filt[:,0]*0.5
        delt_filt[:,-1]=delt_filt[:,-1]*0.5
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
        grad=grad+0.5*(grad1+grad2)
        curve=curve+np.dot(derivs,derivs_filt.transpose())
        chisq=chisq+np.dot(delt,delt_filt)
    curve=0.5*(curve+curve.transpose())
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
        
def invscale(mat):
    vec=1/np.sqrt(np.diag(mat))
    mm=np.outer(vec,vec)
    mat=mm*mat
    #ee,vv=np.linalg.eig(mat)
    #print 'rcond is ',ee.max()/ee.min(),vv[:,np.argmin(ee)]
    return mm*np.linalg.inv(mat)
def fit_timestreams_with_derivs_test(func,pars,tods,to_fit=None,to_scale=None,tol=1e-2,chitol=1e-4,maxiter=10,scale_facs=None):
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
                print 'we have converged'
                converged=True
        else:
            conv_fac=None
        to_print=np.asarray([3600*180.0/np.pi,3600*180.0/np.pi,3600*180.0/np.pi,1.0,1.0,3600*180.0/np.pi,3600*180.0/np.pi,3600*180.0/np.pi*np.sqrt(8*np.log(2)),1.0])*(pp-pars)
        print 'iter',iter,' max_shift is ',conv_fac,' with lamda ',lamda,chi_ref-chi_cur,chi_ref-chi_new
    return pp,chi_cur
def fit_timestreams_with_derivs(func,pars,tods,to_fit=None,to_scale=None,tol=1e-2,maxiter=10,scale_facs=None):
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
            sz=tod.info['dat_calib'].shape
            pred,derivs=func(pp,tod)
            if not (to_fit is None):
                derivs=np.dot(rotmat.transpose(),derivs)
            derivs_filt=0*derivs
            tmp=np.zeros(sz)
            npp=derivs.shape[0]
            nn=derivs.shape[1]
            delt=tod.info['dat_calib']-pred
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
            print 'We have converged.'
            converged=True
        if not (to_fit is None):
            shifts=np.dot(rotmat,shifts)
        if not(scale_facs is None):
            if iter<len(scale_facs):
                print 'rescaling shift by ',scale_facs[iter]
                shifts=shifts*scale_facs[iter]
        to_print=np.asarray([3600*180.0/np.pi,3600*180.0/np.pi,3600*180.0/np.pi,1.0,1.0,3600*180.0/np.pi,3600*180.0/np.pi,3600*180.0/np.pi*np.sqrt(8*np.log(2)),1.0])*(pp-pars)
        print 'iter ',iter,' max shift is ',conv_fac,' with chisq improvement ',chi_ref-chisq,to_print #converged,pp,shifts
        pp=pp+shifts

        iter=iter+1
    return pp,chisq
