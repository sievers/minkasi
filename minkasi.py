import numpy 
import ctypes
import time
import pyfftw


mylib=ctypes.cdll.LoadLibrary("libminkasi.so")
tod2map_simple_c=mylib.tod2map_simple
tod2map_simple_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

tod2map_omp_c=mylib.tod2map_omp
tod2map_omp_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]

map2tod_simple_c=mylib.map2tod_simple
map2tod_simple_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]

map2tod_omp_c=mylib.map2tod_omp
map2tod_omp_c.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,ctypes.c_int]


set_nthread_c=mylib.set_nthread
set_nthread_c.argtypes=[ctypes.c_int]

get_nthread_c=mylib.get_nthread
get_nthread_c.argtypes=[ctypes.c_void_p]

def tod2map_simple(map,dat,ipix):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    tod2map_simple_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data)

def tod2map_omp(map,dat,ipix):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    tod2map_omp_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data,map.size)
    

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
    nthread=numpy.zeros([1,1],dtype='int32')
    get_nthread_c(nthread.ctypes.data)
    return nthread[0,0]


def get_type(nbyte):
    if nbyte==8:
        return numpy.dtype('float64')
    if nbyte==4:
        return numpy.dtype('float32')
    if nbyte==-4:
        return numpy.dtype('int32')
    if nbyte==-8:
        return numpy.dtype('int64')
    if nbyte==1:
        return numpy.dtype('str')
    print 'Unsupported nbyte ' + repr(nbyte) + ' in get_type'
    return None

def read_octave_struct(fname):
    f=open(fname)
    nkey=numpy.fromfile(f,'int32',1)[0]
    #print 'nkey is ' + repr(nkey)
    dat={}
    for i in range(nkey):
        key=f.readline().strip()
        #print 'key is ' + key
        ndim=numpy.fromfile(f,'int32',1)[0]
        dims=numpy.fromfile(f,'int32',ndim)
        dims=numpy.flipud(dims)
        #print 'Dimensions of ' + key + ' are ' + repr(dims)
        nbyte=numpy.fromfile(f,'int32',1)[0]
        #print 'nbyte is ' + repr(nbyte)
        dtype=get_type(nbyte)
        tmp=numpy.fromfile(f,dtype,dims.prod())
        dat[key]=numpy.reshape(tmp,dims)
    f.close()
    return dat



def nsphere_vol(np):
    iseven=(np%2)==0
    if iseven:
        nn=np/2
        vol=(numpy.pi**nn)/numpy.prod(numpy.arange(1,nn+1))
    else:
        nn=(np-1)/2
        vol=2**(nn+1)*numpy.pi**nn/numpy.prod(numpy.arange(1,np+1,2))
    return vol


def _prime_loop(ln,lp,icur,lcur,vals):
    facs=numpy.arange(lcur,ln+1e-3,lp[0])
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
        facs=numpy.arange(lcur,ln,lp[0])
        for fac in facs:
            icur=_prime_loop(ln,lp[1:],icur,fac,vals)
        return icur
    print 'I don''t think I should have gotten here.'
    return icur
                             
        

def find_good_fft_lens(n,primes=[2,3,5,7]):
    lmax=numpy.log(n+0.5)
    np=len(primes)
    vol=nsphere_vol(np)

    r=numpy.log2(n+0.5)
    lp=numpy.log2(primes)
    npoint_max=(vol/2**np)*numpy.prod(r/lp)+30 #add a bit just to make sure we don't act up for small n
    vals=numpy.zeros(npoint_max)
    icur=0
    icur=_prime_loop(r,lp,icur,0.0,vals)
    assert(icur<=npoint_max)
    myvals=numpy.asarray(numpy.round(2**vals[:icur]),dtype='int')
    myvals=numpy.sort(myvals)
    return myvals
    
    

def _linfit_2mat(dat,mat1,mat2):
    np1=mat1.shape[1]
    np2=mat2.shape[1]
    mm=numpy.append(mat1,mat2,axis=1)
    lhs=numpy.dot(mm.transpose(),mm)
    rhs=numpy.dot(mm.transpose(),dat)
    lhs_inv=numpy.linalg.inv(lhs)
    fitp=numpy.dot(lhs_inv,rhs)
    fitp1=fitp[0:np1].copy()
    fitp2=fitp[np1:].copy()
    assert(len(fitp2)==np2)
    return fitp1,fitp2



def smooth_spectra(spec,fwhm):
    nspec=spec.shape[0]
    n=spec.shape[1]

    x=numpy.arange(n)
    sig=fwhm/numpy.sqrt(8*numpy.log(2))
    to_conv=numpy.exp(-0.5*(x/sig)**2)
    to_conv=to_conv/to_conv.sum()/2/n
    to_conv_ft=pyfftw.fft_r2r_1d(to_conv)
    print to_conv_ft.sum(), to_conv.sum()
    xtrans=pyfftw.fft_r2r(spec)
    for i in range(nspec):
        xtrans[i,:]=xtrans[i,:]*to_conv_ft
    return pyfftw.fft_r2r(xtrans),to_conv



def fit_cm_plus_poly(dat,ord=2,cm_ord=1,niter=2,medsub=False):
    n=dat.shape[1]
    ndet=dat.shape[0]
    if medsub:
        med=numpy.median(dat,axis=1)        
        dat=dat-numpy.repeat([med],n,axis=0).transpose()
        
        

    xx=numpy.arange(n)+0.0
    xx=xx-xx.mean()
    xx=xx/xx.max()

    pmat=numpy.polynomial.legendre.legvander(xx,ord)
    cm_pmat=numpy.polynomial.legendre.legvander(xx,cm_ord-1)
    calfacs=numpy.ones(ndet)*1.0
    dd=dat.copy()
    for i in range(1,niter):
        for j in range(ndet):
            dd[j,:]/=calfacs[j]
            
        cm=numpy.median(dd,axis=0)
        cm_mat=numpy.zeros(cm_pmat.shape)
        for i in range(cm_mat.shape[1]):
            cm_mat[:,i]=cm_pmat[:,i]*cm
        fitp_p,fitp_cm=_linfit_2mat(dat.transpose(),pmat,cm_mat)
        pred1=numpy.dot(pmat,fitp_p).transpose()
        pred2=numpy.dot(cm_mat,fitp_cm).transpose()
        pred=pred1+pred2
        dd=dat-pred1
        

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

def run_pcg(b,x0,tods,mapset,precon=None,maxiter=25):
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
    k=0

    zr=r.dot(z)
    x=x0.copy()
    for iter in range(maxiter):
        print iter,zr
        Ap=tods.dot(p)
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
    return x

def apply_noise(tod,dat=None):
    if dat is None:
        dat=tod['dat_calib']
    dat_rot=numpy.dot(tod['v'],dat)
    datft=pyfftw.fft_r2r(dat_rot)
    nn=datft.shape[1]
    datft=datft*tod['mywt'][:,0:nn]
    dat_rot=pyfftw.fft_r2r(datft)
    dat=numpy.dot(tod['v'].transpose(),dat_rot)
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
        self.maps.append(map)
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
    def add(self,mapset):
        mm=self.copy()
        mm.axpy(mapset,1.0)
        return mm

    def sub(self,mapset):
        mm=self.copy()
        mm.axpy(mapset,-1.0)
        return mm

class SkyMapCar:
    def __init__(self,lims,pixsize):
        try:
            self.lims=lims.copy()
        except:
            self.lims=lims[:]
        self.pixsize=pixsize
        self.cosdec=numpy.cos(0.5*(lims[2]+lims[3]))
        nx=numpy.int(numpy.ceil((lims[1]-lims[0])/pixsize*self.cosdec))
        ny=numpy.int(numpy.ceil((lims[3]-lims[2])/pixsize))
        self.nx=nx
        self.ny=ny
        self.npix=nx*ny
        self.map=numpy.zeros([nx,ny])
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
        xpix=numpy.round((tod.info['dx']-self.lims[0])*self.cosdec/self.pixsize)
        ypix=numpy.round((tod.info['dy']-self.lims[2])/self.pixsize)
        #ipix=numpy.asarray(ypix*self.nx+xpix,dtype='int32')
        ipix=numpy.asarray(xpix*self.ny+ypix,dtype='int32')
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
        xvec=numpy.arange(self.nx)
        xvec=xvec-xvec.mean()        
        yvec=numpy.arange(self.ny)
        yvec=yvec-yvec.mean()
        ymat,xmat=numpy.meshgrid(yvec,xvec)
        rmat=numpy.sqrt(xmat**2+ymat**2)
        th=numpy.arctan2(xmat,ymat)
        return rmat,th
    def dot(self,map):
        tot=numpy.sum(self.map*map.map)
        return tot

class Tod:
    def __init__(self,info):
        self.info=info.copy()
    def lims(self):
        xmin=self.info['dx'].min()
        xmax=self.info['dx'].max()
        ymin=self.info['dy'].min()
        ymax=self.info['dy'].max()
        return xmin,xmax,ymin,ymax
    def set_tag(self,tag):
        self.info['tag']=tag
    def copy(self):
        return Tod(self.info)

    def apply_noise(self,dat=None):
        if dat is None:
            dat=self.info['dat_calib']
        dat_rot=numpy.dot(self.info['v'],dat)
        datft=pyfftw.fft_r2r(dat_rot)
        nn=datft.shape[1]
        datft=datft*self.info['mywt'][:,0:nn]
        dat_rot=pyfftw.fft_r2r(datft)
        dat=numpy.dot(self.info['v'].transpose(),dat_rot)
        return dat
    def dot(self,mapset,mapset_out):
        tmp=0.0*self.info['dat_calib']
        for map in mapset.maps:
            map.map2tod(self,tmp)
        self.apply_noise(tmp)
        for map in mapset_out.maps:
            map.tod2map(self,tmp)
    

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
        return [xmin,xmax,ymin,ymax]
    def set_pix(self,map):
        for tod in self.tods:
            ipix=map.get_pix(tod)
            tod.info['ipix']=ipix
    def dot(self,mapset,mapset2=None,report_times=False):
        if mapset2 is None:
            mapset2=mapset.copy()
            mapset2.clear()

        times=numpy.zeros(self.ntod)
        #for tod in self.tods:
        for i in range(self.ntod):
            tod=self.tods[i]
            t1=time.time()
            tod.dot(mapset,mapset2)
            t2=time.time()
            times[i]=t2-t1
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
        tmp=numpy.ones(tod.info['dat_calib'].shape)
        hits.tod2map(tod,tmp)
    return hits
