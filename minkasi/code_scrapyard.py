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
            derivs,pred=func(pp,tod)
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





