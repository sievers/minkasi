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


