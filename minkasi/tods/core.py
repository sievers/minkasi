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

    def get_saved_pix(self,tag=None):
        if tag is None:
            return None
        if tag in self.info.keys():
            return self.info[tag]
        else:
            return None
    def clear_saved_pix(self,tag=None):
        if tag is None:
            return
        if tag in self.info.keys():
            del(self.info[tag])
    def save_pixellization(self,tag,ipix):
        if tag in self.info.keys():
            print('warning - overwriting key ',tag,' in tod.save_pixellization.')
        self.info[tag]=ipix
    

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
        #self.info['ipix']=ipix
        self.info[map.tag]=ipix
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
    def dot(self,mapset,mapset_out,times=False):
        #tmp=0.0*self.info['dat_calib']
        #for map in mapset.maps:
        #    map.map2tod(self,tmp)
        t1=time.time()
        tmp=self.mapset2tod(mapset)
        t2=time.time()
        tmp=self.apply_noise(tmp)
        t3=time.time()
        self.tod2mapset(mapset_out,tmp)
        t4=time.time()
        #for map in mapset_out.maps:
        #    map.tod2map(self,tmp)
        if times:
            return(np.asarray([t2-t1,t3-t2,t4-t3]))
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

    def get_nsamp(self,reduce=True):
        tot=0
        for tod in self.tods:
            tot=tot+tod.get_nsamp()
        if reduce:
            if have_mpi:
                tot=comm.allreduce(tot)
        return tot

    def dot(self,mapset,mapset2=None,report_times=False,cache_maps=False):
        if mapset2 is None:
            mapset2=mapset.copy()
            mapset2.clear()

        if cache_maps:
            mapset2=self.dot_cached(mapset,mapset2)
            return mapset2
            

        times=np.zeros(self.ntod)
        tot_times=0
        #for tod in self.tods:
        for i in range(self.ntod):
            tod=self.tods[i]
            t1=time.time()
            mytimes=tod.dot(mapset,mapset2,True)
            t2=time.time()
            tot_times=tot_times+mytimes
            times[i]=t2-t1
        if have_mpi:
            mapset2.mpi_reduce()
        print(tot_times)
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
