
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
                nchunk=int(np.ceil(nchunk))
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
        beam=interp(rr,prof[:,0],prof[:,1])
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
        self.nx_coarse=int(np.round(hits.shape[0]/self.osamp))
        self.ny_coarse=int(np.round(hits.shape[1]/self.osamp))
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


class HealMap(SkyMap):
    def __init__(self,proj='RING',nside=512,tag='ipix'):
        if not(have_healpy):
            printf("Healpix map requested, but healpy not found.")
            return
        self.proj=proj
        self.nside=nside
        self.nx=healpy.nside2npix(self.nside)
        self.ny=1
        self.caches=None
        self.tag=tag
        self.map=np.zeros([self.nx,self.ny])
    def copy(self):
        newmap=HealMap(self.proj,self.nside,self.tag)
        newmap.map[:]=self.map[:]
        return newmap
    def pix_from_radec(self,ra,dec):
        ipix=healpy.ang2pix(self.nside,np.pi/2-dec,ra,self.proj=='NEST')
        return np.asarray(ipix,dtype='int32')
    #def get_pix(self,tod,savepix=True):
    #    if not(self.tag is None):
    #        ipix=tod.get_saved_pix(self.tag)
    #        if not(ipix is None):
    #            return ipix
    #    ra,dec=tod.get_radec()
    #    #ipix=healpy.ang2pix(self.nside,np.pi/2-tod.info['dy'],tod.info['dx'],self.proj=='NEST')
    #    ipix=healpy.ang2pix(self.nside,np.pi/2-dec,ra,self.proj=='NEST')
    #    if savepix:
    #        tod.save_pixellization(self.tag,ipix)            
    #    return ipix
    def write(self,fname='map.fits',overwrite=True):
        if self.map.shape[1]<=1:
            healpy.write_map(fname,self.map[:,0],nest=(self.proj=='NEST'),overwrite=overwrite)        
    


class SkyMapCar(SkyMap):
    def pix_from_radec(self,ra,dec):
        xpix=np.round((ra-self.lims[0])*self.cosdec/self.pixsize)
        #ypix=np.round((dec-self.lims[2])/self.pixsize)
        ypix=((dec-self.lims[2])/self.pixsize)+0.5
        ipix=np.asarray(xpix*self.ny+ypix,dtype='int32')
        return ipix
        
class SkyMapCarOld:
    def __init__(self,lims,pixsize):
        try:
            self.lims=lims.copy()
        except:
            self.lims=lims[:]
        self.pixsize=pixsize
        self.cosdec=np.cos(0.5*(lims[2]+lims[3]))
        nx=int(np.ceil((lims[1]-lims[0])/pixsize*self.cosdec))
        ny=int(np.ceil((lims[3]-lims[2])/pixsize))
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
