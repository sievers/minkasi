
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
