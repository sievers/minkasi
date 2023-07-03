
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
    def __init__(self,lims,pixsize,poltag='I',proj='CAR',pad=2,primes=None,cosdec=None,nx=None,ny=None,mywcs=None,tag='ipix',purge_pixellization=False,ref_equ=False):
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
        self.tag=tag
        self.purge_pixellization=purge_pixellization
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
        if False:
            newmap=PolMap(self.lims,self.pixsize,self.poltag,self.proj,self.pad,self.primes,cosdec=self.cosdec,nx=self.nx,ny=self.ny,mywcs=self.wcs)
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
    def pix_from_radec(self,ra,dec):
        ndet=ra.shape[0]
        nsamp=ra.shape[1]
        nn=ndet*nsamp
        coords=np.zeros([nn,2])
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
        if False:
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
        else:
            ra,dec=tod.get_radec()
            ipix=self.pix_from_radec(ra,dec)
        if savepix:
            if not(self.tag is None):
                tod.save_pixellization(self.tag,ipix)
        return ipix
    def map2tod(self,tod,dat,do_add=True,do_omp=True):
        ipix=self.get_pix(tod)
        if self.npol>1:
            #polmap2tod(dat,self.map,self.poltag,tod.info['twogamma_saved'],tod.info['ipix'],do_add,do_omp)
            polmap2tod(dat,self.map,self.poltag,tod.info['twogamma_saved'],ipix,do_add,do_omp)
        else:
            #map2tod(dat,self.map,tod.info['ipix'],do_add,do_omp)
            map2tod(dat,self.map,ipix,do_add,do_omp)

        
    def tod2map(self,tod,dat,do_add=True,do_omp=True):
        if do_add==False:
            self.clear()
        ipix=self.get_pix(tod)
        #print('ipix start is ',ipix[0,0:500:100])
        if self.npol>1:
            #tod2polmap(self.map,dat,self.poltag,tod.info['twogamma_saved'],tod.info['ipix'])
            tod2polmap(self.map,dat,self.poltag,tod.info['twogamma_saved'],ipix)
            if self.purge_pixellization:
                tod.clear_saved_pix(self.tag)
            return
        #print("working on nonpolarized bit")

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
                nchunk=int(np.ceil(nchunk))
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
class HealPolMap(PolMap):
    def __init__(self,poltag='I',proj='RING',nside=512,tag='ipix',purge_pixellization=False):
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
        self.tag=tag
        self.purge_pixellization=purge_pixellization
        if self.npol>1:
            self.map=np.zeros([self.nx,self.ny,self.npol])
        else:
            self.map=np.zeros([self.nx,self.ny])
    def copy(self):
        if False:
            newmap=HealPolMap(self.poltag,self.proj,self.nside,self.tag)
            newmap.map[:]=self.map[:]
            return newmap
        else:
            return copy.deepcopy(self)
    #def get_pix(self,tod):
    #    ipix=healpy.ang2pix(self.nside,np.pi/2-tod.info['dy'],tod.info['dx'],self.proj=='NEST')
    #    return ipix
    def pix_from_radec(self,ra,dec):
        ipix=healpy.ang2pix(self.nside,np.pi/2-dec,ra,self.proj=='NEST')
        return np.asarray(ipix,dtype='int32')
    #def get_pix(self,tod,savepix=True):
    #    if not(self.tag is None):
    #        ipix=tod.get_saved_pix(self.tag)
    #        if not(ipix is None):
    #            return ipix
    #    ra,dec=tod.get_radec()
    #    ipix=self.pix_from_radec(ra,dec)
    #    if savepix:
    #        if not(self.tag is None):
    #            tod.save_pixellization(self.tag,ipix)
    #    return ipix

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
    
