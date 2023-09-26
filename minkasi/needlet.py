from numba import jit

from astropy.io import fits

import numpy as np

from scipy.linalg import norm
import scipy.integrate as integrate
from scipy.interpolate import interp1d

from minkasi import get_wcs, tod2map_simple, map2tod, comm
import minkasi 

from pixell import enmap

import matplotlib.pyplot as plt

import copy

class WavSkyMap:
    def __init__(self,lims, filters,pixsize=0,proj='CAR',pad=2, square=False, multiple=False, primes=None,cosdec=None,nx=None,ny=None,mywcs=None,tag='ipix',purge_pixellization=False,ref_equ=False):
        if mywcs is None:
            assert(pixsize!=0) #we had better have a pixel size if we don't have an incoming WCS that contains it
            self.wcs=get_wcs(lims,pixsize,proj,cosdec,ref_equ)
        else:
            self.wcs=mywcs
            pixsize_use=mywcs.wcs.cdelt[1]*np.pi/180
            pixsize=pixsize_use

        self.lims = lims
       
        nx, ny = self.get_npix(pad, nx=nx, ny=ny)
  
        if not(primes is None):
            lens=find_good_fft_lens(2*(nx+ny),primes)
            nx=lens[lens>=nx].min()
            ny=lens[lens>=ny].min() 
            self.primes=primes[:]
        else:
            self.primes=None

        if square:
            if nx != ny:
                nmax = max(nx, ny)
                ratio_x = nmax/nx
                ratio_y = nmax/ny

                self.lims[0] = self.lims[1] - ratio_x*(self.lims[1] - self.lims[0]) #note we are adjusting lims in place here
                self.lims[3] = self.lims[2] + ratio_y*(self.lims[3] - self.lims[2]) #Resize x and y lims by ratio of nx/ny to nmax

                nx, ny = self.get_npix(pad)

        if multiple:
          assert(type(multiple) == int)
          
          xmax = 2*np.ceil(nx / multiple)
          xdiff = self.lims[1] - self.lims[0]
          self.lims[0] = self.lims[1] - xdiff * xmax/nx #Make nx a factor of 2
 
          ymax = 2*np.ceil(ny / multiple) 
          ydiff = self.lims[3] - self.lims[2]
          self.lims[3] = self.lims[2] + ydiff * ymax/ny
          
          nx, ny = self.get_npix(pad) #This may be applying pad a bunch of times
 
        self.nx=nx
        self.ny=ny
        self.pixsize=pixsize
        self.map=np.zeros([nx,ny])
        self.proj=proj
        self.pad=pad
        self.tag=tag
        self.purge_pixellization=purge_pixellization
        self.caches=None
        self.cosdec=cosdec
        self.tod2map_method=None
        self.filters = filters
        self.nfilt = len(self.filters)
        self.wmap = np.zeros([self.nfilt, nx, ny])

    def get_npix(self, pad, nx=None, ny=None):
        corners=np.zeros([4,2])
        corners[0,:]=[self.lims[0],self.lims[2]]
        corners[1,:]=[self.lims[0],self.lims[3]]
        corners[2,:]=[self.lims[1],self.lims[2]]
        corners[3,:]=[self.lims[1],self.lims[3]]

        pix_corners=self.wcs.wcs_world2pix(corners*180/np.pi,1)
        pix_corners=np.round(pix_corners)

        if pix_corners.min()<-0.5:
            print('corners seem to have gone negative in SkyMap projection.  not good, you may want to check this.')
        if True: #try a patch to fix the wcs xxx
            if nx is None:
                nx=(pix_corners[:,0].max()+pad)
            if ny is None:
                ny=(pix_corners[:,1].max()+pad)
        else:#What is this else doing here?
            nx=(pix_corners[:,0].max()+pad)
            ny=(pix_corners[:,1].max()+pad)
        nx=int(nx)
        ny=int(ny)

        return nx, ny

    def get_caches(self):
        npix=self.nx*self.ny
        nthread=get_nthread()
        self.caches=np.zeros([nthread,npix])
    def clear_caches(self):
        self.map[:]=np.reshape(np.sum(self.caches,axis=0),self.map.shape)
        self.caches=None
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
        self.wmap[:]=self.wmap[:]+a*map.wmap[:]
    def assign(self,arr):
        assert(arr.shape[0]==self.nx)
        assert(arr.shape[1]==self.ny)
        self.map[:]=arr
    def pix_from_radec(self,ra,dec):
        ndet=ra.shape[0]
        nsamp=ra.shape[1]
        nn=ndet*nsamp
        coords=np.zeros([nn,2])
        coords[:,0]=np.reshape(ra*180/np.pi,nn)
        coords[:,1]=np.reshape(dec*180/np.pi,nn)

        pix=self.wcs.wcs_world2pix(coords,1)
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
        if False:
            ndet=ra.shape[0]
            nsamp=ra.shape[1]
            nn=ndet*nsamp
            coords=np.zeros([nn,2])
            coords[:,0]=np.reshape(ra*180/np.pi,nn)
            coords[:,1]=np.reshape(dec*180/np.pi,nn)

            pix=self.wcs.wcs_world2pix(coords,1)
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

    def tod2map_simple(self,tod,dat):
        ipix=self.get_pix(tod)
        tod2map_simple(self.map,dat,ipix)


    def map2tod(self,tod,dat,do_add=True,do_omp=True):
        ipix=self.get_pix(tod)
        self.map = np.squeeze(wav2map_real(self.wmap, self.filters), axis = 0) #Right now let's restrict ourself to 1 freqency input maps, so we need to squeeze down the dummy axis added by map2wav_real 
        map2tod(dat,self.map,ipix,do_add,do_omp)

    def tod2map(self,tod,dat=None,do_add=True,do_omp=True):
        if dat is None:
            dat=tod.get_data()
        if do_add==False:
            self.clear()
        ipix=self.get_pix(tod)

        if not(self.caches is None):
            tod2map_cached(self.caches,dat,ipix)

        tod2map_simple(self.map,dat,ipix)
        self.wmap = np.squeeze(map2wav_real(self.map, self.filters), axis = 0) #Right now let's restrict ourself to 1 freqency input maps, so we need to squeeze down the dummy axis added by map2wav_real

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
        tot=np.sum(self.wmap*map.wmap)
        return tot

    def plot(self,plot_info=None):
        vmin=self.map.min()
        vmatod.set_noise_smoothed_svd()
        x=self.map.max()
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
        self.map = np.squeeze(wav2map_real(self.wmap, self.filters), axis = 0) 
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
        if minkasi.have_mpi:
            if chunksize>0:
                nchunk=(1.0*self.nx*self.ny)/chunksize
                nchunk=int(np.ceil(nchunk))
            else:
                nchunk=1
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

                if len(self.map.shape)>1:
                    self.map[:]=np.reshape(tmp,self.map.shape)
    def invert(self):
        mask=np.abs(self.map)>0
        self.map[mask]=1.0/self.map[mask]



class needlet:
    #Lifted from Joelle's needlet code
    def __init__(self, js, B=None, kmax_dimless = None, lightcone = None, L = None):
        '''
        Init function. 

        Parameters:
        -----------
        js: array of ints
            Resolution parameter; effectively sets central location of bandpass
            filter. 

        B : float
            The parameter B of the needlet which controls the width of the filters. 
            Should be larger that 1.

        kmax : int
            The maximum k mode for which the filters will be constructed. Dimensionless units!
        '''

        ####### init attributes
        self.js =js
        self.lightcone = lightcone
        self.kmax_dimless = kmax_dimless

        if self.lightcone is not None: 
            self.lightcone_box = cosmo_box(lightcone, L)
            self.kmax_dimless = self.lightcone_box.kmax_dimless
        self.nfilt = len(self.js)
        self.k_arr = np.append(np.array([0]), np.logspace(0, np.log10(self.kmax_dimless), int(10*self.kmax_dimless)))
        if B is not None:
            self.B = B
        else:
            self. B = self.k_arr[-1]**(1/self.js[-1])*1.01 #Set needlet width to just cover k_arr
        self.bands = self.get_needlet_bands_1d()



    def get_needlet_bands_1d(self):
        needs=[]
        bl2=np.vectorize(self.__b2_need)


        for j in self.js:
            xi=(self.k_arr/self.B**j)
            bl=np.sqrt(bl2(xi))
            needs.append(bl)
        needs = np.squeeze(needs)
        needs[0][0] = 1 #Want the k=0 mode to get the map average
        return needs

    def get_needlet_filters_2d(self,fourier_radii, return_filt=False, plot = False):

        filters=[]
        for j in self.js:
            interp_func = interp1d(self.k_arr,self.bands[j], fill_value='extrapolate') # interpolating is faster than computing bands for every row. 
                                                                                      #We should not extrapolate but occasionally the last bin will be very slightly outside range
            filter_2d = []
            for row in fourier_radii:
                filter_2d.append(interp_func(row))
            filters.append(np.array(filter_2d))

        self.filters=np.array(filters)


        if plot:
            n = self.nfilt//2
            fig,ax = plt.subplots(nrows = 2, ncols=n)

            for i in range(n):
                ax[0,i].imshow(self.filters[i])
                ax[0,i].set_title(f"2D filter for j={i}")
                ax[0,i].set_yticklabels([])
                ax[0,i].set_xticklabels([])

                ax[1,i].imshow(self.filters[i+n])
                ax[1,i].set_title(f"2D filter for j={i+n}")
                ax[1,i].set_yticklabels([])
                ax[1,i].set_xticklabels([])

            plt.show()

        if return_filt:
            return self.filters

    def filter_lightcone(self, return_fourier=False, plot = False, plot_norm = "lin", n_nu_plot = None):

        filtered_slices_real = []
        filtered_slices_fourier = []
        fourier_radii = self.lightcone_box.get_grid_dimless_2d(return_grid=True)
        self.get_needlet_filters_2d(fourier_radii)

        for i in range(len(self.lightcone)):
            lightcone_ft = np.fft.fftn(np.fft.fftshift(self.lightcone[i]))

            filtered_slice_real = []
            filtered_slice_fourier = []
            for filt in self.filters:
                fourier_filtered = np.fft.fftshift(filt)*lightcone_ft
                filtered_slice_fourier.append(fourier_filtered)
                filtered_slice_real.append(np.fft.fftshift(np.real(np.fft.ifftn(fourier_filtered))))# should maybe catch if imag(fourier_filtered)!=0 here 

            filtered_slices_real.append(np.array(filtered_slice_real))
            filtered_slices_fourier.append(np.array(filtered_slice_fourier))

        if return_fourier:
            return np.array(filtered_slices_real), np.array(filtered_slices_fourier)
        else:
            return np.array(filtered_slices_real)

    def back_transform(self,filtered_boxes):
        back_transformed = []
        for nu in range(len(self.lightcone)):
            fourier_boxes = []
            for b in filtered_boxes[nu]:
                fourier_boxes.append(np.fft.fftn(np.fft.fftshift(b)))

            back_transform = np.zeros_like(fourier_boxes[0])
            for i in range(self.nfilt):
                back_transform += np.fft.fftshift(fourier_boxes[i])*self.filters[i]
            back_transform = np.fft.fftshift(np.real(np.fft.ifftn(np.fft.fftshift(back_transform))))
            back_transformed.append(back_transform)
        return np.array(back_transformed)


    #============================== functions for computing 1d filters ================================#
    #======================= Adapted from https://github.com/javicarron/mtneedlet''' ==================#
    #==================================================================================================#

    def __f_need(self,t):
        '''Auxiliar function f to define the standard needlet'''
        if t <= -1.:
            return(0.)
        elif t >= 1.:
            return(0.)
        else:
            return(np.exp(1./(t**2.-1.)))

    def __psi(self,u):
        '''Auxiliar function psi to define the standard needlet'''
        return(integrate.quad(self.__f_need,-1,u)[0]/integrate.quad(self.__f_need,-1,1)[0])

    def __phi(self,q):
        '''Auxiliar function phi to define the standard needlet'''
        B=float(self.B)
        if q < 0.:
            raise ValueError('The multipole should be a non-negative value')
        elif q <= 1./B:
            return(1.)
        elif q >= 1.:
            return(0)
        else:
            return(self.__psi(1.-(2.*B/(B-1.)*(q-1./B))))

    def __b2_need(self,xi):
        '''Auxiliar function b^2 to define the standard needlet'''
        b2=self.__phi(xi/self.B)-self.__phi(xi)
        return(np.max([0.,b2]))
        ## np.max in order to avoid negative roots due to precision errors

    #==============================================================================================#
    #====================================== plotting functions ====================================#
    #==============================================================================================#

    def plot_bands(self):
        fig,ax = plt.subplots()

        self.sum_sq = np.zeros_like(self.k_arr)
        for j,b in enumerate(self.bands):
            ax.plot(self.k_arr, b, label = f"j={j}")
            self.sum_sq+= b**2

        ax.plot(self.k_arr, self.sum_sq, label = f"$\sum b^2$", color = "k")
        ax.set_xscale("log")
        ax.legend(loc = "lower right", ncols = self.nfilt//2)
        ax.set_xlabel("k [dimless]")
        ax.set_ylabel(r"$b_j$")
        plt.show()




###############################################################################################

class cosmo_box:

    def __init__(self, box,L):

        #initializing some attributes
        self.box = box
        self.L = L

        #----------------------------- box specs ------------------------------#
        self.dims = len(self.box.shape) #dimensions of box
        self.N = self.box.shape[1] #number of pixels along one axis of 2D slice
        self.origin = self.N//2 #origin by fft conventions


        self.delta_k = 2*np.pi/self.L #kspace resolution of 1 pixel

        self.kmax_dimless = self.get_kmax_dimless() # maximum |k| in fourier grid, dimensionless (i.e. pixel space)
        self.kmax = self.kmax_dimless*self.delta_k # same as above, but with dimensions


    #======================= fourier functions ====================#    
    def get_kmax_dimless(self):
        self.get_grid_dimless_2d()
        return np.max(self.grid_dimless)

    def get_grid_dimless_2d(self, return_grid = False):
        '''
        Generates a fourier space dimensionless grid, finds 
        radial distance of each pixel from origin.
        '''

        self.indices = (np.indices((self.N,self.N)) - self.origin)
        self.grid_dimless = norm(self.indices, axis = 0) #dimensionless kspace radius of each pix

        if return_grid:
            return self.grid_dimless

def new_map2wav(imaps, filters):
    npix =  imaps.shape[-2]*imaps.shape[-1]
    weights = np.array([np.sum(f**2)/npix for f in filters])

    fmap = np.fft.fft(imaps, norm='backward')#We're gonna do a hack-y thing where we force numpy to un normalize the ffts by calling norm=backwards going forwards and norm=forwards going backwards. We are imposing our own norm
    fmap_npix = 1
    for dim in np.shape(imaps): fmap_npix *= dim 
    to_return = np.zeros((imaps.shape[0], filters.shape[0], imaps.shape[-2], imaps.shape[-1])) 
    for i in range(len(imaps)):
        for j in range(len(filters)):
            fsmall  = fmap[i]
            fsmall *= filters[j] / (weights[i]**0.5 * fmap_npix)

            to_return[i][j] = np.fft.ifft(fsmall, norm='forward').real
    return to_return



def map2wav_real(imaps, filters):
    """
    Transform from a regular map to a multimap of wavelet coefficients. Adapted from Joelles code + enmap.wavelets
    
    Parameters
    ----------
    imap: np.array()
        input map
    basis: 
        needlet basis

    Returns
    -------
    wmap: np.array
        multimap of wavelet coefficients
    """
    
    if len(imaps.shape) == 2:
        imaps = np.expand_dims(imaps, axis = 0)
    elif len(imaps.shape) != 3:
        print("Error: input map must have dim = 2 or 3")
        return
    filtered_slices_real = []

    npix =  imaps.shape[-2]*imaps.shape[-1]
    weights = np.array([np.sum(f**2)/npix for f in filters])
    
    for i in range(len(imaps)):
        lightcone_ft = np.fft.fftn(np.fft.fftshift(imaps[i]))

        filtered_slice_real = []
        for filt in filters:
            fourier_filtered = (np.fft.fftshift(filt)*lightcone_ft)# / (weights[i]**2 * lightcone_ft.shape[-1] * lightcone_ft.shape[-2])
            filtered_slice_real.append(np.fft.fftshift(np.real(np.fft.ifftn(fourier_filtered))))# should maybe catch if imag(fourier_filtered)!=0 here 

        filtered_slices_real.append(np.array(filtered_slice_real))
 
    return np.array(filtered_slices_real)

def wav2map_real(wav_mapset, filters):
    if len(wav_mapset.shape) == 3:
        wav_mapset = np.expand_dims(wav_mapset, axis = 0)

    elif len(wav_mapset.shape) != 4:
        print("Error: input wave mapset must have dim = 3 or 4")
        return

    npix =  wav_mapset.shape[-2]*wav_mapset.shape[-1]
    weights = np.array([np.sum(f**2)/npix for f in filters])


    back_transformed = []
    for nu in range(len(wav_mapset)):
        fourier_boxes = []
        for b in wav_mapset[nu]:
            fourier_boxes.append(np.fft.fftn(np.fft.fftshift(b)))

        npix_f = fourier_boxes[0].shape[-1] * fourier_boxes[0].shape[-2]

        back_transform = np.zeros_like(fourier_boxes[0])
        for i in range(wav_mapset.shape[1]): 
            back_transform += np.fft.fftshift(fourier_boxes[i])*filters[i] #* (weights[i]**2 / npix_f)
        back_transform = np.fft.fftshift(np.real(np.fft.ifftn(np.fft.fftshift(back_transform))))
        back_transformed.append(back_transform)
    return np.array(back_transformed)































