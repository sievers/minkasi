from numba import jit

import numpy as np

from scipy.linalg import norm
import scipy.integrate as integrate
from scipy.interpolate import interp1d

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
        return(np.squeeze(needs))

    def get_needlet_filters_2d(self,fourier_radii, return_filt=False, plot = False):

        filters=[]
        for j in self.js:
            interp_func = interp1d(self.k_arr,self.bands[j]) # interpolating is faster than computing bands for every row
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



@jit(nopython=True)
def map2wav(imaps, filters):
    """
    Transform from a regular map to a multimap of wavelet coefficients. Adapted from enmap.wavelets.
    
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
    
    filtered_slices_real = []
    filtered_slices_fourier = []

    for i in range(len(imaps)):
        lightcone_ft = np.fft.fftn(np.fft.fftshift(imaps[i]))

        filtered_slice_real = []
        filtered_slice_fourier = []
        for filt in filters:
            fourier_filtered = np.fft.fftshift(filt)*lightcone_ft
            filtered_slice_fourier.append(fourier_filtered)
            filtered_slice_real.append(np.fft.fftshift(np.real(np.fft.ifftn(fourier_filtered))))# should maybe catch if imag(fourier_filtered)!=0 here 

        filtered_slices_real.append(np.array(filtered_slice_real))
        filtered_slices_fourier.append(np.array(filtered_slice_fourier))



































