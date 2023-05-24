import pyregion

from astropy.io import fits
from astropy.wcs import WCS

import numpy as np

import copy

__all__ = ["anulizer"]

def anulizer(epanda_region, hdu, plot = False):
    """
    Helper function for making binned profiles from pyregion and map.

    This function computes the binned mean and variance in regions of a map as specified by a pyregion ShapeList. This function handles
    the various different types of pyregions and sepcifications, i.e. specifying multiple regions explicitly or using the pyregion 
    defualt radial or angular bins.

    Parameters
    ----------
    epanda_region : 'pyregion.core.ShapeList'
        pyregion ShapeList specifying the regions overwhich to compute the profile bins. 
    hdu : 'astropy.io.fits.hdu.hdulist.HDUList'
        astropy HDUList containing the relevant map at index 0. Must contain the map data at hdu[0].data
    plot : Bool, optional
        specifies whether to plot the bin regions

    Returns
    -------
    means : np.array(float)
        simple average of data within each binning region
    var : np.array(float)
        variance within the same regions as computed by bootstrapping the pixels
    """

    if len(epanda_region) == 1 and (r2[0].coord_list[9]):
        #Handles epanda ShapeLists where n radial bins has been specified
        return _anulizer_rbins(epanda_region, hdu, plot = plot)
    elif len(epanda_region) > 1:
        #Handles multiple explicitly defined epanda regions
        return _aunulizer_nregions(epanda_region, hdu, plot = plot)

def _aunulizer_nregions(epanda_region, hdu, plot = False):
    #                    0   1       2        3          4         5        6         7       8            9                10
    #Epanda has format (ra, dec, ang_start, ang_end, n_ang_bins, inner a, Inner b, outer a, outer b, number of radial bins, PA)
 
    means = np.zeros(len(epanda_region))
    var = np.zeros(len(epanda_region))
    filters = epanda_region.get_filter()
    
    for i in range(len(var)):
        cur_filter = filters[i]
        tempmask = cur_filter.mask(hdu[0].data.shape)
        masked_data = hdu[0].data*tempmask
        if plot:
            plt.imshow(masked_data,origin = 'lower')
            plt.show()
            plt.close()

        
        vals = masked_data[np.abs(masked_data)>1e-8]
        
        means[i] = np.mean(vals)
        var[i] = np.var(bootstrap(vals))
    return means, var


