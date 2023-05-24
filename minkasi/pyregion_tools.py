import pyregion

from astropy.io import fits
from astropy.wcs import WCS

import numpy as np

import copy

__all__ = ["anulizer"]

def anulizer(shapeList, hdu, plot = False):
    """
    Helper function for making binned profiles from pyregion and map.

    This function computes the binned mean and variance in regions of a map as specified by a pyregion ShapeList. This function handles
    the various different types of pyregions and sepcifications, i.e. specifying multiple regions explicitly or using the pyregion 
    defualt radial or angular bins.

    Parameters
    ----------
    shapeList : 'pyregion.core.ShapeList'
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
    if len(shapeList) > 1:
        #Handles multiple explicitly defined regions of any type. Since they are explicitly defined one function can handle all
        return _aunulizer_nregions(shapeList, hdu, plot = plot)

    elif shapeList[0].name == 'epanda':
        #Handles epanda ShapeLists where n radial bins has been specified
        return _anulizer_rbins(shapeList, hdu, plot = plot)
       
    else:
        print('Error: region type {} not currently supported'.format(shapeList[0].name))
        return

def _aunulizer_nregions(shapeList, hdu, plot = False):
    """
    Anulizing function for shapeList where each region has been explicilty defined.

    This function takes a list of regions and compute the average and variance within each of those regions.
    It assumes that each region is explicitly defined, meaning there are no radial bins, angular bins, etc. within a region. 
    Further this function assumes that you know what youre doing. For example you can hand it a list of overlapping regions and
    it will return their means/vars without warning. Currently only supported method for computing variance is bootstrapping of
    pixels but TODO add more methods. Since the regions are all explicitly defined we don't need to compute any bins/regions
    ourselves and this function can handle all types of regions. 

    Parameters
    ----------
    shapeList : 'pyregion.core.ShapeList'
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

    means = np.zeros(len(shapeList))
    var = np.zeros(len(shapeList))
    filters = shapeList.get_filter()
    
    for i in range(len(var)):
        cur_filter = filters[i]
        tempmask = cur_filter.mask(hdu[0].data.shape)
        #Sets values in the map outside the mask to zero. This may be cleaner with masked arrays
        masked_data = hdu[0].data*tempmask
        if plot:
            plt.imshow(masked_data,origin = 'lower')
            plt.show()
            plt.close()

        #Since the masked data points are still in the map just set to zero, we have to extract those with non-zero value
        #to compute the mean/var. This is the part that would be cleaner with masked arrays.
        vals = masked_data[np.abs(masked_data)>1e-10]
        
        means[i] = np.mean(vals)
        var[i] = np.var(bootstrap(vals))
    return means, var


