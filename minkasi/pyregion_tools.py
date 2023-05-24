import pyregion

from astropy.io import fits
from astropy.wcs import WCS

import numpy as np

import copy

__all__ = ["region_binner",
           "bootstrap"]

def bootstrap(data, n = 10000):
    """
    Bootstraps data.

    Given an input data vector, the bootstrapping proceedure selects a random subsample of data, with replacement, and computes a statistic 
    on that subsample. TODO: currently only mean is supported, should add others. The variance of the resulting statistics is a good measure
    of the true variance of that statistic. In other words, if we have some data, and we compute mean(data) and want to know what var(data)
    is, bootstrapping is a good way to do that.

    Parameters
    ----------
    data : numpy.array
        array of data which we wish to bootstrap
    n : int, optional
        number of instances of bootstrapping to perform

    Returns
    -------
    stats : numpy.array
        array of average of the bootstrapped samples
    """

    stats = np.zeros(n)
    data = np.array(data)
    for i in range(n):
        flags = np.random.randint(len(data), size = len(data))
        stats[i] = np.mean(data[flags])
    return stats

def region_binner(shapeList, hdu, plot = False, return_rs = True):
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
        return _region_binner_explicit_regions(shapeList, hdu, plot = plot, return_rs = return_rs)

    elif shapeList[0].name == 'epanda':
        #Handles epanda ShapeLists where n radial bins has been specified
        return _region_binner_epanda_rbins(shapeList, hdu, plot = plot)
       
    else:
        print('Error: region type {} not currently supported'.format(shapeList[0].name))
        return


    

def _region_binner_explicit_regions(shapeList, hdu, plot = False, return_rs = True):
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
    return_rs : Bool, optional
        specifies whether to generate and return radii associated with means/vars

    Returns
    -------
    means : np.array(float)
        simple average of data within each binning region
    var : np.array(float)
        variance within the same regions as computed by bootstrapping the pixels
    rs : np.array(float)
        region radii associated with means/var
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

    if return_rs:
        rs = np.zeros(len(means))
   
        header = hdu[0].header
        cdelt = header.get('CDELT2', 1.0)

        for i in range(len(rs)):
            if shapeList[i].name == 'epanda':
                #If region is epanda, return the average of r_semimajor at the endpoints of the region
                rs[i] = np.mean([shapeList[i].coord_list[5], shapeList[i].coord_list[7]])*cdelt*3600 
            elif shapeList[i].name == 'panda':
                #If region is  panda, return average of r at endpoints of region
                rs[i] = np.mean(shapeList[i].coord_list[4:5])*cdelt*3600
            else:
                print("Error: region type {} not supported for automatic radii generation".format(shapeList[i].name))
                return means, var
        return means, var, rs
    
    return means, var

def _region_binner_epanda_rbins(shapeList, hdu, plot=False, return_rs = True):
    #                    0   1       2        3          4         5        6         7       8            9                10
    #Epanda has format (ra, dec, ang_start, ang_end, n_ang_bins, inner a, Inner b, outer a, outer b, number of radial bins, PA)
    coords = shapeList[0].coord_list

    step_a = (coords[7]-coords[5])/coords[-2]
    step_b = (coords[8]-coords[6])/coords[-2]
    
    #N bins located at second to last
    means = np.zeros(coords[-2])
    var = np.zeros(coords[-2])
    
    if return_rs:
        rs = np.zeros(coords[-2])

    for i in range(coords[-2]):
        temp_r = copy.deepcopy(shapeList)

        #note that the inner_a and outer_a in the coord_list are for the region as a whole
        inner_a = coords[5] + i*step_a
        outer_a = coords[5] + (i)*step_a+step_a
        
        inner_b = coords[6] + i*step_b
        outer_b = coords[6] + (i)*step_b+step_b

        temp_r[0].coord_list[5] = inner_a
        temp_r[0].coord_list[7] = outer_a
        
        temp_r[0].coord_list[6] = inner_b
        temp_r[0].coord_list[8] = outer_b
        
        temp_r[0].coord_list[-2] = 1
        
        tempfilter = temp_r.get_filter()
     
        tempmask = tempfilter.mask(hdu[0].data.shape)
        
        if plot:
            plt.imshow(hdu[0].data*tempmask,origin = 'lower')
            plt.show()
            plt.close()

        masked_data = hdu[0].data*tempmask
        vals = masked_data[np.abs(masked_data)>1e-8]
        
        means[i] = np.mean(vals)
        var[i] = np.var(bootstrap(vals))
        
        if return_rs:
            rs[i] = inner_a + i*step_a/2    
    if return_rs:
        return means, var, rs
    
    return means, var


