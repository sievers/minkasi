#This is a template script to show how to fit multi-component models
#directly to timestreams.  The initial part (where the TODs, noise model etc.)
#are set up is the same as general mapping scripts, although we don't
#need to bother with setting a map/pixellization in general (although if
#your timestream model used a map you might).  This script works under
#MPI as well.

import numpy
import numpy as np
from matplotlib import pyplot as plt
import minkasi
import time
import glob
from importlib import reload
reload(minkasi)
plt.ion()

#find tod files we want to map
outroot='maps/zw3146/zw3146'
dir='../data/Zw3146/'
#dir='../data/moo1110/'
#dir='../data/moo1046/'
tod_names=glob.glob(dir+'Sig*.fits')  
tod_names.sort() #make sure everyone agrees on the order of the file names
#tod_names=tod_names[:20] #you can cut the number of TODs here for testing purposes


#if running MPI, you would want to split up files between processes
#one easy way is to say to this:
tod_names=tod_names[minkasi.myrank::minkasi.nproc]
#NB - minkasi checks to see if MPI is around, if not
#it sets rank to 0 an nproc to 1, so this would still
#run in a non-MPI environment



todvec=minkasi.TodVec()

#loop over each file, and read it.
for fname in tod_names:
    t1=time.time()
    dat=minkasi.read_tod_from_fits(fname)
    t2=time.time()
    minkasi.truncate_tod(dat) #truncate_tod chops samples from the end to make
                              #the length happy for ffts
    #minkasi.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
    #                              #this fixes that.  You don't need to, though.
    #minkasi.truncate_tod(dat)    #since our length changed, make sure we have a happy length#
    #

    #figure out a guess at common mode #and (assumed) linear detector drifts/offset
    #drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    dd=minkasi.fit_cm_plus_poly(dat['dat_calib'])  
    
    dat['dat_calib']=dd
    t3=time.time()
    tod=minkasi.Tod(dat)
    todvec.add_tod(tod)
    print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)

#make a template map with desired pixel size an limits that cover the data
#todvec.lims() is MPI-aware and will return global limits, not just
#the ones from private TODs
lims=todvec.lims()
pixsize=2.0/3600*numpy.pi/180
#map=minkasi.SkyMap(lims,pixsize) #we don't need a map when fitting timestreams

#once we have a map, we can figure out the pixellization of the data.  Save that
#so we don't have to recompute.  Also calculate a noise model.  The one here
#is to rotate the data into SVD space, then smooth the power spectrum of each mode. 
# The smoothing could do with a bit of tuning..


for tod in todvec.tods:
    #ipix=map.get_pix(tod) #don't need pixellization for a timestream fit...
    #tod.info['ipix']=ipix
    #tod.set_noise_smoothed_svd()
    tod.set_noise(minkasi.NoiseSmoothedSVD)


#we need an initial guess since this fitting routine is
#for nonlinear models.  This guess came from looking
#at a map/some initial fits.  The better the guess, the
#faster the convergence.
d2r=np.pi/180
sig=9/2.35/3600*d2r
theta_0=40/3600*d2r
beta_pars=np.asarray([155.91355*d2r,4.1877*d2r,theta_0,0.7,-8.2e-4])
src1_pars=np.asarray([155.9374*d2r,4.1775*d2r,3.1e-5,9.15e-4])
src2_pars=np.asarray([155.90447*d2r,4.1516*d2r,2.6e-5,5.1e-4])

pars=np.hstack([beta_pars,src1_pars,src2_pars])  #we need to combine parameters into a single vector
npar=np.hstack([len(beta_pars),len(src1_pars),len(src2_pars)]) #and the fitter needs to know how many per function
#this array of functions needs to return the model timestreams and derivatives w.r.t. parameters
#of the timestreams.  
funs=[minkasi.derivs_from_isobeta_c,minkasi.derivs_from_gauss_c,minkasi.derivs_from_gauss_c] 
#we can keep some parameters fixed at their input values if so desired.
to_fit=np.ones(len(pars),dtype='bool')
to_fit[3]=False  #Let's keep beta pegged to 0.7

t1=time.time()
pars_fit,chisq,curve,errs=minkasi.fit_timestreams_with_derivs_manyfun(funs,pars,npar,todvec,to_fit)
t2=time.time()
if minkasi.myrank==0:
    print('took ',t2-t1,' seconds to fit timestreams')
    for i in range(len(pars_fit)):
        print('parameter ',i,' is ',pars_fit[i],' with error ',errs[i])

minkasi.comm.barrier()
