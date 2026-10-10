import numpy as np
#from matplotlib import pyplot as plt
import minkasi
import time
import glob
from importlib import reload
reload(minkasi)
#plt.ion()

mydir = '/home/scratch/cromero/mustang/MUSTANG2/Reductions/MOO_1142/'
tod_names=glob.glob(mydir+'TS_EaCMS0f0_51_18_Jun_2025/Signal_TOD*.fits')

### Fit initial features as Gaussians; notably with the locations, amplitude, and widths free.
maxiter = 25
ntods = 106  # maximum number is 106
mis = repr(maxiter)
nts = repr(ntods)
Gaussian_file = mydir+'/MinkasiOutputs/moo_1142_lowestChi_Gaussians_16July2026_v0_'+mis+'iters_'+nts+'scans.npy'
Curve_file = mydir+'/MinkasiOutputs/moo_1142_lowestChi_Curve_16July2026_v0_'+mis+'iters_'+nts+'scans.npy'
Gaussian_file_at_end = mydir+'/MinkasiOutputs/moo_1142_ending_Gaussians_16July2026_v0_'+mis+'iters_'+nts+'scans.npy'
Curve_file_at_end = mydir+'/MinkasiOutputs/moo_1142_ending_Curve_16July2026_v0_'+mis+'iters_'+nts+'scans.npy'

mcl_ra = 175.69766  # RA of the main cluster
mcl_dec = 15.453600
mcl_sigma = 1.0e-04  # Radians ~ 20 arcseconds
mcl_depth = -6.0e-4  # Roughly 600 mK_RJ decrement?
wcl_ra = 175.68840   # RA of the west (sub)cluster
wcl_dec = 15.455600
wcl_sigma = 2.7e-5  # Radians ~ 5 arcseconds
wcl_depth = -1.0e-4  # Roughly 100 mK_RJ decrement
#########################################################
### Point sources
bcg_ra = 175.69784  # Ra of the BCG
bcg_dec = 15.45318
field_ra = 175.61436  # RA of a field galaxy
field_dec = 15.507738
pt_sigma = 2.7e-5  # ~ 5 arcseconds (10 arcseconds FWHM)
bcg_amp = 1.0e-4  # mK peak, maybe.
field_amp = 1.8e-3  # 1.8 mK peak ~ a bit above 2 mJy.
d2r = np.pi/180

np_src = 4  # 4 parameters per Gaussian
n_gauss = 4  # 4 Gaussians: 2 clusters, 2 point sources
gaussp = np.zeros(np_src*n_gauss)  # Array for all parameters
gaussp[:np_src] = [mcl_ra*d2r, mcl_dec*d2r,  mcl_sigma, mcl_depth]
gaussp[np_src:2*np_src] = [wcl_ra*d2r, wcl_dec*d2r,  wcl_sigma, wcl_depth]
gaussp[np_src*2:3*np_src] = [bcg_ra*d2r, bcg_dec*d2r,  pt_sigma, bcg_amp]
gaussp[np_src*3:4*np_src] = [field_ra*d2r, field_dec*d2r,  pt_sigma, field_amp]

npar=np.hstack([np_src]*n_gauss) #and the fitter needs to know how many per function
# Leave this trail of comments to show my confusion and...shit doesn't work guys.
#funs = [minkasi.derivs_from_gauss_c]*4
#funs = [minkasi.lib.minkasi.fill_gauss_derivs_c]*4
funs = [minkasi.fitting.core.derivs_from_gauss_c]*4
to_fit=np.ones(np_src*n_gauss,dtype='bool')

use_Gaussian_center = False
initial_map = mydir+"MinkasiOutputs/Minkasi_naivePCG_map.fits"

##############################################################################################

tod_names=tod_names[minkasi.myrank::minkasi.nproc]
todvec=minkasi.tods.TodVec()

for fname in tod_names[:ntods]:
    t1=time.time()
    dat=minkasi.tods.io.read_tod_from_fits(fname)
    t2=time.time()
    ### Downsample is optional. Ideally MUSTANG-2 TODs are appropriately downsampled.
    #minkasi.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
    minkasi.tods.processing.truncate_tod(dat)    #since our length changed, make sure we have a happy length

    #figure out a guess at common mode #and (assumed) linear detector drifts/offset
    #drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    dd=minkasi.tods.processing.fit_cm_plus_poly(dat['dat_calib'])  
    
    dat['dat_calib']=dd
    t3=time.time()
    tod=minkasi.tods.Tod(dat)
    todvec.add_tod(tod)
    print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)

#make a template map with desired pixel size an limits that cover the data
#todvec.lims() is MPI-aware and will return global limits, not just
#the ones from private TODs
lims=todvec.lims()
pixsize=2.0/3600*np.pi/180  # 2 arcseconds, stored as radians
map=minkasi.maps.SkyMap(lims,pixsize)

#once we have a map, we can figure out the pixellization of the data.  Save that
#so we don't have to recompute.  Also calculate a noise model.  The one here
#(and currently the only supported one) is to rotate the data into SVD space, then
#smooth the power spectrum of each mode.  Other models would not be hard
#to implement.  The smoothing could do with a bit of tuning as well.


for tod in todvec.tods:
    ipix=map.get_pix(tod)
    tod.info['ipix']=ipix
    #tod.set_noise_smoothed_svd()
    tod.set_noise(minkasi.mapmaking.noise.NoiseSmoothedSVD)


t1=time.time()
pars_fit,chisq,curve,errs=minkasi.fitting.fit_timestreams_with_derivs_manyfun(funs,gaussp,npar,todvec,to_fit, maxiter=maxiter)
t2=time.time()

print(pars_fit)
np.save(Gaussian_file,pars_fit)
np.save(Curve_file,curve)
       
"""    
for iter in range(20):
    chisq=0.0
    curve=0.0
    deriv=0.0
    for tt in range(todvec.ntod):
        aa,bb,mychisq=minkasi.get_curve_deriv_tod_manygauss(gaussp,todvec.tods[tt])
        curve=curve+aa
        deriv=deriv+bb
        chisq=chisq+mychisq
    chisq=minkasi.comm.allreduce(chisq)
    curve=minkasi.comm.allreduce(curve)
    deriv=minkasi.comm.allreduce(deriv)
    if iter==0:
        chi_ref=chisq
        chi_min=chisq
    shift=np.dot(np.linalg.inv(curve),deriv)
    #shift2=np.matmul(np.linalg.inv(curve),deriv)
    #gaussp=gaussp+0.9*shift # original line
    gaussp=gaussp+0.5*shift # 2 Feb 2019. Does this behave like I expect (CR)?
    if minkasi.myrank==0:
        print('shift is \n',shift,'\n ----------------------------------------------- \n',chisq-chi_ref)
        #print('shift is \n',shift,'\n',shift2,'\n',gaussp,'\n',chisq-chi_ref)
        
    if chisq < chi_min:
       np.save(Gaussian_file,gaussp)
       np.save(Curve_file,curve)
    #assert(1==0)

if minkasi.myrank==0:
    np.save(Gaussian_file_at_end,gaussp)
    np.save(Curve_file_at_end,curve)

chisq=minkasi.comm.allreduce(chisq) #random mpi call to make sure rank 0 has time to write before the assert
"""
