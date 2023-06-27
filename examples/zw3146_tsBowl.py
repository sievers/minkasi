#This is a template script to show how to fit multi-component models
#directly to timestreams.  The initial part (where the TODs, noise model etc.)
#are set up is the same as general mapping scripts, although we don't
#need to bother with setting a map/pixellization in general (although if
#your timestream model used a map you might).  This script works under
#MPI as well.


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
tod_names=glob.glob(dir+'Sig*.fits')  
tod_names.sort() #make sure everyone agrees on the order of the file names

tod_names=tod_names[minkasi.myrank::minkasi.nproc]

todvec=minkasi.TodVec()

#loop over each file, and read it.
for fname in tod_names:
    t1=time.time()
    dat=minkasi.read_tod_from_fits(fname)
    t2=time.time()
    minkasi.truncate_tod(dat) #truncate_tod chops samples from the end to make

    dd=minkasi.fit_cm_plus_poly(dat['dat_calib'])  
    
    dat['dat_calib']=dd
    t3=time.time()
    tod=minkasi.Tod(dat)
    todvec.add_tod(tod)
    print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)

minkasi.barrier()


lims=todvec.lims()
pixsize=2.0/3600*numpy.pi/180
map=minkasi.SkyMap(lims,pixsize) 

for tod in todvec.tods:
    ipix=map.get_pix(tod) #don't need pixellization for a timestream fit...
    tod.info['ipix']=ipix
    tod.set_noise_smoothed_svd()
    tod.set_noise(minkasi.NoiseSmoothedSVD)

tsVec = minkasi.tsModel(todvec = todvec, modelclass = minkasi.tsBowl)

#We add two things for pcg to do simulatenously here: make the maps from the tods
#and fit the polynomials to tsVec
mapset.add_map(map)
mapset.add_map(tsVec)

hits=minkasi.make_hits(todvec,map)

rhs=mapset.copy()
todvec.make_rhs(rhs)

x0=rhs.copy()
x0.clear()

#preconditioner is 1/ hit count map.  helps a lot for 
#convergence.
precon=mapset.copy()
tmp=hits.map.copy()
ii=tmp>0
tmp[ii]=1.0/tmp[ii]

precon.maps[0].map[:]=tmp[:]

#tsBowl precon is 1/todlength
if len(mapset.maps) > 1:
    for key in precon.maps[1].data.keys():
        temp = np.ones(precon.maps[1].data[key].params.shape)
        temp /= precon.maps[1].data[key].vecs.shape[1]
        precon.maps[1].data[key].params = temp

mapset_out=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=50)
if minkasi.myrank==0:
    if len(mapset.maps)==1:
        mapset_out.maps[0].write('/scratch/r/rbond/jorlo/noBowl_map_precon_mpi_py3.fits') #and write out the map as a FITS file
    else:
        mapset_out.maps[0].write('/scratch/r/rbond/jorlo/tsBowl_map_precon_mpi_py3.fits')
else:
    print('not writing map on process ',minkasi.myrank)

if minkasi.nproc>1:
    minkasi.MPI.Finalize()

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
    print('No Sub: ')
    print('took ',t2-t1,' seconds to fit timestreams')
    for i in range(len(pars_fit)):
        print('parameter ',i,' is ',pars_fit[i],' with error ',errs[i])

minkasi.comm.barrier()

for tod in todvec.tods():
    todname = tod.info['fname']
    tod.info['dat_calib'] -= np.dot(mapset_out.maps[1].data[fname].params, mapset_out[1].data[fname].vecs.T)   

