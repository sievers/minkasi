import minkasi
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
import minkasi_jax.presets_by_source as pbs

dir = '/scratch/r/rbond/jorlo/MS0735//TS_EaCMS0f0_51_5_Oct_2021/'
tod_names=glob.glob(dir+'Sig*.fits')

bad_tod, addtag = pbs.get_bad_tods("MS0735", ndo=False, odo=False)
tod_names = minkasi.cut_blacklist(tod_names, bad_tod)
tod_names.sort()
print(minkasi.nproc)
tod_names = tod_names[minkasi.myrank::minkasi.nproc]

todvec=minkasi.TodVec()

#loop over each file, and read it.
for i, fname in enumerate(tod_names):
    if i ==194: continue #This one does't make noise right, gotta figure out why that is
    t1=time.time()
    dat=minkasi.read_tod_from_fits(fname)
    t2=time.time()
    minkasi.truncate_tod(dat) #truncate_tod chops samples from the end to make
                              #the length happy for ffts
    minkasi.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
                                  #this fixes that.  You don't need to, though.
    minkasi.truncate_tod(dat)    #since our length changed, make sure we have a happy length

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
pixsize=2.0/3600*np.pi/180
map=minkasi.SkyMap(lims,pixsize)

#once we have a map, we can figure out the pixellization of the data.  Save that
#so we don't have to recompute.  Also calculate a noise model.  The one here
#(and currently the only supported one) is to rotate the data into SVD space, then
#smooth the power spectrum of each mode.  Other models would not be hard
#to implement.  The smoothing could do with a bit of tuning as well.

mapset = minkasi.Mapset()

for i, tod in enumerate(todvec.tods):
    print(i)

    ipix=map.get_pix(tod)
    tod.info['ipix']=ipix
    #tod.set_noise_smoothed_svd()
    tod.set_noise(minkasi.NoiseSmoothedSVD)

    #tsBowl = minkasi.tsBowl(tod)
    #mapset.add_map(tsBowl)

tsVec = minkasi.tsModel(todvec = todvec, modelclass = minkasi.tsBowl)
mapset.add_map(tsVec)

#make A^T N^1 d.  TODs need to understand what to do with maps
#but maps don't necessarily need to understand what to do with TODs, 
#hence putting make_rhs in the vector of TODs. 
#Again, make_rhs is MPI-aware, so this should do the right thing
#if you run with many processes.
rhs=mapset.copy()
todvec.make_rhs(rhs)

#this is our starting guess.  Default to starting at 0,
#but you could start with a better guess if you have one.
x0=rhs.copy()
x0.clear()

#preconditioner is 1/ hit count map.  helps a lot for 
#convergence.
#precon=mapset.copy()
#tmp=hits.map.copy()
#ii=tmp>0
#tmp[ii]=1.0/tmp[ii]

#precon.maps[0].map[:]=tmp[:]

#run PCG!
plot_info={}
plot_info['vmin']=-6e-4
plot_info['vmax']=6e-4
plot_iters=[1,2,3,5,10,15,20,25,30,35,40,45,49]

mapset_out=minkasi.run_pcg(rhs,x0,todvec,maxiter=50)
if minkasi.myrank==0:
    mapset_out.maps[0].write('/scratch/r/rbond/jorlo/first_map_precon_mpi_py3.fits') #and write out the map as a FITS file
else:
    print('not writing map on process ',minkasi.myrank)

if minkasi.nproc>1:
    minkasi.MPI.Finalize()

#if you wanted to run another round of PCG starting from the previous solution, 
#you could, replacing x0 with mapset_out.  
#mapset_out2=minkasi.run_pcg(rhs,mapset_out,todvec,mapset,maxiter=50)
#mapset_out2.maps[0].write('second_map.fits')



