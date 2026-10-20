import numpy as np
import minkasi.minkasi_all as minkasi

import time
import glob
import os
#reload(minkasi)

#set file root for output maps
outroot = "/mnt/welch/USERS/jorlo/Reductions/AT2026dbl/AT2026dbl" #CHANGE ME!
#Note the end of this path is a filename, files will be written to
#RXJ1347/RXJ1347_1.fits, RXJ1347/RXJ1347_5.fits, etc. thru RXJ1347/RXJ1347_final.fits


#find tod files we want to map
idir = "/mnt/welch/MUSTANG/M2-TODs/AT2026dbl/" #CHANGE ME
tod_names=glob.glob(idir+'/Sig*.fits')
if len(tod_names)==0:
    print('We found no TOD files.  Double check your path?')
    assert(1==0)

#Use presets by source to automatically get and cut TODs
#that were manually flagged for removal
bad_tod, _ = minkasi.get_bad_tods("Davida")
tod_names = minkasi.cut_blacklist(tod_names, bad_tod)

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
    try:
        dat=minkasi.read_tod_from_fits(fname)
    except:
        print("BAD FILE:", fname)
    t2=time.time()
    minkasi.truncate_tod(dat) #truncate_tod chops samples from the end to make
                              #the length happy for ffts
    minkasi.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
                                  #this fixes that.  You don't need to, though.
    minkasi.truncate_tod(dat)    #since our length changed, make sure we have a happy length

    #figure out a guess at common mode #and (assumed) linear detector drifts/offset
    #drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    if dat["dat_calib"].shape[1] < 100:
        continue

    dd=minkasi.fit_cm_plus_poly(dat['dat_calib'])

    dat['dat_calib']=dd
    t3=time.time()
    tod=minkasi.Tod(dat)
    todvec.add_tod(tod)
    print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)


lims=todvec.lims()
print(lims)
pixsize=2.0/3600*np.pi/180
map=minkasi.SkyMap(lims,pixsize)

#once we have a map, we can figure out the pixellization of the data.  Save that
#so we don't have to recompute.  Also calculate a noise model.  The one here
#(and currently the only supported one) is to rotate the data into SVD space, then
#smooth the power spectrum of each mode.  Other models would not be hard
#to implement.  The smoothing could do with a bit of tuning as well.
for tod in todvec.tods:
    ipix=map.get_pix(tod)
    tod.info['ipix']=ipix
    tod.set_noise(minkasi.NoiseSmoothedSVD)

#get the hit count map.  We use this as a preconditioner
#which helps small-scale convergence quite a bit.
hits=minkasi.make_hits(todvec,map)

#setup the mapset.  In general this can have many things
#in addition to map(s) of the sky, but for now we'll just
#use a single skymap.
mapset=minkasi.Mapset()
mapset.add_map(map)

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
precon=mapset.copy()
tmp=hits.map.copy()
ii=tmp>0
tmp[ii]=1.0/tmp[ii]
precon.maps[0].map[:]=tmp[:]

#run PCG!

save_iters=[1,5,15,20,25,50]

#run PCG!
mapset_out=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=50, save_iters=save_iters, outroot = outroot)

if minkasi.myrank==0:
    mapset_out.maps[0].write(outroot+'_initial.fits') #and write out the map as a FITS file
else:
    print('not writing map on process ',minkasi.myrank)

npass = 5
for niter in range(npass):
    maxiter = 26 + 25 * (niter + 1)
    # first, re-do the noise with the current best-guess map
    for tod in todvec.tods:
        mat = 0 * tod.info["dat_calib"]
        for mm in mapset_out.maps:
            mm.map2tod(tod, mat)
        tod.set_noise(
            minkasi.NoiseSmoothedSVD, dat=tod.info["dat_calib"] - mat, #note here we are feeding tod.info["dat_calib] - mat to dat, as opposed to
                                                                       #say copying tod.info["dat_calib"], subtracting mat, and passing it.
                                                                       #This is the safest way to avoid changing the values stored in tod.info
        )

    priorset = None

    rhs = mapset.copy()
    todvec.make_rhs(rhs)

    precon = mapset.copy() #Hist map preconditioner
    precon.maps[0].map[:] = hits.map[:]
    mapset_out = minkasi.run_pcg_wprior(
        rhs,
        mapset_out,
        todvec,
        priorset,
        precon,
        maxiter=maxiter,
        outroot= str(outroot) + "_niter_" + repr(niter + 1),
        save_iters=save_iters,
    )
    if minkasi.myrank == 0:
        mapset_out.maps[0].write(
            str(outroot) + "_niter_" + repr(niter + 1) + ".fits"
        )

minkasi.barrier()

if minkasi.myrank==0:
    print("Maps written to :", outroot)

#we need an initial guess since this fitting routine is
#for nonlinear models.  This guess came from looking
#at a map/some initial fits.  The better the guess, the
#faster the convergence.
d2r=np.pi/180
sig=9/2.35/3600*d2r
x0 = np.deg2rad(135.322392)
y0 = np.deg2rad(18.602147)
pars = np.asarray([x0, y0, sig, 1.])
npar = np.asarray([len(pars)])

#this array of functions needs to return the model timestreams and derivatives w.r.t. parameters
#of the timestreams.  
funs=[minkasi.derivs_from_gauss_c]
#we can keep some parameters fixed at their input values if so desired.
to_fit=np.ones(len(pars),dtype='bool')
to_fit[2] = False
print("Initial pars: ", pars)
t1=time.time()
pars_fit,chisq,curve,errs=minkasi.fit_timestreams_with_derivs_manyfun(funs,pars,npar,todvec,to_fit)
t2=time.time()
if minkasi.myrank==0:
    print('took ',t2-t1,' seconds to fit timestreams')
    for i in range(len(pars_fit)):
        print('parameter ',i,' is ',pars_fit[i],' with error ',errs[i])

minkasi.comm.barrier()
