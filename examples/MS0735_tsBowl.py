import minkasi.minkasi_all as minkasi
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
from minkasi.tools import presets_by_source as pbs
import os
from astropy.coordinates import Angle
from astropy import units as u

dir = '/mnt/welch/MUSTANG/M2-TODs/MS0735/mustang2/'
outroot = "/mnt/welch/USERS/jorlo/Reductions/MS0735/bowling/"
tod_names=glob.glob(dir+'Sig*.fits')

bad_tod, addtag = pbs.get_bad_tods("MS0735", ndo=False, odo=False)

tod_names = minkasi.cut_blacklist(tod_names, bad_tod)


print("nproc: ", minkasi.nproc)
tod_names = tod_names[minkasi.myrank::minkasi.nproc]

todvec=minkasi.TodVec()

flatten = True

#loop over each file, and read it.
for i, fname in enumerate(tod_names):
    if 'Signal_TOD-AGBT21A_123_03-s20.fits' in fname : continue


    t1=time.time()
    dat=minkasi.read_tod_from_fits(fname)
    t2=time.time()
    minkasi.truncate_tod(dat) #truncate_tod chops samples from the end to make
                              #the length happy for ffts
    minkasi.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
                                  #this fixes that.  You don't need to, though.
    minkasi.truncate_tod(dat)    #since our length changed, make sure we have a happy length
    dd, pred2, cm = minkasi.fit_cm_plus_poly(dat["dat_calib"], cm_ord=3, full_out=True)
   
    dat['dat_calib']=dd
    if flatten: 
        dat['dat_calib'] -= pred2
    t3=time.time()
    tod=minkasi.Tod(dat)
    todvec.add_tod(tod)
    print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)

minkasi.barrier()

lims=todvec.lims()
pixsize=2.0/3600*np.pi/180
map=minkasi.SkyMap(lims,pixsize)

mapset = minkasi.Mapset()

for i, tod in enumerate(todvec.tods):
    ipix=map.get_pix(tod)
    tod.info['ipix']=ipix
    try:
        tod.set_noise(minkasi.NoiseSmoothedSVD)
    except:
        print(i, tod.info['fname'])


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

todvec_copy = minkasi.TodVec()

for tod in todvec.tods:
    todvec_copy.add_tod(tod.copy())

#run PCG!
plot_info={}
plot_info['vmin']=-6e-4
plot_info['vmax']=6e-4
plot_iters=[1,2,3,5,10,15,20,25,30,35,40,45,49]
mapset_out=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=200)

if minkasi.myrank==0:
    if len(mapset.maps)==1:
        mapset_out.maps[0].write(outroot+'_noBowl_initial.fits') #and write out the map as a FITS file
    else:
        mapset_out.maps[0].write(outroot+'_tsBowl_initial.fits')
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
        if len(mapset.maps)==1:
            mapset_out.maps[0].write(
                str(outroot) + "_noBowl_niter_" + repr(niter + 1) + ".fits"
            )
        else:
            mapset_out.maps[0].write(
                str(outroot) + "_tsBowl_niter_" + repr(niter + 1) + ".fits"
            )

minkasi.barrier()

if minkasi.myrank==0:
    print("Maps written to :", outroot)
