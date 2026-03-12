import numpy as np
import minkasi.minkasi_all as minkasi

from copy import deepcopy

import time
import glob
import os

# reload(minkasi)

# set file root for output maps

do_cmsub = False
if do_cmsub:
    participle = "w"
else:
    participle = "wo"

outroot = "/mnt/welch/USERS/jorlo/Reductions/id34/id34_{}_cm".format(participle)  # CHANGE ME!
# Note the end of this path is a filename, files will be written to
# RXJ1347/RXJ1347_1.fits, RXJ1347/RXJ1347_5.fits, etc. thru RXJ1347/RXJ1347_final.fits


# find tod files we want to map
idir = "/mnt/welch/NIKA2/id34/"  # CHANGE ME
tod_names = glob.glob(idir + "/stru_scan*.xdr")
if len(tod_names) == 0:
    print("We found no TOD files.  Double check your path?")
    assert 1 == 0

# Use presets by source to automatically get and cut TODs
# that were manually flagged for removal
bad_tod, _ = minkasi.get_bad_tods("id34")
tod_names = minkasi.cut_blacklist(tod_names, bad_tod)

# if running MPI, you would want to split up files between processes
# one easy way is to say to this:
tod_names = tod_names[minkasi.myrank :: minkasi.nproc]
# NB - minkasi checks to see if MPI is around, if not
# it sets rank to 0 an nproc to 1, so this would still
# run in a non-MPI environment

do_subscans = False


todvec = minkasi.TodVec()

# loop over each file, and read it.
for fname in tod_names:
    t1 = time.time()
    dat = minkasi.read_tod_from_fits_NIKA2(fname)
    if dat is None:
        continue
    t2 = time.time()

    if do_cmsub:
        dd = minkasi.fit_cm_plus_poly(dat["dat_calib"])
        dat["dat_calib"] = dd
   
    if do_subscans:
        turnarounds = [0, *dat["turnarounds"], -1]
    
        for i in range(len(turnarounds)-1):
            cur_dat = deepcopy(dat)
            for key in cur_dat.keys():
                try:
                    cur_dat[key] = cur_dat[key][:, turnarounds[i]:turnarounds[i+1]]
                except:
                    try:
                        if len(cur_dat[key]) > turnarounds[-2]: #Check to see if this is an array that is of length nsamp, not ndet
                            cur_dat[key] = cur_dat[key][turnarounds[i]:turnarounds[i+1]]
                    except:
                        continue
    
            minkasi.truncate_tod(cur_dat)  # truncate_tod chops samples from the end to make
            # the length happy for ffts
    
            tod = minkasi.Tod(cur_dat)
            todvec.add_tod(tod)

    else:
        minkasi.truncate_tod(dat) #truncate_tod chops samples from the end to make
                                      #the length happy for ffts
        minkasi.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
                                      #this fixes that.  You don't need to, though.
        minkasi.truncate_tod(dat)    #since our length changed, make sure we have a happy length
        tod=minkasi.Tod(dat)
        todvec.add_tod(tod)
    t3 = time.time()
    print(
        "took ", t2 - t1, " ", t3 - t2, " seconds to read and downsample file ", fname
    )

# Apply cuts and gapfill
cuts_model = minkasi.tsModel(todvec, minkasi.CutsCompact)
for i, tod in enumerate(todvec.tods):
    cuts = tod.info["cuts"]
    if len(cuts) == 0:
        continue
    for key in cuts.keys():
        for j in range(len(cuts[key])):
            cuts_model.data[todvec.tods[i].info["fname"]].add_cut(
                key, cuts[key][j][0], cuts[key][j][1]
            )
# We may want to fill the cuts so that the noise model works better
# This is optional but can help with glitches
for tod, (fname, cut) in zip(
    todvec.tods, cuts_model.data.items()
):  # Using the fact that dict entries are ordered in modern python here
    if np.sum(cut.nseg) == 0:
        continue
    new_cut = minkasi.gapfill_eig(tod.get_data(), cut, tod, insert_cuts=True)
    cuts_model.data[fname] = new_cut

# make a template map with desired pixel size an limits that cover the data
# todvec.lims() is MPI-aware and will return global limits, not just
# the ones from private TODs
lims = todvec.lims()
pixsize = 2.0 / 3600 * np.pi / 180
map = minkasi.SkyMap(lims, pixsize)

# once we have a map, we can figure out the pixellization of the data.  Save that
# so we don't have to recompute.  Also calculate a noise model.  The one here
# (and currently the only supported one) is to rotate the data into SVD space, then
# smooth the power spectrum of each mode.  Other models would not be hard
# to implement.  The smoothing could do with a bit of tuning as well.
for tod in todvec.tods:
    ipix = map.get_pix(tod)
    tod.info["ipix"] = ipix
    tod.set_noise(minkasi.NoiseSmoothedSVD)

# get the hit count map.  We use this as a preconditioner
# which helps small-scale convergence quite a bit.
hits = minkasi.make_hits(todvec, map)

# setup the mapset.  In general this can have many things
# in addition to map(s) of the sky, but for now we'll just
# use a single skymap.
mapset = minkasi.Mapset()
mapset.add_map(map)
# We also add the cuts to the mapset because we want to solve them
mapset.add_map(cuts_model)

# make A^T N^1 d.  TODs need to understand what to do with maps
# but maps don't necessarily need to understand what to do with TODs,
# hence putting make_rhs in the vector of TODs.
# Again, make_rhs is MPI-aware, so this should do the right thing
# if you run with many processes.

rhs = mapset.copy()
todvec.make_rhs(rhs)

# this is our starting guess.  Default to starting at 0,
# but you could start with a better guess if you have one.
x0 = rhs.copy()
x0.clear()

# preconditioner is 1/ hit count map.  helps a lot for
# convergence.
precon = mapset.copy()
tmp = hits.map.copy()
ii = tmp > 0
tmp[ii] = 1.0 / tmp[ii]
precon.maps[0].map[:] = tmp[:]

# run PCG!

save_iters = [1, 5, 15, 20, 25, 50]

# run PCG!
mapset_out = minkasi.run_pcg(
    rhs, x0, todvec, precon, maxiter=50, save_iters=save_iters, outroot=outroot
)

if minkasi.myrank == 0:
    mapset_out.maps[0].write(
        outroot + "_initial.fits"
    )  # and write out the map as a FITS file
else:
    print("not writing map on process ", minkasi.myrank)

# This first pass of fitting is a reasonable approximation. However, the SVD noise estimation we use invariable includes some signal in the noise model.
# In other words, our noise model assumes our TODs are 100% noise, performs an SVD, and uses the first N modes of that SVD as the noise estime. However,
# some portion of those modes are actually signal. To counteract this, we take the initial best-fit map (which is our best estimate of the signal),
# subtract it from a copy of the TODs, and reestimate the noise. Subtracting the best-fit map should remove the largest signal modes, making our estimate of the
# noise cleaner. We can then remake the map and itterate on this process. Note the best-fit map is not subtracted from the TODs themselves for the purposes
# of mapmaking. That would remove signal from the final map. It is only removed from a copy of the TODs (technically a copy of the data in the TODs) used
# in noise estimation.

npass = 5
for niter in range(npass):
    maxiter = 26 + 25 * (niter + 1)
    # first, re-do the noise with the current best-guess map
    for tod in todvec.tods:
        mat = 0 * tod.info["dat_calib"]
        for mm in mapset_out.maps:
            mm.map2tod(tod, mat)
        tod.set_noise(
            minkasi.NoiseSmoothedSVD,
            dat=tod.info["dat_calib"]
            - mat,  # note here we are feeding tod.info["dat_calib] - mat to dat, as opposed to
            # say copying tod.info["dat_calib"], subtracting mat, and passing it.
            # This is the safest way to avoid changing the values stored in tod.info
        )

    priorset = None

    rhs = mapset.copy()
    todvec.make_rhs(rhs)

    precon = mapset.copy()  # Hist map preconditioner
    precon.maps[0].map[:] = hits.map[:]
    mapset_out = minkasi.run_pcg_wprior(
        rhs,
        mapset_out,
        todvec,
        priorset,
        precon,
        maxiter=maxiter,
        outroot=str(outroot) + "_niter_" + repr(niter + 1),
        save_iters=save_iters,
    )
    if minkasi.myrank == 0:
        mapset_out.maps[0].write(str(outroot) + "_niter_" + repr(niter + 1) + ".fits")

minkasi.barrier()

if minkasi.myrank == 0:
    print("Maps written to :", outroot)
