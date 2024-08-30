import glob
import time

import minkasi.minkasi_all as minkasi
import numpy as np

# NOTE: Most comments not relevant to doing cuts removed, see minkasi_mpi_example.py for them

# set file root for output maps
outroot = "./RXJ1347"  # CHANGE ME!

# find tod files we want to map
idir = "/mnt/welch/MUSTANG/M2-TODs/RXJ1347/"  # CHANGE ME
tod_names = glob.glob(idir + "/Sig*.fits")
if len(tod_names) == 0:
    print("We found no TOD files.  Double check your path?")
    assert 1 == 0

bad_tod, _ = minkasi.get_bad_tods("RXJ1347")
tod_names = minkasi.cut_blacklist(tod_names, bad_tod)

tod_names = tod_names[minkasi.myrank :: minkasi.nproc]


todvec = minkasi.TodVec()

# loop over each file, and read it.
for fname in tod_names:
    t1 = time.time()
    dat = minkasi.read_tod_from_fits(fname)
    t2 = time.time()
    minkasi.truncate_tod(dat)
    minkasi.downsample_tod(dat)
    minkasi.truncate_tod(dat)
    dd = minkasi.fit_cm_plus_poly(dat["dat_calib"])
    dat["dat_calib"] = dd
    t3 = time.time()
    tod = minkasi.Tod(dat)
    todvec.add_tod(tod)
    print(
        "took ", t2 - t1, " ", t3 - t2, " seconds to read and downsample file ", fname
    )

# Now lets setup cuts
# We should really streamline this...
cuts_model = minkasi.tsModel(todvec, minkasi.CutsCompact)
# Lets also add some cuts
# cuts_model.data is a dict that maps tod.info["fname"] to a CutsCompact
# Lets say we want to flag some samples in the first det of the TOD
cuts_model.data[todvec.tods[0].info["fname"]].add_cut(0, 100, 150)
# We may want to fill the cuts so that the noise model works better
# This is optional but can help with glitches
for tod, (fname, cut) in zip(
    todvec.tods, cuts_model.data.items()
):  # Using the fact that dict entries are ordered in modern python here
    if np.sum(cut.nseg) == 0:
        continue
    new_cut = minkasi.gapfill_eig(tod.get_data(), cut, tod, insert_cuts=True)
    cuts_model.data[fname] = new_cut

lims = todvec.lims()
if lims is None:
    raise ValueError("lims is None!")
pixsize = 2.0 / 3600 * np.pi / 180
map = minkasi.SkyMap(lims, pixsize)

for tod in todvec.tods:
    ipix = map.get_pix(tod)
    tod.info["ipix"] = ipix
    tod.set_noise(minkasi.NoiseSmoothedSVD)

hits = minkasi.make_hits(todvec, map)

mapset = minkasi.Mapset()
mapset.add_map(map)
# We also add the cuts to the mapset because we want to solve them
mapset.add_map(cuts_model)

rhs = mapset.copy()
todvec.make_rhs(rhs)
x0 = rhs.copy()
x0.clear()
precon = mapset.copy()
tmp = hits.map.copy()
ii = tmp > 0
tmp[ii] = 1.0 / tmp[ii]
precon.maps[0].map[:] = tmp[:]
save_iters = [1, 5, 15, 20, 25, 50]
mapset_out = minkasi.run_pcg(
    rhs, x0, todvec, precon, maxiter=50, save_iters=save_iters, outroot=outroot
)

if minkasi.myrank == 0:
    mapset_out.maps[0].write(
        outroot + "_initial.fits"
    )  # and write out the map as a FITS file
else:
    print("not writing map on process ", minkasi.myrank)

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
