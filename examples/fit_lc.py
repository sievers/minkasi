import numpy as np
import minkasi.minkasi_all as minkasi

from astropy.io import fits

from datetime import datetime
import time
import glob
import os
import dill as pk
#reload(minkasi)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("obs_id")

args = parser.parse_args()

obs_id = args.obs_id

#set file root for output maps
outroot = "/mnt/welch/USERS/jorlo/Reductions/Davida_{}".format(obs_id) #CHANGE ME!
#Note the end of this path is a filename, files will be written to
#RXJ1347/RXJ1347_1.fits, RXJ1347/RXJ1347_5.fits, etc. thru RXJ1347/RXJ1347_final.fits


#find tod files we want to map
idir = "/mnt/welch/MUSTANG/M2-TODs/Davida_{}/".format(obs_id) #CHANGE ME
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

if len(tod_names) > 1:
    print("Error: lightcurve fitting should only be performed on single TODs")
    assert(1==0)

todvec=minkasi.TodVec()

#loop over each file, and read it.
for fname in tod_names:
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
    gain = dat["calinfo"]["antgain"]
    dd /= gain
    dat['dat_calib']=dd
    t3=time.time()
    tod=minkasi.Tod(dat)
    todvec.add_tod(tod)
    print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)


lims=todvec.lims()
pixsize=2.0/3600*np.pi/180
map=minkasi.SkyMap(lims,pixsize)


for tod in todvec.tods:
    ipix=map.get_pix(tod)
    tod.info['ipix']=ipix
    tod.set_noise(minkasi.NoiseSmoothedSVD)

hits=minkasi.make_hits(todvec,map)
mapset=minkasi.Mapset()
mapset.add_map(map)
rhs=mapset.copy()
todvec.make_rhs(rhs)
x0=rhs.copy()
x0.clear()
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

hdul = fits.open(fname) #Abusive but there should only be one TOD open
start_time = hdul[1].header["OBSSTART"]
end_time = hdul[1].header["OBSSTOP"]
format_data = "%d-%b-%Y %H:%M:%S.%f"
start_time = datetime.strptime(start_time, format_data)
end_time = datetime.strptime(end_time, format_data)

tbins = 2 * 60 #tbin in seconds
samp_bins = int(tbins / tod.info["dt"])

d2r=np.pi/180
sig=np.mean([todvec.tods[0].info["calinfo"]["bmaj"], todvec.tods[0].info["calinfo"]["bmin"]])/2.35/3600*d2r
lc_bins = np.arange(0, tod.get_data_dims()[-1], samp_bins).tolist()

x0 = np.deg2rad(36.3388645)
y0 = np.deg2rad(-5.5947821)
#x0, y0 = 6.38262592e-01, -12.76932687e-02

pars = np.asarray(len(lc_bins) * [x0, y0, sig, 0.005])
npar = np.asarray(len(lc_bins) * [4])

#this array of functions needs to return the model timestreams and derivatives w.r.t. parameters
#of the timestreams.  
funs=len(lc_bins) * [minkasi.lc_derivs_from_gauss]
#we can keep some parameters fixed at their input values if so desired.
to_fit=np.ones(len(pars),dtype='bool')
#to_fit[2]=False
#to_fit[6]=False

pos_offset = 100/3600*d2r
#priors = ["flat", "flat", "flat", None] 
#prior_vals = [[x0-pos_offset, x0+pos_offset],[y0-pos_offset, y0+pos_offset], [sig/2, sig*2], []]

priors = len(lc_bins)*[None, None, "flat", None]
prior_vals = len(lc_bins)*[[],[], [-100, 100], []]

t1=time.time()
pars_fit,chisq,curve,errs=minkasi.fit_timestreams_with_derivs_manyfun(funs,pars,npar,todvec,to_fit,maxiter=100, priors=priors, prior_vals=prior_vals,lc_bins=lc_bins)
t2=time.time()
if minkasi.myrank==0:
    print('took ',t2-t1,' seconds to fit timestreams')
    for i in range(len(pars_fit)):
        print('parameter ',i,' is ',pars_fit[i],' with error ',errs[i])

minkasi.comm.barrier()

lc_bins.append(tod.get_data_dims()[-1])
lc_bins = np.array(lc_bins, dtype=float)
lc_bins *= tod.info["dt"]
lc_bins += start_time.timestamp()
amps, amp_errs = pars_fit[3::4], errs[3::4]
sigs, sig_errs = pars_fit[2::4], errs[2::4]
x0s, x0_errs = pars_fit[::4], errs[::4]
y0s, y0_errs = pars_fit[1::4], errs[1::4]

m2_dict = {"bin_edges":lc_bins, "amps":amps, "amp_errs":amp_errs, "beam_fwhm":sigs, "beam_err":sig_errs, "ra":x0s, "ra_errs":x0_errs, "dec":y0s, "dec_errs":y0_errs}
with open(outroot + "/M2_results_{}.pk".format(obs_id), "wb") as f:
    print("Wrote to ", outroot+ "/M2_results_{}.pk".format(obs_id))
    pk.dump(m2_dict, f)

if minkasi.myrank==0:
    print("Maps written to :", outroot)

