import numpy as np
from matplotlib import pyplot as plt

import minkasi.minkasi_all as minkasi
from minkasi.needlet.needlet import WavSkyMap
from minkasi.needlet.needlet import Needlet, cosmo_box
from minkasi.needlet.needlet import wav2map_real, map2wav_real
from minkasi.maps.mapset import PriorMapset, Mapset
from minkasi.maps.skymap import SkyMap
from minkasi.parallel import comm, get_nthread, have_mpi, nproc, myrank, MPI

import time, glob, os
import pdb

from pixell import enmap, utils

import astropy.units as u

import dill as pk

plt.ion()

#find tod files we want to map

name = "MOOJ1142"
idir = "/mnt/welch/MUSTANG/M2-TODs/{}/".format(name) #CHANGE ME
tod_names=glob.glob(idir+'Sig*.fits')
tod_names=tod_names[minkasi.myrank::minkasi.nproc]
todvec=minkasi.TodVec()

ntods = 10 
#loop over each file, and read it.
for i, fname in enumerate(tod_names):
    if i >= ntods: break
    t1=time.time()
    dat=minkasi.read_tod_from_fits(fname)
    t2=time.time()
    minkasi.truncate_tod(dat) #truncate_tod chops samples from the end to make
    minkasi.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
    minkasi.truncate_tod(dat)    #since our length changed, make sure we have a happy length

    dd=minkasi.fit_cm_plus_poly(dat['dat_calib'])

    dat['dat_calib']=dd
    t3=time.time()
    tod=minkasi.Tod(dat)
    todvec.add_tod(tod)
    print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)

lims=todvec.lims()
pixsize=3.0/3600*np.pi/180

#wmap = SkyMap(lims, pixsize, primes = [2,3,5,7], square = True).map #Really shitty way to get the right map geometry for making filters
wmap = SkyMap(lims, pixsize, square = True, multiple=2).map
need = Needlet(np.arange(10), lightcone=wmap, L=10*60*np.sqrt(2), pixsize = pixsize * (3600 * 180) / np.pi)
fourier_radii = need.lightcone_box.get_grid_dimless_2d(return_grid=True)
need.get_needlet_filters_2d(fourier_radii)

#wmap = WavSkyMap(need, lims, pixsize, primes = [2,3,5,7], square = True)
wmap = WavSkyMap(need, lims, pixsize, square = True, multiple=2)

for tod in todvec.tods:
    ipix=wmap.get_pix(tod)
    tod.info['ipix']=ipix
    tod.set_noise(minkasi.NoiseSmoothedSVD)

hits=minkasi.make_hits(todvec,wmap.real_map)

#Run a few iterations to get small scales to converge, subtract off so we only need to do larger scales
mapset=Mapset()
mapset.add_map(wmap)

rhs=mapset.copy()
todvec.make_rhs(rhs)

#this is our starting guess.  Default to starting at 0,
x0=rhs.copy()
x0.clear()

#preconditioner is 1/ hit count map.  helps a lot for
precon=mapset.copy()
tmp=hits.map.copy()
ii=tmp>0
tmp[ii]=1.0/tmp[ii]
precon.maps[0].map[:]=np.sqrt(tmp)
precon.maps[0].map[:]=tmp[:]

#outroot = '/scratch/r/rbond/jorlo/MS0735/needlets/needle'
#outroot = '/scratch/r/rbond/jorlo/A399-401'
outroot = "/mnt/welch/USERS/jorlo/needlets/{}".format(name)
print("Saving to ", outroot)
save_iters = [1,5,10,15,20,25, 50, 100, 150,200,250,300,350,400,450, 499]
#run PCG!
mapset_out=minkasi.run_pcg_wprior(rhs,x0,todvec,maxiter=50, save_iters=save_iters, outroot = outroot)

if minkasi.myrank==0:
    mapset_out.maps[0].write(outroot+"_need.fits")
minkasi.barrier()
#Subtract off modes outside the joint ACT+M2 window


for tod in todvec.tods:
    tmp = np.zeros(tod.info["dat_calib"].shape)
    mapset_out.maps[0].map2tod(tod, tmp)
    tod.info["dat_calib"] -= tmp
    tod.set_noise(minkasi.NoiseSmoothedSVD)


if os.path.exists("{}_response.pk".format(name)):
    with open("{}_response.pk".format(name), "rb") as f:
        response_matrix = pk.load(f)
else:
    response_matrix = wmap.get_response_matrix(todvec, max_res = 9.0)
    with open("{}_response.pk".format(name), "wb") as f:
        pk.dump(response_matrix, f)


#Read in ACT data and noise
root = "/mnt/welch/USERS/jorlo/maps/dr6_20230316_splits/cmb_night_*_f090_3pass_4way_set*_ivar.fits"
paths = glob.glob(root)

ra = np.mean([lims[0],lims[1]]); dec = np.mean([lims[2], lims[3]])
rad = 0.5*utils.degree
point = np.array([dec, ra])
box = np.array([point - rad, point + rad])

ivar_maps = []
for path in paths:
    ivar_map = enmap.read_map(path, box=box)
    two_res = minkasi.SkyMapTwoRes(np.array(ivar_map), lims, 10, big_wcs=ivar_map.wcs)
    two_res.set_beam_gauss(3)
    two_res.set_mask(hits.map)

    ivar_map = two_res.coarse2fine(wmap.real_map.map, ivar_map)
  
    ivar_maps.append(ivar_map.ravel())
ivar_maps = np.array(ivar_maps)

#Unit conversions from uK-arcmin to uK_RJ that are likely very wrong
#beam_fwhm = 2.2*u.arcmin
#fwhm_to_sigma = 1. / (8 * np.log(2))**0.5
#beam_sigma = beam_fwhm * fwhm_to_sigma
#omega_B_ACT = 2 * np.pi * beam_sigma**2
#print(omega_B_ACT)


#big_map /= (np.sqrt(omega_B_ACT.value) * 1.23)

cov_map = np.cov(ivar_maps, rowvar=False)

if os.path.exists("{}_act_response.pk".format(name)):
    with open("{}_act_response.pk".format(name), "rb") as f:
        act_response_matrix = pk.load(f)
else:
    act_response_matrix = wmap.get_response_matrix_map(cov_map, max_res=60.0)
    with open("{}_act_response.pk".format(name), "wb") as f:
        pk.dump(act_response_matrix, f)

for i in range(len(response_matrix)):
    cur = response_matrix[i] + act_response_matrix[i]
    #TODO: this should block expand to nx*ny by nx*ny, not nx by ny
    cur = astropy.nddata.block_replicate(cur, wmap.map.shape[1]/cur.shape[1]) #Conserve_sum = False?



