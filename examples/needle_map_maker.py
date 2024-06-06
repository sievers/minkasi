import numpy as np
from matplotlib import pyplot as plt
import glob
import time

import os
from astropy.coordinates import Angle
from astropy import units as u
from astropy.io import fits

import minkasi.minkasi_all as minkasi
from minkasi.maps.skymap import SkyMap
from minkasi.needlet.needlet import WavSkyMap
from minkasi.needlet.needlet import Needlet, cosmo_box
from minkasi.needlet.needlet import wav2map_real, map2wav_real
from minkasi.maps.mapset import PriorMapset, Mapset

#%load_ext autoreload
#%autoreload 2

#find tod files we want to map
#idir = "/scratch/r/rbond/jorlo/M2-TODs/RXJ1347/" #CHANGE ME
idir = "/mnt/welch/MUSTANG/M2-TODs/RXJ1347/"
#idir = "/scratch/r/rbond/jorlo/M2-TODs/A399-401/"
tod_names=glob.glob(idir+'Sig*.fits')


n_tods = 999999

tod_names = tod_names[:n_tods]
tod_names=tod_names[minkasi.myrank::minkasi.nproc]

todvec=minkasi.TodVec()

flatten = False


#loop over each file, and read it.
for i, fname in enumerate(tod_names):
    if i > n_tods: break
    if fname == '/scratch/r/rbond/jorlo/MS0735//TS_EaCMS0f0_51_5_Oct_2021/Signal_TOD-AGBT21A_123_03-s20.fits': continue


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

template_map = SkyMap(lims, pixsize, square = True, multiple=2).map #Really shitty way to get the right map geometry for making filters
need = Needlet(np.arange(10), lightcone=template_map, L=10*60*np.sqrt(2), pixsize = pixsize * (3600 * 180) / np.pi)
fourier_radii = need.lightcone_box.get_grid_dimless_2d(return_grid=True)
need.get_needlet_filters_2d(fourier_radii)
'''
wmap = WavSkyMap(need.filters, lims, pixsize, square = True, multiple=2)

delta_x, delta_y = wmap.lims[1]-wmap.lims[0], wmap.lims[3]-wmap.lims[2]

new_lims = wmap.lims
new_lims[0] -= 0.5 * delta_x
new_lims[1] += 0.5 * delta_x
new_lims[2] -= 0.5 * delta_y
new_lims[3] += 0.5 * delta_y

lims = new_lims

wmap = WavSkyMap(np.zeros(1), new_lims, pixsize, square = True, multiple=2).map
need = needlet(np.arange(10), lightcone=wmap, L=10*np.sqrt(2)*60, pixsize = pixsize*(3600*180)/np.pi)
fourier_radii = need.lightcone_box.get_grid_dimless_2d(return_grid=True)
need.get_needlet_filters_2d(fourier_radii)
'''

map_size = pixsize * template_map.shape[-1] * ( 180 * 60 ) / np.pi #in arcmin
fourier_radii_phys = fourier_radii * 2 * np.pi / map_size #Units inverse arcmin

cut_arcmin = 2 #put a prior on scales larger than this
fourier_prior = np.where((fourier_radii_phys <= (2 * np.pi / cut_arcmin)), 1e12, 0)
#flags = np.where((fourier_radii_phys <= (2 * np.pi / cut_arcmin)))[0]

wmap = WavSkyMap(need, lims, pixsize, square = True, multiple=2)

for tod in todvec.tods:
    ipix=wmap.get_pix(tod)
    tod.info['ipix']=ipix
    tod.set_noise(minkasi.NoiseSmoothedSVD) 

hits=minkasi.make_hits(todvec,wmap)

mapset=Mapset()
mapset.add_map(wmap)

prior = wmap.copy()
#prior.map[0] = prior_map
#prior.map[0][:] = 1e3
#prior.map[1][:] = 1e-3
#for i in range(len(prior.map)):
    #prior.map[i] = prior_map*need.filters[i]

real_prior = np.fft.fftshift(np.fft.ifftn(fourier_prior).real)

prior.map = map2wav_real(real_prior, need.filters)

k_arr_phys = need.k_arr* 2*np.pi / map_size

for i in range(len(prior.map)):
    if np.max(k_arr_phys[np.where(need.bands[i] > 0)]) >= (2 * np.pi / cut_arcmin): #0 out prior map below the cut scale
        prior.map[i][:] = 0

prior_mapset = PriorMapset()

prior_mapset.add_map(prior)


#make A^T N^1 d.  TODs need to understand what to do with maps
#but maps don't necessarily need to understand what to do with TODs,
#hence putting make_rhs in the vector of TODs.
#Again, make_rhs is MPI-aware, so this should do the right thing
#if you run with many processes.
rhs=mapset.copy()
todvec.make_rhs(rhs)

#if minkasi.myrank==0:
#    rhs.maps[0].write('/scratch/r/rbond/jorlo/MS0735/needlets/needle_rhs.fits')
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
#precon.maps[0].map[:]=numpy.sqrt(tmp)
#precon.maps[0].map[:]=tmp[:]

#outroot = '/scratch/r/rbond/jorlo/MS0735/needlets/needle'
#outroot = '/scratch/r/rbond/jorlo/A399-401'
outroot = "/mnt/welch/USERS/jorlo/needlets/RXJ1347"

save_iters = [1,5,10,15,20,25, 50, 100, 150,200,250,300,350,400,450, 499]
#run PCG!
mapset_out=minkasi.run_pcg_wprior(rhs,x0,todvec,maxiter=50, save_iters=save_iters, outroot = outroot)#, prior = prior_mapset)

if minkasi.myrank==0:
    mapset_out.maps[0].write(outroot+"_need.fits")

    #mapset_out.maps[0].write('/scratch/r/rbond/jorlo/A399-401/A399-401_needle.fits') #and write out the map as a FITS file
else:
    print('not writing map on process ',minkasi.myrank)

#if you wanted to run another round of PCG starting from the previous solution,
#you could, replacing x0 with mapset_out.
#mapset_out2=minkasi.run_pcg(rhs,mapset_out,todvec,mapset,maxiter=50)
#mapset_out2.maps[0].write('second_map.fits')

'''
toc = time.time()
for i in range(len(need.filters)):
    print(i, end = '\r')
    need_lims = need.get_need_lims(i, real_space = True)
    if need_lims[0] > 180 or need_lims[1] < 60: continue #If small end of filter is larger than M2 recoverable (~3arcmin) or large end is smaller than ACT beamsize (1 arcsec), don't bother 
    svd = wmap.get_svd(i, down_samp = 10)
    temp_map = wmap.copy()

    smat = np.diag(svd.S)
    for j, S in enumerate(svd.S): #parallelize here
        if S > 1e-6:
            wmapset = Mapset()
            temp_map.clear()
            cur = np.dot(svd.U[...,j], np.dot(smat[j], svd.Vh[...,j]))
            temp_map.map[i] = np.reshape(cur, [306, 306])
            wmapset.add_map(temp_map)
            todvec.dot(wmapset)
            #RN this doesn't do anything with the dot product
tic = time.time()
'''


