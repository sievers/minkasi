import numpy as np
from matplotlib import pyplot as plt
import glob
import time

import os
from astropy.coordinates import Angle
from astropy import units as u
from astropy.io import fits

import minkasi.minkasi_all as minkasi
from minkasi.needlet.needlet import WavSkyMap
from minkasi.needlet.needlet import needlet, cosmo_box
from minkasi.needlet.needlet import wav2map_real, map2wav_real
from minkasi.maps.mapset import PriorMapset, Mapset

try:
    import mpi4py.rc

    mpi4py.rc.threads = False
    from mpi4py import MPI

    print("mpi4py imported")
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nproc = comm.Get_size()
    print("nproc:, ", nproc)
    if nproc > 1:
        have_mpi = True
    else:
        have_mpi = False

except:
    MPI = None
    comm = None
    have_mpi = False
    myrank = 0
    nproc = 1

#idir = "/home/jack/M2-TODs/RXJ1347/"
idir = "/scratch/r/rbond/jorlo//M2-TODs/RXJ1347/"
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

wmap = WavSkyMap(np.zeros(1), lims, pixsize, square = True, multiple=2).map #Really shitty way to get the right map geometry for making filters
wmap  = WavSkyMap(np.zeros(1), lims, pixsize, square = True, multiple=2).map #TODO: fix squaring issue
need = needlet(np.arange(10), lightcone=wmap, L=10*60*np.sqrt(2), pixsize = pixsize * (3600 * 180) / np.pi)
fourier_radii = need.lightcone_box.get_grid_dimless_2d(return_grid=True)
need.get_needlet_filters_2d(fourier_radii)

wmap = WavSkyMap(need.filters, lims, pixsize, square = True, multiple=2)

for tod in todvec.tods:
    ipix=wmap.get_pix(tod)
    tod.info['ipix']=ipix
    tod.set_noise(minkasi.NoiseSmoothedSVD)

hits=minkasi.make_hits(todvec,wmap)


#Subtract off modes outside the joint ACT+M2 window
filt = 4
if minkasi.myrank==0:

    #Something here doesn't play nice with mpi so we do it single threaded and send it out
    svd = wmap.get_svd(filt, down_samp = 10) #TODO: Parallelize
    tol = 1e-1
    mask = np.where((np.abs(svd.S) > np.max(np.abs(svd.S))*tol))[0]

    mask = mask[:(len(mask) // minkasi.nproc)*minkasi.nproc]
    U = svd.U[mask, :] #check if it's mask, : or :, mask
    Vh = svd.Vh[mask, :]
    S = svd.S[mask]
    print(U.shape, Vh.shape, S.shape)
    smat = np.diag(S)

    for i in range(1, minkasi.nproc):
        comm.send(U, dest = i, tag = 0)
        comm.send(Vh, dest = i, tag = 1)
        comm.send(smat, dest = i, tag = 2)
else:
    U = comm.recv(source = 0, tag = 0)
    Vh = comm.recv(source = 0, tag = 1)
    smat = comm.recv(source = 0, tag = 2)


minkasi.barrier()
toc = time.time()

svd = np.stack(Vh).T #Stack up all svds for all wavelets in window. Shape [nSVDs, map.ravel]
print(svd.shape)
svd_ANA = np.zeros([len(smat), len(smat)])


for j in range(minkasi.myrank, len(smat), minkasi.nproc):
    temp_map = wmap.copy()
    cur = Vh[j,:]

    wmapset = Mapset()
    temp_map.clear()
    temp_map.map[filt] = np.reshape(cur, [306, 306])
    wmapset.add_map(temp_map)
    mapout = todvec.dot(wmapset) #Dot this with whole vector of SVD componants that looks like maps

    svd_ANA[:, j] = np.ravel(np.dot(np.ravel(mapout.maps[0].map[filt]), svd))

 
comm.barrier()
import pickle as pk

if minkasi.myrank == 0 :
    with open("/scratch/r/rbond/jorlo/svd_ANA.pk", "wb") as f:
        pk.dump(svd_ANA, f)
#svd_ANA = comm.all_reduce(svd_ANA)

#RHS = usual rhs with stacked svd.U
#LHS = svd_ANA
#chis2 = dot(lhs, rhs)
if minkasi.myrank == 0:
    print("out")
tic = time.time()

print(tic-toc)
imap = minkasi.SkyMap(lims, pixsize)

mapset=minkasi.Mapset()
mapset.add_map(imap)


rhs=mapset.copy()
todvec.make_rhs(svd)
chis2 = np.dot(svd_ANA, rhs)
print(chis2)

