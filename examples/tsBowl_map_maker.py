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
print("nproc: ", minkasi.nproc)
tod_names = tod_names[minkasi.myrank::minkasi.nproc]

todvec=minkasi.TodVec()

#loop over each file, and read it.
for i, fname in enumerate(tod_names):
#    if i >= 20: continue
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

minkasi.barrier()

lims=todvec.lims()
pixsize=2.0/3600*np.pi/180
map=minkasi.SkyMap(lims,pixsize)

mapset = minkasi.Mapset()

for i, tod in enumerate(todvec.tods):
    ipix=map.get_pix(tod)
    tod.info['ipix']=ipix
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

for key in precon.maps[1].data.keys():
    temp = np.ones(precon.maps[1].data[key].params.shape)
    temp /= precon.maps[1].data[key].vecs.shape[1]
    precon.maps[1].data[key].params = temp



#run PCG!
plot_info={}
plot_info['vmin']=-6e-4
plot_info['vmax']=6e-4
plot_iters=[1,2,3,5,10,15,20,25,30,35,40,45,49]

mapset_out=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=50)
if minkasi.myrank==0:
    mapset_out.maps[0].write('/scratch/r/rbond/jorlo/tsBowl_map_precon_mpi_py3.fits') #and write out the map as a FITS file
    '''    
    tod = todvec.tods[0]
    key = tod.info['fname']
    tsBowl = mapset_out.maps[1].data[key]
    
    dd, pred2, cm = minkasi.fit_cm_plus_poly(tod.info['dat_calib'], cm_ord=3, full_out=True) 
    
    tod.info['dat_calib'] -= pred2 
    for i in range(tod.info['dat_calib'].shape[0]):
        plt.plot(tod.info['apix'][i], tod.info['dat_calib'][i])
        plt.plot(tod.info['apix'][i], np.dot(tsBowl.params[i], tsBowl.vecs[i].T))
    
        plt.ylim(-0.05, 0.05)
        plt.xlabel('apix')
        plt.ylabel('dat_calib - pred2')
        plt.savefig('/scratch/r/rbond/jorlo/MS0735/tsBowl/precon_{}.png'.format(i))
        plt.close()
    '''



else:
    print('not writing map on process ',minkasi.myrank)

if minkasi.nproc>1:
    minkasi.MPI.Finalize()

#if you wanted to run another round of PCG starting from the previous solution, 
#you could, replacing x0 with mapset_out.  
#mapset_out2=minkasi.run_pcg(rhs,mapset_out,todvec,mapset,maxiter=50)
#mapset_out2.maps[0].write('second_map.fits')



