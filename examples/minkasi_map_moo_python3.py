import numpy
import numpy as np
from matplotlib import pyplot as plt
import minkasi
import time
import glob
from importlib import reload
reload(minkasi)
plt.ion()

def smooth_map(map,npix=3):
    n=map.shape[0]
    m=map.shape[1]
    v1=np.fft.fftfreq(n)*n
    v2=np.fft.fftfreq(m)*m

    rmat=np.outer(v1,np.ones(m))**2+np.outer(np.ones(n),v2)**2
    kernel=np.exp(-0.5*rmat/npix**2)
    mapft=np.fft.rfft2(map)
    kft=np.fft.rfft2(kernel)
    return np.fft.irfft2(mapft*kft/kft[0,0])
assert(1==0)
    

#find tod files we want to map
#dir='../data/moo1110/'
dir='/Users/sievers/mustang/data/moo1110/'
#dir='../data/moo1046/'
tod_names=glob.glob(dir+'Sig*.fits')  

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
pixsize=2.0/3600*numpy.pi/180
map=minkasi.SkyMap(lims,pixsize)

#once we have a map, we can figure out the pixellization of the data.  Save that
#so we don't have to recompute.  Also calculate a noise model.  The one here
#(and currently the only supported one) is to rotate the data into SVD space, then
#smooth the power spectrum of each mode.  Other models would not be hard
#to implement.  The smoothing could do with a bit of tuning as well.


for tod in todvec.tods:
    ipix=map.get_pix(tod)
    tod.info['ipix']=ipix
    #tod.set_noise_smoothed_svd()
    tod.set_noise(minkasi.NoiseSmoothedSVD)



inds=[0,1,-1]
for ind in inds:
    tod=todvec.tods[ind]
    plt.clf();
    dd=tod.info['dat_calib'].copy()
    tvec=np.arange(dd.shape[1])*tod.info['dt']
    plt.plot(tvec,dd.T)
    plt.xlabel("Time (s)")
    plt.ylabel("Detector Temperature (K)")
    tt=tod.info['fname']
    mystr=tod.info['fname']
    i1=mystr.find('-');i2=mystr.find('.',3);tag=mystr[i1+1:i2]
    plt.savefig('all_detectors_' + tag +'.png')

    plt.clf();
    dat_filt=tod.apply_noise(tod.info['dat_calib'])
    plt.clf();
    plt.plot(tvec,dat_filt[::10,:].T/np.sqrt(tod.info['dt'])/1000)
    plt.savefig('noise_filtered_dets_'+tag+'.png')

#assert(1==0)





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
#precon.maps[0].map[:]=numpy.sqrt(tmp)
precon.maps[0].map[:]=tmp[:]

#run PCG!
plot_info={}
plot_info['vmin']=-6e-4
plot_info['vmax']=6e-4
plot_iters=[1,2,3,5,10,15,20,25,30,35,40,45,49]

mapset_out=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=50,plot_iters=plot_iters,plot_info=plot_info)
if minkasi.myrank==0:
    mapset_out.maps[0].write('first_map_precon_mpi_py3.fits') #and write out the map as a FITS file
else:
    print('not writing map on process ',minkasi.myrank)

if minkasi.nproc>1:
    minkasi.MPI.Finalize()

#if you wanted to run another round of PCG starting from the previous solution, 
#you could, replacing x0 with mapset_out.  
#mapset_out2=minkasi.run_pcg(rhs,mapset_out,todvec,mapset,maxiter=50)
#mapset_out2.maps[0].write('second_map.fits')

