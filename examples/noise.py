import pdb

import numpy as np
import minkasi.minkasi_all as minkasi

import time
import glob
import os
#reload(minkasi)

#set file root for output maps
outroot = "/mnt/welch/USERS/jorlo/Reductions/RXJ_sim/RXJ_"
#Note the end of this path is a filename, files will be written to
#RXJ1347/RXJ1347_1.fits, RXJ1347/RXJ1347_5.fits, etc. thru RXJ1347/RXJ1347_final.fits


#find tod files we want to map
idir = "/mnt/welch/MUSTANG/M2-TODs/RXJ1347/" #CHANGE ME
tod_names=glob.glob(idir+'/Sig*.fits')
if len(tod_names)==0:
    print('We found no TOD files.  Double check your path?')
    assert(1==0)

#Use presets by source to automatically get and cut TODs
#that were manually flagged for removal
bad_tod, _ = minkasi.get_bad_tods("RXJ1347")
tod_names = minkasi.cut_blacklist(tod_names, bad_tod)

tod_names=tod_names[minkasi.myrank::minkasi.nproc]

todvec=minkasi.TodVec()

for fname in tod_names:
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

def make_map(tod_names, outname, lims):
   
    todvec=minkasi.TodVec()
 
    for fname in tod_names:
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
    mapset_out=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=50, save_iters=save_iters, outroot = os.path.join(outroot, outname)) 

    return mapset_out.maps[0].map

todlim = int(len(tod_names)/2)

front = make_map(tod_names[:todlim], "front", lims)
back = make_map(tod_names[todlim:], "back", lims)

pdb.set_trace()






















