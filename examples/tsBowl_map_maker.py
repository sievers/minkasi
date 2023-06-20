import minkasi
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
import minkasi_jax.presets_by_source as pbs
import os
from astropy.coordinates import Angle
from astropy import units as u

dir = '/scratch/r/rbond/jorlo/MS0735//TS_EaCMS0f0_51_5_Oct_2021/'
tod_names=glob.glob(dir+'Sig*.fits')

bad_tod, addtag = pbs.get_bad_tods("MS0735", ndo=False, odo=False)
#bad_tod.append('Signal_TOD-AGBT21A_123_03-s20.fits')
tod_names = minkasi.cut_blacklist(tod_names, bad_tod)


print("nproc: ", minkasi.nproc)
tod_names = tod_names[minkasi.myrank::minkasi.nproc]

todvec=minkasi.TodVec()

#loop over each file, and read it.
for i, fname in enumerate(tod_names):
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
    #print(i)
    #print(tod.info['fname'])
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
        mapset_out.maps[0].write('/scratch/r/rbond/jorlo/noBowl_map_precon_mpi_py3.fits') #and write out the map as a FITS file
    else:
        mapset_out.maps[0].write('/scratch/r/rbond/jorlo/tsBowl_map_precon_mpi_py3.fits')
else:
    print('not writing map on process ',minkasi.myrank)




#if you wanted to run another round of PCG starting from the previous solution, 
#you could, replacing x0 with mapset_out.  
#mapset_out2=minkasi.run_pcg(rhs,mapset_out,todvec,mapset,maxiter=50)
#mapset_out2.maps[0].write('second_map.fits')

d2r=np.pi/180
sig=9/2.35/3600*d2r
theta0 = np.deg2rad(97)

x0 = Angle('07 41 44.5 hours').to(u.radian).value
y0 = Angle('74:14:38.7 degrees').to(u.radian).value
beta_pars=np.asarray([x0,y0,theta0,0.98,-8.2e-1])

x0_src = Angle('07 41 44.5 hours').to(u.radian).value
y0_src = Angle('74:14:38.7 degrees').to(u.radian).value

src1_pars=np.asarray([x0_src, y0_src,1.37e-5,1.7e-4])


#src2_pars=np.asarray([155.90447*d2r,4.1516*d2r,2.6e-5,5.1e-4])

pars=np.hstack([beta_pars,src1_pars])  #we need to combine parameters into a single vector
npar=np.hstack([len(beta_pars),len(src1_pars)]) #and the fitter needs to know how many per function
#this array of functions needs to return the model timestreams and derivatives w.r.t. parameters
#of the timestreams.  
funs=[minkasi.derivs_from_isobeta_c,minkasi.derivs_from_gauss_c]
#we can keep some parameters fixed at their input values if so desired.
to_fit=np.ones(len(pars),dtype='bool')
to_fit[[0,1,2,5,6]]=False  #Let's keep beta pegged to 0.7

'''
t1=time.time()
pars_fit,chisq,curve,errs=minkasi.fit_timestreams_with_derivs_manyfun(funs,pars,npar,todvec_copy,to_fit)
t2=time.time()
if minkasi.myrank==0:
    print('No Sub: ')
    print('took ',t2-t1,' seconds to fit timestreams')
    for i in range(len(pars_fit)):
        print('parameter ',i,' is ',pars_fit[i],' with error ',errs[i])
'''
minkasi.comm.barrier()

if minkasi.myrank==0:
    for tod in todvec.tods:
        todname = tod.info['fname']
        temp = minkasi.map2todbowl(mapset_out.maps[1].data[todname].vecs, mapset_out.maps[1].data[todname].params)  
        tod.info['dat_calib'] -= temp 
'''
t1=time.time()
pars_fit,chisq,curve,errs=minkasi.fit_timestreams_with_derivs_manyfun(funs,pars,npar,todvec,to_fit)
t2=time.time()
if minkasi.myrank==0:
    print('Sub: ')
    print('took ',t2-t1,' seconds to fit timestreams')
    for i in range(len(pars_fit)):
        print('parameter ',i,' is ',pars_fit[i],' with error ',errs[i])
'''

lims=todvec.lims()
pixsize=2.0/3600*np.pi/180
map=minkasi.SkyMap(lims,pixsize)

mapset = minkasi.Mapset()
mapset.add_map(map)
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


mapset_out=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=50)

if minkasi.myrank==0:
    mapset_out.maps[0].write('/scratch/r/rbond/jorlo/sub_tsBowl_map_precon_mpi_py3.fits') #and write out the map as a FITS file


else:
    print('not writing map on process ',minkasi.myrank)
