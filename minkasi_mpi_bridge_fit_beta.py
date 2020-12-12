import numpy as np
from matplotlib import pyplot as plt
import minkasi
import time
import glob
from astropy.io import fits
from astropy.wcs import WCS



def read_map(fname):
    f=fits.open(fname)
    raw=f[0].data.T
    tmp=raw.copy()
    f.close()
    return tmp


try:
    from importlib import reload
except:
    pass
reload(minkasi)
plt.ion()

#find tod files we want to map

if True:
    dir='../data/bridge/'
    act_dir='/Users/sievers/actpol/pixel_noise/'
else:
    dir='../data/bridge/TS_EaCMS0f0_51_21_Jan_2020/'
    act_dir='./act_priors/'


tod_names=glob.glob(dir+'*.fits')  
#tod_names=tod_names[:3]
outroot='maps/bridge2/bridge_fit_beta'


act_name=act_dir+'act_90ghz_stacked.fits'
act_map=read_map(act_dir+'act_90ghz_stacked.fits')
act_map=np.asarray(act_map,dtype='double')
act_ivar_file=act_dir+'act_90ghz_stacked_ivar.fits'
act_ivar=read_map(act_ivar_file)
act_ivar=np.asarray(act_ivar,dtype='double')
ww=WCS(act_dir+'act_90ghz_stacked.fits')


#if running MPI, you would want to split up files between processes
#one easy way is to say to this:
tod_names=tod_names[minkasi.myrank::minkasi.nproc]
if minkasi.nproc==1:
    tod_names=tod_names[:5]
#NB - minkasi checks to see if MPI is around, if not
#it sets rank to 0 an nproc to 1, so this would still
#run in a non-MPI environment


todvec=minkasi.TodVec()

#loop over each file, and read it.
for fname in tod_names:
    t1=time.time()
    dat=minkasi.read_tod_from_fits(fname)
    t2=time.time()
    dat['dat_calib']=1e6*dat['dat_calib'] #bring M2 into uK, on the same scale as ACT
    minkasi.truncate_tod(dat) #truncate_tod chops samples from the end to make
                              #the length happy for ffts
    #minkasi.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
                                  #this fixes that.  You don't need to, though.
    #minkasi.truncate_tod(dat)    #since our length changed, make sure we have a happy length

    #figure out a guess at common mode #and (assumed) linear detector drifts/offset
    #drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    dd=minkasi.fit_cm_plus_poly(dat['dat_calib'])  
    
    dat['dat_calib']=dd
    t3=time.time()
    tod=minkasi.Tod(dat)
    todvec.add_tod(tod)
    print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)





#pars=np.zeros(5)
#pars[0]=np.mean(tod.info['dx'])
#pars[1]=np.mean(tod.info['dy'])
#pars[2]=30./3600*np.pi/180 #theta
#pars[3]=-0.7
#pars[4]=-250
#d1,p1=minkasi.get_ts_derivs_many_funcs(tod,pars,np.asarray(5,dtype='int'),[minkasi.derivs_from_isobeta_c])


dd=tod.info['dat_calib']
tmp=np.random.randn(dd.shape[0],dd.shape[1])
nn=(1+np.random.rand(dd.shape[0]))*1
tmp=np.dot(np.diag(nn),tmp)
#tod.info['dat_calib']=tmp
tod.set_noise(minkasi.NoiseSmoothedSVD,tmp)
asdf=tod.noise.get_det_weights()

mystd=np.std(tmp,axis=1)
pp=np.polyfit(1/mystd**2,asdf,1)


cm=np.random.randn(tmp.shape[1])
tmp2=tmp+np.outer(np.ones(tmp.shape[0]),cm)*20

tod.set_noise(minkasi.NoiseCMWhite,tmp2)
#asdf2=tod.noise.get_det_weights()
asdf2=tod.get_det_weights()
pp2=np.polyfit(1/mystd**2,asdf2,1)


#make a template map with desired pixel size an limits that cover the data
#todvec.lims() is MPI-aware and will return global limits, not just
#the ones from private TODs
lims=todvec.lims()
mypad=np.asarray([-5,5,-5,5])/60*np.pi/180
for i in range(len(lims)):
    lims[i]=lims[i]+mypad[i]


#act_dir='/Users/sievers/actpol/pixel_noise/'
#act_name=act_dir+'act_90ghz_stacked.fits'
act_deconvolved=act_dir+'act_90ghz_stacked_deconvolved_0.1.fits'
#act_ivar=act_dir+'act_90ghz_stacked_ivar.fits'

osamp=15
act_prior=minkasi.SkyMapTwoRes(act_name,lims,osamp=osamp,smooth_fac=0.1)
act_prior.get_map_deconvolved(act_deconvolved)
fwhm_pix=1.4*150/90/0.5
act_prior.set_beam_gauss(fwhm_pix)
act_prior.set_noise_white(act_ivar_file)


print("lims are ",lims)
print("new lims are ",act_prior.small_lims)

map=minkasi.SkyMap(act_prior.small_lims,mywcs=act_prior.small_wcs,pad=0)


#map=minkasi.PolMap(lims,pixsize)

#once we have a map, we can figure out the pixellization of the data.  Save that
#so we don't have to recompute.  Also calculate a noise model.  The one here
#(and currently the only supported one) is to rotate the data into SVD space, then
#smooth the power spectrum of each mode.  Other models would not be hard
#to implement.  The smoothing could do with a bit of tuning as well.
for tod in todvec.tods:
    ipix=map.get_pix(tod)
    tod.info['ipix']=ipix
    tod.set_noise(minkasi.NoiseSmoothedSVD);tag='svd'
    #tod.set_noise_cm_white();tag='white'
if isinstance(map,minkasi.PolMap):
    tag=tag+'_pol'



pars_401=np.asarray([44.73*np.pi/180,13.57*np.pi/180,298.0/3600*np.pi/180,-0.9,-600])
pars_bridge=np.asarray([44.59*np.pi/180,13.31*np.pi/180,450/3600*np.pi/180,-0.9,-20])
pars_src1=np.asarray([44.6333*np.pi/180,13.5711*np.pi/180,10/2.35/3600*np.pi/180,8000])
pars_src2=np.asarray([44.5755*np.pi/180,13.2388*np.pi/180,10/2.35/3600*np.pi/180,500])
if False:
    pars=np.hstack([pars_401,pars_bridge,pars_src1,pars_src2])
    npar_func=np.asarray([5,5,4,4])
    funcs=[minkasi.derivs_from_isobeta_c,minkasi.derivs_from_isobeta_c,minkasi.derivs_from_gauss_c,minkasi.derivs_from_gauss_c]
else:
    pars=pars_401
    npar_func=np.asarray([5])
    #funcs=[minkasi.derivs_from_gauss_c]
    funcs=[minkasi.derivs_from_isobeta_c]


#chisq,grad,curve=minkasi.get_ts_curve_derivs_many_funcs(todvec,pars,npar_func,funcs)
pars_fit=minkasi.fit_timestreams_with_derivs_manyfun(funcs,pars,npar_func,todvec)


assert(1==0)



#get the hit count map.  We use this as a preconditioner
#which helps small-scale convergence quite a bit.
print('starting hits')
hits=minkasi.make_hits(todvec,map)
print('finished hits.')

act_prior.set_mask(hits.map)

map_act_finepix=map.copy()
map_act_finepix.clear()
imin=act_prior.map_corner[0]
imax=imin+act_prior.nx_coarse

jmin=act_prior.map_corner[1]
jmax=imin+act_prior.ny_coarse
ni=imax-imin
nj=jmax-jmin
for i in range(ni):
    for j in range(nj):
        map_act_finepix.map[(i*osamp):((i+1)*osamp),(j*osamp):((j+1)*osamp)]=act_map[i+imin,j+jmin]
#map_act_finepix.write('test_act_finepix.fits')


if False:
    naive=map.copy()
    naive.clear()
    for tod in todvec.tods:
        tmp=tod.info['dat_calib'].copy()
        u,s,v=np.linalg.svd(tmp,0)
        pred=np.outer(u[:,0],s[0]*v[0,:])
        tmp=tmp-pred
    
    #cm=np.median(tmp,axis=0)
    #for i in range(tmp.shape[0]):
    #    tmp[i,:]=tmp[i,:]-cm
        naive.tod2map(tod,tmp)
    naive.mpi_reduce()
    naive.map[hits.map>0]=naive.map[hits.map>0]/hits.map[hits.map>0]
    if minkasi.myrank==0:
        naive.write(outroot+'_naive.fits')
        hits.write(outroot+'_hits.fits')

hits_org=hits.copy()
hits.invert()

#assert(1==0)

#setup the mapset.  In general this can have many things
#in addition to map(s) of the sky, but for now we'll just 
#use a single skymap.

#for tod in todvec.tods:
#     tod.set_noise(minkasi.NoiseSmoothedSVD)
weightmap=minkasi.make_hits(todvec,map,do_weights=True)
mask=weightmap.map>0
tmp=weightmap.map.copy()
tmp[mask]=1./np.sqrt(tmp[mask])
noisemap=weightmap.copy()
noisemap.map[:]=tmp
if minkasi.myrank==0:
    noisemap.write(outroot+'_noise.fits')
    weightmap.write(outroot+'_weights.fits')




mapset=minkasi.Mapset()
mapset.add_map(map)

coarse_map=minkasi.SkyMapCoarse(act_map)
mapset.add_map(coarse_map)

#npoly=50
#polys=minkasi.tsModel(todvec,minkasi.tsPoly,order=npoly)
#mapset.add_map(polys)



#make A^T N^1 d.  TODs need to understand what to do with maps
#but maps don't necessarily need to understand what to do with TODs, 
#hence putting make_rhs in the vector of TODs. 
#Again, make_rhs is MPI-aware, so this should do the right thing
#if you run with many processes.
rhs=mapset.copy()
rhs.clear()
todvec.make_rhs(rhs)

act_prior.get_rhs(rhs)
#rhs.maps[0].map=rhs.maps[0].map+act_prior.get_rhs()


#this is our starting guess.  Default to starting at 0,
#but you could start with a better guess if you have one.
x0=rhs.copy()
x0.clear()

#preconditioner is 1/ hit count map.  helps a lot for 
#convergence.
precon=mapset.copy()
#tmp=hits.map.copy()
#ii=tmp>0
#tmp[ii]=1.0/tmp[ii]
#precon.maps[0].map[:]=np.sqrt(tmp)
precon.maps[0].map[:]=hits.map[:]
precon.maps[1].map[:]=1e-4
#for tod in todvec.tods:
#    cc=precon.maps[1].data[tod.info['fname']]
#    cc.map[:]=1.0



priorset=minkasi.Mapset()
priorset.add_map(act_prior)




if False:
    #check symmetry of finemap against fine map
    i=50;j=300;di=55;dj=25;

    mapset_out=mapset.copy()
    mapset.clear()
    mapset_out.clear()
    mapset.maps[0].map[i,j]=1.0

    act_prior.apply_prior(mapset,mapset_out)
    val=mapset_out.maps[0].map[i+di,j+dj]

    mapset.clear()
    mapset_out.clear()
    mapset.maps[0].map[i+di,j+dj]=1.0
    act_prior.apply_prior(mapset,mapset_out)
    val2=mapset_out.maps[0].map[i,j]
    print("vals are ",val,val2,val-val2)
    assert(1==0)

if False:
    #check symmetry of coarse map against coarse map
    i=50;j=300;di=5;dj=3;

    mapset_out=mapset.copy()
    mapset.clear()
    mapset_out.clear()
    mapset.maps[1].map[i,j]=1.0

    act_prior.apply_prior(mapset,mapset_out)
    val=mapset_out.maps[1].map[i+di,j+dj]

    mapset.clear()
    mapset_out.clear()
    mapset.maps[1].map[i+di,j+dj]=1.0
    act_prior.apply_prior(mapset,mapset_out)
    val2=mapset_out.maps[1].map[i,j]
    print("coarse vals are ",val,val2,val-val2)
    assert(1==0)

if False:
    #check symmetry of fine map against coarse map where coarse map is partially filled by fine map
    i=310;j=44;ii=20;jj=2
    ii=ii+act_prior.map_corner[0]
    jj=jj+act_prior.map_corner[1]
    mapset_out=mapset.copy()
    mapset.clear()
    mapset_out.clear()
    mapset.maps[0].map[i,j]=1.0

    act_prior.apply_prior(mapset,mapset_out)
    val=mapset_out.maps[1].map[ii,jj]
    myval=mapset_out.maps[0].map[i,j]


    mapset.clear()
    mapset_out.clear()
    mapset.maps[1].map[ii,jj]=1.0
    act_prior.apply_prior(mapset,mapset_out)
    val2=mapset_out.maps[0].map[i,j]
    print("partial_pixel vals are ",val,val2,val-val2)
    assert(1==0)

#tmp1=0*map.map;tmp1[i,j]=1.0;tmp1b=act_prior.apply_Qinv(tmp1)

if False:
    #old and broken, don't use this
    mapset_out=mapset.copy()
    mapset.clear()
    mapset_out.clear()
    mapset.maps[0].map[50,300]=1.0
    priorset.apply_prior(mapset,mapset_out)
    val=mapset_out.maps[1].map.max()
    ii=np.where(mapset_out.maps[1].map==val)
    
    mapset.clear()
    mapset_out.clear()
    mapset.maps[1].map[ii[0][0],ii[1][0]]=1.0
    priorset.apply_prior(mapset,mapset_out)
    print('values are ',val,mapset_out.maps[0].map[50,300])



    i=50;j=300;di=55;dj=25;tmp1=0*map.map;tmp1[i,j]=1.0;tmp1b=act_prior.apply_Qinv(tmp1)
    tmp2=0*map.map;tmp2[i+di,j+dj]=1.0;tmp2b=act_prior.apply_Qinv(tmp2);aa=tmp1b[i+di,j+dj];bb=tmp2b[i,j];print(aa,bb,bb-aa)
    print(hits_org.map[i,j],hits_org.map[i+di,j+dj])
    assert(1==0)

#run PCG
iters=[5,10,15,20,25,50,75,100,150,200]

mapset_out=minkasi.run_pcg_wprior(rhs,x0,todvec,priorset,precon,maxiter=101,outroot=outroot+"_pass1",save_iters=iters)
#mapset_out=minkasi.run_pcg_wprior(rhs,x0,todvec,None,precon,maxiter=101,outroot=outroot+"_noprior",save_iters=iters)
if minkasi.myrank==0:
    mapset_out.maps[0].write(outroot+'_pass1_'+tag+'.fits') #and write out the map as a FITS file
    #np.save(outroot+'_pass1_coarse.npy',mapset_out.maps[1])
    coarse=priorset.maps[0].maps2coarse(mapset_out.maps[0].map,mapset_out.maps[1].map)
    header=fits.getheader(act_name)
    minkasi.write_fits_map_wheader(coarse,outroot+'_pass1_coarse.fits',header)
    coarse_smooth=priorset.maps[0].beam_convolve(coarse)
    minkasi.write_fits_map_wheader(coarse_smooth,outroot+'_pass1_coarse_smooth.fits',header)

    fine=priorset.maps[0].maps2fine(mapset_out.maps[0].map,mapset_out.maps[1].map)
    tmp=map.copy()
    tmp.map[:]=fine
    tmp.write(outroot+'_pass1_wact_'+tag+'.fits')
else:
    print('not writing map on process ',minkasi.myrank)

for tod in todvec.tods:
    tmp=tod.info['dat_calib']-tod.mapset2tod(mapset_out)
    tod.set_noise(minkasi.NoiseSmoothedSVD,dat=tmp);
    del tmp

rhs=mapset.copy()
rhs.clear()
todvec.make_rhs(rhs)
act_prior.get_rhs(rhs)

mapset_out2=minkasi.run_pcg_wprior(rhs,mapset_out,todvec,priorset,precon,maxiter=201,outroot=outroot+"_pass2",save_iters=iters)
if minkasi.myrank==0:
    mapset_out2.maps[0].write(outroot+'_pass2_'+tag+'.fits') #and write out the map as a FITS file
    #np.save(outroot+'_pass1_coarse.npy',mapset_out2.maps[1])
    coarse=priorset.maps[0].maps2coarse(mapset_out2.maps[0].map,mapset_out2.maps[1].map)
    header=fits.getheader(act_name)
    minkasi.write_fits_map_wheader(coarse,outroot+'_pass2_coarse.fits',header)
    coarse_smooth=priorset.maps[0].beam_convolve(coarse)
    minkasi.write_fits_map_wheader(coarse_smooth,outroot+'_pass2_coarse_smooth.fits',header)

    fine=priorset.maps[0].maps2fine(mapset_out2.maps[0].map,mapset_out2.maps[1].map)
    tmp=map.copy()
    tmp.map[:]=fine
    tmp.write(outroot+'_pass2_wact_'+tag+'.fits')
else:
    print('not writing map on process ',minkasi.myrank)
    time.sleep(15)


minkasi.barrier()
assert(1==0)


noise_iter=4
for niter in range(noise_iter):
    maxiter=50*(niter+1)+1
    #first, re-do the noise with the current best-guess map
    for tod in todvec.tods:
        mat=0*tod.info['dat_calib']
        for mm in mapset_out.maps:
            mm.map2tod(tod,mat)
        tod.set_noise(minkasi.NoiseSmoothedSVD,tod.info['dat_calib']-mat)


    gradmap=hits.copy()
    
    gradmap.map[:]=minkasi.get_grad_mask_2d(mapset_out.maps[0],todvec,thresh=1.8)
    prior=minkasi.tsModel(todvec,minkasi.CutsCompact)
    for tod in todvec.tods:
        prior.data[tod.info['fname']]=tod.prior_from_skymap(gradmap)
        print('prior on tod ' + tod.info['fname']+ ' length is ' + repr(prior.data[tod.info['fname']].map.size))

    mapset=minkasi.Mapset()
    mapset.add_map(mapset_out.maps[0])
    pp=prior.copy()
    pp.clear()
    mapset.add_map(pp)

    priorset=minkasi.Mapset()
    priorset.add_map(map)
    priorset.add_map(prior)
    priorset.maps[0]=None

    rhs=mapset.copy()
    todvec.make_rhs(rhs)

    precon=mapset.copy()
    precon.maps[0].map[:]=hits.map[:]
    for tod in todvec.tods:
        cc=precon.maps[1].data[tod.info['fname']]
        cc.map[:]=1.0
    mapset_out=minkasi.run_pcg_wprior(rhs,mapset,todvec,priorset,precon,maxiter=maxiter,outroot=outroot+'_niter_'+repr(niter),save_iters=iters)
    if minkasi.myrank==0:
        mapset_out.maps[0].write(outroot+'_niter_'+repr(niter)+'.fits')

minkasi.barrier()

