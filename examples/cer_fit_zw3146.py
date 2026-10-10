import numpy
import numpy as np
from matplotlib import pyplot as plt
import minkasi
import time
import glob
from importlib import reload
reload(minkasi)
plt.ion()

mydir = '/home/scratch/cromero/mustang/MUSTANG2/Reductions/MOO_1142/'
tod_names=glob.glob(mydir+'TS_EaCMS0f0_51_18_Jun_2025/Signal_TOD*.fits')

### Fit initial features as Gaussians; notably with the locations, amplitude, and widths free.
Gaussian_file = 'TBD'

#edges=numpy.asarray([0,5,15,25,35,45,55,70,85,100,120,140,160,180,200,220,240],dtype='double')
edges=numpy.asarray([0,5,15,25,35,45,55,70,85,100,120,150,180,240],dtype='double')
#clus_cent = numpy.asarray([155.91545,4.1835417,0.0004,0.0],dtype='double')

use_Gaussian_center = False
initial_map = mydir+"MinkasiOutputs/Minkasi_naivePCG_map.fits"

##############################################################################################

tod_names=tod_names[minkasi.myrank::minkasi.nproc]
todvec=minkasi.TodVec()

for fname in tod_names:
    t1=time.time()
    dat=minkasi.read_tod_from_fits(fname)
    t2=time.time()
    ### Downsample is optional. Ideally MUSTANG-2 TODs are appropriately downsampled.
    #minkasi.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
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
pixsize=2.0/3600*numpy.pi/180  # 2 arcseconds, stored as radians
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
    
fitp=numpy.load(ptsrc_file)
npar_src=4
coords=numpy.zeros([fitp.shape[0]/4,2])
coords[:,0]=fitp[0::npar_src]
coords[:,1]=fitp[1::npar_src]
coords=coords*180/numpy.pi
pixcents=map.wcs.wcs_world2pix(coords,1)
nsrc=pixcents.shape[0]-1
print '##########################################'
print coords, pixcents
print '##########################################'
rings=minkasi.make_rings(edges,clus_cent,map,fwhm=11.1)
npar=rings.shape[0]
mm=map.copy()
lhsr=numpy.zeros([npar+nsrc,npar+nsrc])
rhsr=numpy.zeros(npar+nsrc)
srcp=numpy.zeros(npar_src)

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

mapset_out=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=50)#,plot_iters=plot_iters,plot_info=plot_info)

if minkasi.myrank==0:
    print(type(mapset_out.maps[0]))
    mapset_out.maps[0].write(initial_map) #and write out the map as a FITS file
else:
    print('not writing map on process ',minkasi.myrank)

####################################################################################
    
for tod in todvec.tods:
    model_ts=[None]*(npar+nsrc)
    model_ts_filt=[None]*(npar+nsrc)
    for i in range(npar):
        mm.map[:]=rings[i,:,:]
        dat=numpy.zeros(tod.info['dat_calib'].shape)
        mm.map2tod(tod,dat)
        dat_filt=tod.apply_noise(dat)
        model_ts[i]=dat
        model_ts_filt[i]=dat_filt
    for i in range(nsrc):
        srcp[:]=fitp[(i+1)*npar_src:(i+2)*npar_src]
        srcp[-1]=1.0
        derivs,pred=minkasi.derivs_from_gauss_c(srcp,tod)
        pred_filt=tod.apply_noise(pred)
        model_ts[i+npar]=pred
        model_ts_filt[i+npar]=pred_filt
    for i in range(npar+nsrc):
        rhsr[i]=rhsr[i]+mydot(model_ts_filt[i],tod.info['dat_calib'])
        for j in range(i,npar+nsrc):
            lhsr[i,j]=lhsr[i,j]+mydot(model_ts[i],model_ts_filt[j])
            lhsr[j,i]=lhsr[i,j]
    print 'finished tod'

lhsr=minkasi.comm.allreduce(lhsr)
rhsr=minkasi.comm.allreduce(rhsr)
k=1
if minkasi.myrank==0:
    numpy.save(outdir+'zwicky_3146_rings_curve'+tag+'_pass_'+repr(k)+'.npy',lhsr)
    numpy.save(outdir+'zwicky_3146_rings_deriv'+tag+'_pass_'+repr(k)+'.npy',rhsr)
    numpy.save(outdir+'zwicky_3146_rings_edges'+tag+'_pass_'+repr(k)+'.npy',edges)
        #numpy.save(outdir+'zwicky_3146_rings_curve'+tag+'.npy',lhsr)
        #numpy.save(outdir+'zwicky_3146_rings_deriv'+tag+'.npy',rhsr)
        #numpy.save(outdir+'zwicky_3146_rings_edges'+tag+'.npy',edges)
minkasi.comm.Barrier()

npass=6
niter=50
for k in range(2,npass):
    print 'working on pass ',k
    #let's re-estimate the noise with the current model subtracted off 
    #to try to avoid bowls around bright things
    for tod in todvec.tods:
        dat=tod.info['dat_calib'].copy()
        tmp=numpy.zeros(dat.shape)
        outmap=mapset_out.maps[0]
        outmap.map2tod(tod,tmp)
        tod.info['dat_calib'][:]=dat-tmp
        tod.set_noise_smoothed_svd(fwhm=svdfwhm)
        tod.info['dat_calib'][:]=dat

    rhs=mapset.copy()
    todvec.make_rhs(rhs)

    #if you wanted to run another round of PCG starting from the previous solution, 
    #you could, replacing x0 with mapset_out.  
    mapset_out2=minkasi.run_pcg(rhs,mapset_out,todvec,precon,maxiter=niter)
    if minkasi.myrank==0:
        mapset_out2.maps[0].write(outroot+'_pass_'+repr(k)+'.fits')
    mapset_out=mapset_out2

    for tod in todvec.tods:
        model_ts=[None]*(npar+nsrc)
        model_ts_filt=[None]*(npar+nsrc)
        for i in range(npar):
            mm.map[:]=rings[i,:,:]
            dat=numpy.zeros(tod.info['dat_calib'].shape)
            mm.map2tod(tod,dat)
            dat_filt=tod.apply_noise(dat)
            model_ts[i]=dat
            model_ts_filt[i]=dat_filt
        for i in range(nsrc):
            srcp[:]=fitp[(i+1)*npar_src:(i+2)*npar_src]
            srcp[-1]=1.0
            derivs,pred=minkasi.derivs_from_gauss_c(srcp,tod)
            pred_filt=tod.apply_noise(pred)
            model_ts[i+npar]=pred
            model_ts_filt[i+npar]=pred_filt
        for i in range(npar+nsrc):
            rhsr[i]=rhsr[i]+mydot(model_ts_filt[i],tod.info['dat_calib'])
            for j in range(i,npar+nsrc):
                lhsr[i,j]=lhsr[i,j]+mydot(model_ts[i],model_ts_filt[j])
                lhsr[j,i]=lhsr[i,j]
        print 'finished tod'

    lhsr=minkasi.comm.allreduce(lhsr)
    rhsr=minkasi.comm.allreduce(rhsr)
    if minkasi.myrank==0:
        numpy.save(outdir+'zwicky_3146_rings_curve'+tag+'_pass_'+repr(k)+'.npy',lhsr)
        numpy.save(outdir+'zwicky_3146_rings_deriv'+tag+'_pass_'+repr(k)+'.npy',rhsr)
        numpy.save(outdir+'zwicky_3146_rings_edges'+tag+'_pass_'+repr(k)+'.npy',edges)
        #numpy.save(outdir+'zwicky_3146_rings_curve'+tag+'.npy',lhsr)
        #numpy.save(outdir+'zwicky_3146_rings_deriv'+tag+'.npy',rhsr)
        #numpy.save(outdir+'zwicky_3146_rings_edges'+tag+'.npy',edges)
    minkasi.comm.Barrier()

    cov = numpy.linalg.inv(lhsr)
    val = numpy.matmul(cov,rhsr)

    print '-----------------------------------------------------'
    print val
    print '-----------------------------------------------------'

    dmd1   = numpy.matmul(lhsr,val)
    chi1   = numpy.matmul(numpy.transpose(val),dmd1)
    sigma1 = numpy.sqrt(chi1)

    print 'Assuming all data: ',sigma1,' sigma'
    print '-----------------------------------------------------'

    trim   = 8
    dmd2   = numpy.matmul(lhsr[:-trim,:-trim],val[:-trim])
    chi2   = numpy.matmul(numpy.transpose(val[:-trim]),dmd2)
    sigma2 = numpy.sqrt(chi2)

    print 'Trimming the data some: ',sigma2,' sigma'
    print '-----------------------------------------------------'

