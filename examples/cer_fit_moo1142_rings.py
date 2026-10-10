import numpy
import numpy as np
from matplotlib import pyplot as plt
import minkasi
from minkasi.parallel import comm, have_mpi
import time
import glob
from importlib import reload
reload(minkasi)
plt.ion()

mydir = '/home/scratch/cromero/mustang/MUSTANG2/Reductions/MOO_1142/'
outdir = mydir+'MinkasiOutputs/'
tod_names=glob.glob(mydir+'TS_EaCMS0f0_51_18_Jun_2025/Signal_TOD*.fits')
ringoutdir=outdir+'IterRings/'
tag = '_21Aug2026'

### Choose the input file (based on maximum iterations and number of TODs)
maxiter = 25
ntods = 106  # maximum number is 106
mis = repr(maxiter)
nts = repr(ntods)
Gaussian_file = mydir+'MinkasiOutputs/moo_1142_lowestChi_Gaussians_16July2026_v0_'+mis+'iters_'+nts+'scans.npy'

### Elliptical parameters
aa = 1.0  # No "major" axis scaling. X = "major" axis
bb = 1.0  # No "minor" axis scaling. Y = "major" axis.
rot = 0.0  # No rotation. X is along RA.

### Now reselect iterations and #TODs for the rings.
#maxiter = 5
npass = 10
ntods = 106  # maximum number is 106
npstr = repr(npass)
nts = repr(ntods)
RingsOutputs = 'moo_1142_rings'
nclus = 2
tag = tag+'_'+nts+'scans'

simclus  = False    # Sets nsrcs to 0
sub_cm   = False
chisq    = 3.0     # Try adjusting due to simulations

svdfwhm = 20        # Number of integrations over which to smooth.
svdcut  = False      # Run set_noise_smoothed_svd_v2()
svdstr  = '_svd'+str(svdfwhm)+"_"+"{0:.1f}".format(chisq)+"Chi"        # TBS = trimmed bad scans
#npass=10
kstart=1      # Normally 1.
sig2fwhm = numpy.sqrt(8.0 * numpy.log(2.0)) # FWHM <-> sigma (Gaussian width)
fw2sr    = numpy.pi/180.0/3600.0/sig2fwhm   # Converts FWHM in arcsec to sig in radians
rfwhm    = 10.7       # Set to 0 if you want to use FWHM (sigmas) from fit_Gauss
doconv   = True
m2fwhm   = 10.7       # Beam to convolve rings by -- to recover deconvolved profile
#rfwhm    = 0       # Set to 0 if you want to use FWHM (sigmas) from fit_Gauss
pord     = 2        # Polynomial order.
FPFPO    = True
noslope  = False     # Should usually be false...but I'm trying to replicate old results (13 May 2019)


#edges=numpy.asarray([0,5,15,25,35,45,55,70,85,100,120,140,160,180,200,220,240],dtype='double')
#edges=numpy.asarray([0,5,15,25,35,45,55,70,85,100,120,150,180,240],dtype='double')
edges=numpy.asarray([0,5,15,30,45,60,90,120,150,180],dtype='double')
#clus_cent = numpy.asarray([155.91545,4.1835417,0.0004,0.0],dtype='double')

use_Gaussian_center = False
initial_map = mydir+"MinkasiOutputs/Minkasi_naivePCG_map.fits"

##############################################################################################

tod_names=tod_names[minkasi.myrank::minkasi.nproc]
todvec=minkasi.tods.TodVec()
origvec=minkasi.tods.TodVec()

for fname in tod_names[:ntods]:
    t1=time.time()
    dat=minkasi.tods.io.read_tod_from_fits(fname)
    t2=time.time()
    ### Downsample is optional. Ideally MUSTANG-2 TODs are appropriately downsampled.
    #minkasi.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
    minkasi.tods.processing.truncate_tod(dat)    #since our length changed, make sure we have a happy length

    #figure out a guess at common mode #and (assumed) linear detector drifts/offset
    #drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.

    #dd=minkasi.tods.processing.fit_cm_plus_poly(dat['dat_calib'])  
    
    #dat['dat_calib']=dd
    t3=time.time()
    tod=minkasi.tods.Tod(dat)
    todvec.add_tod(tod)
    print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)

#make a template map with desired pixel size an limits that cover the data
#todvec.lims() is MPI-aware and will return global limits, not just
#the ones from private TODs
lims=todvec.lims()
pixsize=2.0/3600*numpy.pi/180  # 2 arcseconds, stored as radians
map=minkasi.maps.SkyMap(lims,pixsize)

#once we have a map, we can figure out the pixellization of the data.  Save that
#so we don't have to recompute.  Also calculate a noise model.  The one here
#(and currently the only supported one) is to rotate the data into SVD space, then
#smooth the power spectrum of each mode.  Other models would not be hard
#to implement.  The smoothing could do with a bit of tuning as well.


for tod in todvec.tods:
    ipix=map.get_pix(tod)
    tod.info['ipix']=ipix
    #tod.set_noise_smoothed_svd()
    tod.set_noise(minkasi.mapmaking.noise.NoiseSmoothedSVD)
    
fitp=numpy.load(Gaussian_file)
npar_src=4
coords=numpy.zeros([fitp.shape[0]//4,2])
coords[:,0]=fitp[0::npar_src]
coords[:,1]=fitp[1::npar_src]
coords=coords*180/numpy.pi
pixcents=map.wcs.wcs_world2pix(coords,1)
nsrc=pixcents.shape[0]-nclus
print('##########################################')
print(coords, pixcents)
print('##########################################')
clus1_cent = coords[0,:]
clus2_cent = coords[1,:]

npxc = edges.shape[0]-1  # Number of parameters per cluster
npar = npxc*nclus
mm = map.copy()

val = numpy.zeros(npar+nsrc)     # Initiate with zero amplitudes
zval = val.copy()

lhsr = numpy.zeros([npar+nsrc,npar+nsrc])
rhsr = numpy.zeros(npar+nsrc)
srcp = numpy.zeros(npar_src)

#get the hit count map.  We use this as a preconditioner
#which helps small-scale convergence quite a bit.
hits=minkasi.mapmaking.noise.make_hits(todvec,map)

#setup the mapset.  In general this can have many things
#in addition to map(s) of the sky, but for now we'll just 
#use a single skymap.
mapset=minkasi.maps.Mapset()
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


def mydot(m1,m2):
    tot=0.0
    tot=tot+0.5*numpy.sum(m1[:,0]*m2[:,0])
    tot=tot+0.5*numpy.sum(m1[:,-1]*m2[:,-1])
    tot=tot+numpy.sum(m1[:,1:-1]*m2[:,1:-1])
    return tot


if kstart > 1:
    lhs   = numpy.load(ringoutdir+RingsOutputs+'_curve'+tag+'_pass_'+repr(kstart-1)+'.npy')
    rhs   = numpy.load(ringoutdir+RingsOutputs+'_deriv'+tag+'_pass_'+repr(kstart-1)+'.npy')
    edges = numpy.load(ringoutdir+RingsOutputs+'_edges'+tag+'_pass_'+repr(kstart-1)+'.npy')
    cov = numpy.linalg.inv(lhs)
    val = numpy.matmul(cov,rhs)

#rings1=minkasi.tools.rings.make_rings_wSlope(edges,clus1_cent,map,fwhm=11.1)
#rings2=minkasi.tools.rings.make_rings_wSlope(edges,clus2_cent,map,fwhm=11.1)
    
if noslope: 
    rings1=minkasi.tools.rings.make_rings_wSlope(edges,coords[0,:],zval[:npxc],map,fwhm=m2fwhm,aa=aa,bb=bb,rot=rot)  # previously 11.1
    rings2=minkasi.tools.rings.make_rings_wSlope(edges,coords[0,:],zval[npxc:npxc*2],map,fwhm=m2fwhm,aa=aa,bb=bb,rot=rot)  # previously 11.1
else:
    #rings1=minkasi.tools.rings.make_rings_wSlope(edges,coords[0,:],val[:npxc],map,fwhm=m2fwhm,aa=aa,bb=bb,rot=rot,doconv=doconv)
    rings1=minkasi.tools.rings.make_rings_wSlope(edges,coords[0,:],val[:npxc],map,fwhm=m2fwhm,aa=aa,bb=bb,rot=rot)
    rings2=minkasi.tools.rings.make_rings_wSlope(edges,coords[0,:],val[npxc:npxc*2],map,fwhm=m2fwhm,aa=aa,bb=bb,rot=rot)
#import pdb;pdb.set_trace()
    #lhs=numpy.zeros([npar+nsrc,npar+nsrc])
    #rhs=numpy.zeros(npar+nsrc)
    #srcp=numpy.zeros(npar_src)
    
for k in range(kstart,npass+1):
    lhs=numpy.zeros([npar+nsrc,npar+nsrc])
    rhs=numpy.zeros(npar+nsrc)
    srcp=numpy.zeros(npar_src)
    print('working on pass ',k)
    print('And the tag is: ',tag)
    #let's re-estimate the noise with the current model subtracted off 
    mymap=map.copy()
    for tod,orig in zip(todvec.tods,origvec.tods):
        ### Copied here just in case...but it should just be called once (before the loop)
        #ipix=map.get_pix(tod)
        #tod.info['ipix']=ipix               
        trucop=orig.copy()
        trudat=trucop.info['dat_calib'] # There is a pun in these
        whodat=trucop.info['dat_calib'] # variables.
        tmp=numpy.zeros(trudat.shape)
        for j in range(nclus):
            myrings = rings1 if j == 0 else rings2
            for i in range(npxc):
                mm.map[:]=myrings[i,:,:]
                dat=numpy.zeros(tod.info['dat_calib'].shape)
                mm.map2tod(tod,dat)
                whodat = whodat-tmp
##################################################################################################
                #dat_filt=tod.apply_noise(dat)
                #model_ts[i+j*npxc]=dat
                #model_ts_filt[i+j*npxc]=dat_filt            
        #for i in range(npar):
        #    mymap.map[:]=rings[i,:,:]*val[i] # Value (amplitude of the ring) applied here!
        #    tmp=numpy.zeros(tod.info['dat_calib'].shape)
        #    mymap.map2tod(tod,tmp)
        #    whodat = whodat-tmp
##################################################################################################
        
        for i in range(nsrc):
            srcp[:]=fitp[(i+1)*npar_src:(i+2)*npar_src]
            srcp[-1]=val[i+npar]  # Value (amplitude of the source) applied here!
            if rfwhm*fw2sr > srcp[-2]: srcp[-2] = rfwhm*fw2sr  # into radians, sigma
            derivs,pred=minkasi.fitting.core.derivs_from_gauss_c(srcp,tod)
            whodat = whodat-pred

        ring_tods = trudat - whodat  # This is actually "model_tods", as it includes the point sources.
        if not FPFPO or k == 0: 
            if sub_cm:
                dd,fcm,cm =minkasi.tods.processing.fit_cm_plus_poly(whodat,full_out=True,ord=pord)  
                dd = dd-fcm
            else:
                dd=minkasi.tods.processing.fit_cm_plus_poly(whodat,ord=pord)  
        else:
            dd=minkasi.tods.processing.fit_cm_plus_poly(trudat,ord=pord) - ring_tods

        tod.info['dat_calib'][:]=dd  # Set the noise with all known signal subtracted.
        tod.set_noise_smoothed_svd(fwhm=svdfwhm,chisq=chisq) # Done here.
        tod.info['dat_calib'][:]=dd + ring_tods  # Now restore the known signal... but no polynomial.


    #    rings=minkasi.make_rings_wSlope(edges,coords[0,:],zval,map,fwhm=m2fwhm,aa=aa,bb=bb,rot=rot)
    #else:
    #    rings=minkasi.make_rings_wSlope(edges,coords[0,:],val,map,fwhm=m2fwhm,aa=aa,bb=bb,rot=rot)
    #rings=minkasi.make_rings(edges,coords[0,:],map,fwhm=m2fwhm)
    for tod in todvec.tods:
        model_ts=[None]*(npar+nsrc)
        model_ts_filt=[None]*(npar+nsrc)
        for j in range(nclus):
            myrings = rings1 if j == 0 else rings2
            for i in range(npxc):
                mm.map[:]=myrings[i,:,:]
                #for i in range(npar):
                #    mm.map[:]=myrings[i,:,:]
                dat=numpy.zeros(tod.info['dat_calib'].shape)
                mm.map2tod(tod,dat)
                dat_filt=tod.apply_noise(dat)
                model_ts[i+j*npxc]=dat
                model_ts_filt[i+j*npxc]=dat_filt
        for i in range(nsrc):
            srcp[:]=fitp[(i+1)*npar_src:(i+2)*npar_src]
            srcp[-1]=1.0
            derivs,pred=minkasi.fitting.core.derivs_from_gauss_c(srcp,tod)
            pred_filt=tod.apply_noise(pred)
            model_ts[i+npar]=pred
            model_ts_filt[i+npar]=pred_filt
        for i in range(npar+nsrc):
            if model_ts[i] is None:
                import pdb;pdb.set_trace()
            if model_ts_filt[i] is None:
                import pdb;pdb.set_trace()
                
            rhs[i]=rhs[i]+mydot(model_ts_filt[i],tod.info['dat_calib'])
            for j in range(i,npar+nsrc):
                el_lhs = mydot(model_ts[i],model_ts_filt[j])
                if el_lhs == 0:
                    print('oh uh')
                    import pdb;pdb.set_trace()
                lhs[i,j]=lhs[i,j]+el_lhs
                lhs[j,i]=lhs[i,j]

        print('finished tod')

    #print(lhs.shape)
    #import pdb;pdb.set_trace()
        
    if have_mpi:
        lhs=comm.allreduce(lhs)
        rhs=comm.allreduce(rhs)
    #lhs=minkasi.parallel.comm.allreduce(lhs)
    #rhs=minkasi.parallel.comm.allreduce(rhs)
    if minkasi.myrank==0:
        numpy.save(ringoutdir+RingsOutputs+'_curve'+tag+'_pass_'+repr(k)+'.npy',lhs)
        numpy.save(ringoutdir+RingsOutputs+'_deriv'+tag+'_pass_'+repr(k)+'.npy',rhs)
        numpy.save(ringoutdir+RingsOutputs+'_edges'+tag+'_pass_'+repr(k)+'.npy',edges)
    #minkasi.comm.Barrier()
    if have_mpi:
        comm.Barrier()

    print(numpy.diag(lhs))

    cov = numpy.linalg.inv(lhs)
    val = numpy.matmul(cov,rhs)

    print('-----------------------------------------------------')
    print(val)
    print('-----------------------------------------------------')

    dmd1   = numpy.matmul(lhs,val)
    chi1   = numpy.matmul(numpy.transpose(val),dmd1)
    sigma1 = numpy.sqrt(chi1)
    
    print('Assuming all data: ',sigma1,' sigma')
    print('-----------------------------------------------------')
    
    trim   = nsrc
    if trim > 0:
        dmd2   = numpy.matmul(lhs[:-trim,:-trim],val[:-trim])
        chi2   = numpy.matmul(numpy.transpose(val[:-trim]),dmd2)
        sigma2 = numpy.sqrt(chi2)
    
        print('Trimming the data some: ',sigma2,' sigma')
    print('-----------------------------------------------------')

    if noslope:
        rings1=minkasi.tools.rings.make_rings_wSlope(edges,coords[0,:],zval[:npxc],map,fwhm=m2fwhm,aa=aa,bb=bb,rot=rot)  # previously 11.1
        rings2=minkasi.tools.rings.make_rings_wSlope(edges,coords[0,:],zval[npxc:npxc*2],map,fwhm=m2fwhm,aa=aa,bb=bb,rot=rot)  # previously 11.1
    else:
        #rings1=minkasi.tools.rings.make_rings_wSlope(edges,coords[0,:],val[:npxc],map,fwhm=m2fwhm,aa=aa,bb=bb,rot=rot,doconv=doconv)
        rings1=minkasi.tools.rings.make_rings_wSlope(edges,coords[0,:],val[:npxc],map,fwhm=m2fwhm,aa=aa,bb=bb,rot=rot)
        rings2=minkasi.tools.rings.make_rings_wSlope(edges,coords[0,:],val[npxc:npxc*2],map,fwhm=m2fwhm,aa=aa,bb=bb,rot=rot)


