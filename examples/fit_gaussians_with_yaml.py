import numpy as np
from matplotlib import pyplot as plt
import minkasi
import time
import glob
import yaml
import os
import argparse
from pathlib import Path
from importlib import reload
from datetime import datetime
import shutil

reload(minkasi)
#plt.ion()

def main(config_file):

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    mydir = config["paths"]["base_dir"]
    tod_folder = config["paths"]["tod_folder"]
    myglob = config["paths"]["glob"]
    full_glob = os.path.join(mydir,tod_folder,myglob)
    tod_names = glob.glob(full_glob)
    username = os.environ.get("USER")
    output_base = config["paths"]["output_folder"]
    date_string = datetime.now().strftime("%d%b%Y")
    target = config["general"]["target_name"]
    
    auto_trim = config["general"]["auto_trim"]
    verbose = config["general"]["verbose"]

    if auto_trim:
        bad_tod, _ = minkasi.tools.presets_by_source.get_bad_tods(target)
        tod_names = minkasi.tods.io.cut_blacklist(tod_names, bad_tod)

    ### Fit initial features as Gaussians; notably with the locations, amplitude, and widths free.
    maxiter = config["fitpars"]["maxiter"]
    ntods_in = config["fitpars"]["ntods"]
    og_step = config["fitpars"]["og_step"]
    ntods = len(tod_names) if ntods_in is None else ntods_in
    
    mis = repr(maxiter)
    nts = repr(ntods)

    output_specific = "_".join([date_string, mis+'iters', nts+'scans'])
    outdir = os.path.join("/home/scratch", username, output_base, target, "Gaussian_Fits",
                          output_specific)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    config_file = Path(config_file)
    shutil.copy2(config_file, Path(outdir) / config_file.name)

    Gaussian_out = os.path.join(outdir, "_".join([target,"Gaussians.npy"]))
    Curve_out = os.path.join(outdir, "_".join([target,"Curve.npy"]))

    cluster_ras = np.asarray(config["startvars"]["cluster_ras"], dtype=np.float64)        # In degrees
    cluster_decs = np.asarray(config["startvars"]["cluster_decs"], dtype=np.float64)      # In degrees
    cluster_amps = np.asarray(config["startvars"]["cluster_amps"], dtype=np.float64)      # In map units (e.g. Kelvin_RJ)
    cluster_widths = np.asarray(config["startvars"]["cluster_widths"], dtype=np.float64)  # In arcseconds

    ptsrc_ras = np.asarray(config["startvars"]["ptsrc_ras"], dtype=np.float64)            # In degrees
    ptsrc_decs = np.asarray(config["startvars"]["ptsrc_decs"], dtype=np.float64)          # In degrees
    ptsrc_amps = np.asarray(config["startvars"]["ptsrc_amps"], dtype=np.float64)          # In map units (e.g. Kelvin_RJ)
    ptsrc_widths = np.asarray(config["startvars"]["ptsrc_widths"], dtype=np.float64)      # In arcseconds

    d2r = np.pi/180

    np_src = 4  # 4 parameters per Gaussian
    n_gauss = len(cluster_ras) + len(ptsrc_ras)
    gaussp = np.zeros(np_src*n_gauss)  # Array for all parameters

##########-----------------------------------------##################################
    gaussp[0::np_src] = np.hstack([cluster_ras*d2r, ptsrc_ras*d2r])
    gaussp[1::np_src] = np.hstack([cluster_decs*d2r, ptsrc_decs*d2r])
    #gaussp[2::np_src] = np.hstack([cluster_amps, ptsrc_amps])
    gaussp[2::np_src] = np.hstack([cluster_widths*d2r/3600.0, ptsrc_widths*d2r/3600.0])  # Originally in arcseconds.
    gaussp[3::np_src] = np.hstack([cluster_amps, ptsrc_amps])

    npar=np.hstack([np_src]*n_gauss) #and the fitter needs to know how many per function
    funs = [minkasi.fitting.core.derivs_from_gauss_c]*n_gauss
    to_fit=np.ones(np_src*n_gauss,dtype='bool')

    #fit_widths = config["fitpars"]["fit_widths"]  # True or False
    to_fit[2::np_src] = config["fitpars"]["fit_widths"] 
    to_fit[0::np_src] = config["fitpars"]["fit_cen"] 
    to_fit[1::np_src] = config["fitpars"]["fit_cen"] 

    initial_map = os.path.join(outdir,"Minkasi_naivePCG_map.fits")

##############################################################################################
    
    tod_names=tod_names[minkasi.myrank::minkasi.nproc]
    todvec=minkasi.tods.TodVec()

    for fname in tod_names[:ntods]:
        t1=time.time()
        dat=minkasi.tods.io.read_tod_from_fits(fname)
        t2=time.time()
        ### Downsample is optional. Ideally MUSTANG-2 TODs are appropriately downsampled.
        #minkasi.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
        minkasi.tods.processing.truncate_tod(dat)    #since our length changed, make sure we have a happy length

        #figure out a guess at common mode #and (assumed) linear detector drifts/offset
        #drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
        dd=minkasi.tods.processing.fit_cm_plus_poly(dat['dat_calib'])  
    
        dat['dat_calib']=dd
        t3=time.time()
        tod=minkasi.tods.Tod(dat)
        todvec.add_tod(tod)
        print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)

    #make a template map with desired pixel size an limits that cover the data
    #todvec.lims() is MPI-aware and will return global limits, not just
    #the ones from private TODs
    lims=todvec.lims()
    pixsize=2.0/3600*np.pi/180  # 2 arcseconds, stored as radians
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


    t1=time.time()
    pars_fit,chisq,curve,errs=minkasi.fitting.fit_timestreams_with_derivs_manyfun(funs,gaussp,npar,todvec,to_fit, maxiter=maxiter, og_step=og_step, verbose=verbose)
    t2=time.time()

    print(pars_fit)
    np.save(Gaussian_out,pars_fit)
    np.save(Curve_out,curve)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML configuration file")
    args = parser.parse_args()

    main(args.config)    
       
