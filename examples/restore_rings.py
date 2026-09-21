from datetime import datetime
import yaml
from pathlib import Path
import os
import numpy
import minkasi
from matplotlib import pyplot as plt
import glob
import argparse
plt.ion()

def main(config_file):

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    mydir = config["paths"]["base_dir"]
    tod_folder = config["paths"]["tod_folder"]
    gauss_folder = config["paths"]["gauss_folder"]
    ring_folder = config["paths"]["ring_folder"]
    myglob = config["paths"]["glob"]
    full_glob = os.path.join(mydir,tod_folder,myglob)
    tod_names = glob.glob(full_glob)  # Actually, keep it, even if not reading them in.
    username = os.environ.get("USER")
    output_base = config["paths"]["output_folder"]
    date_string = datetime.now().strftime("%d%b%Y")   # Today's date
    target = config["general"]["target_name"]
    tshoot = config["general"]["tshoot"]
    auto_trim = config["general"]["auto_trim"]
    verbose = config["general"]["verbose"]
    npartitions = config["fitpars"]["npartitions"]     # True if looking at SZ signal

    if auto_trim:
        bad_tod, _ = minkasi.tools.presets_by_source.get_bad_tods(target)
        tod_names = minkasi.tods.io.cut_blacklist(tod_names, bad_tod)

    if ring_folder.lower() != "most_recent":
        part_index = 0   # Stand-in
        restore_ring_partition(config, part_index)
        

    myfig = plt.figure(1, figsize=(5+npartitions/4.0,4+npartitions/5.0), dpi=200)
    myax = myfig.add_subplot(111)

    vals = None
    errs = None
    for pp in range(npartitions):

        if config["general"]["group_parts"] == False:     # True if looking at SZ signal
            myfig.clf()
            myax = myfig.add_subplot(111)
        outdir, RingsOutputs, vals, errs = restore_ring_partition(config, pp, tod_names, myfig, myax, vals=vals, errs=errs)
        if config["general"]["group_parts"] == False:     # True if looking at SZ signal
            myax.legend()
            myfig.tight_layout()
            myfig.savefig( os.path.join(outdir, RingsOutputs+"_Individual_SBrightnessProfile.png") )
        
    if config["general"]["group_parts"]:     # True if looking at SZ signal
        myax.legend()
        myfig.tight_layout()
        myfig.savefig( os.path.join(outdir, RingsOutputs+"_Grouped_SBrightnessProfiles.png") )
    npvals = numpy.array(vals)
    nperrs = numpy.array(errs)

    if npartitions > 1:
        scatter = numpy.std(vals, axis=0)
        avg_unc = numpy.mean(nperrs, axis=0)
        offset = scatter / avg_unc
        amean = numpy.mean(offset)
        gmean = numpy.exp(numpy.mean(numpy.log(offset)))
        print("Arithmetic mean of offsets: ",amean)
        print("Geometric mean of offsets: ",gmean)
        #import pdb;pdb.set_trace()
        amsqr = amean**2
        gmsqr = gmean**2
        print("Suggested increase in reduced chi-squared adjustment (rchisq in yaml file) of: ",
              (amsqr+gmsqr)/2.0)

    print("Done")
    
def restore_ring_partition(config, part_index, tod_names, myfig, myax, vals=None, errs=None):
    
    mydir = config["paths"]["base_dir"]
    tod_folder = config["paths"]["tod_folder"]
    gauss_folder = config["paths"]["gauss_folder"]
    ring_folder = config["paths"]["ring_folder"]
    #myglob = config["paths"]["glob"]
    #full_glob = os.path.join(mydir,tod_folder,myglob)
    #tod_names = glob.glob(full_glob)  # Actually, keep it, even if not reading them in.
    username = os.environ.get("USER")
    output_base = config["paths"]["output_folder"]
    date_string = datetime.now().strftime("%d%b%Y")   # Today's date
    target = config["general"]["target_name"]
    tshoot = config["general"]["tshoot"]
    auto_trim = config["general"]["auto_trim"]
    verbose = config["general"]["verbose"]
    npartitions = config["fitpars"]["npartitions"]     # True if looking at SZ signal

    npass = config["fitpars"]["npass"]
    ntods_in = config["fitpars"]["ntods"]
    simclus = config["fitpars"]["simclus"]
    sub_cm = config["fitpars"]["sub_cm"]
    rchisq = config["fitpars"]["rchisq"]
    svdfwhm = config["fitpars"]["svdfwhm"]        # Number of integrations over which to smooth.
    svdcut = config["fitpars"]["svdcut"]      # Run set_noise_smoothed_svd_v2()
    svdstr = '_svd'+str(svdfwhm)+"_"+"{0:.1f}".format(rchisq)+"Chi"        # TBS = trimmed bad scans
    kstart = config["fitpars"]["kstart"]      # Normally 1.
    rfwhm = config["fitpars"]["rfwhm"]        # Set to 0 if you want to use FWHM (sigmas) from fit_Gauss
    doconv = config["fitpars"]["doconv"]
    m2fwhm = config["fitpars"]["m2fwhm"]      # Beam to convolve rings by -- to recover deconvolved profile
    #rfwhm = 0       # Widths for point sources! (May be different from convolution kernel)
    pord = config["fitpars"]["pord"]        # Polynomial order.
    FPFPO = config["fitpars"]["FPFPO"]
    noslope = config["fitpars"]["noslope"]     # Should usually be false 
    pix_arcsec = config["fitpars"]["pix_arcsec"]     # Should usually be false 
    decrement = config["fitpars"]["decrement"]     # True if looking at SZ signal
    npartitions = config["fitpars"]["npartitions"]     # True if looking at SZ signal

    tod_names=tod_names[part_index::npartitions]
    ntods_available = len(tod_names)  # Update here.
    
    if ntods_in is None:
        ntods = len(tod_names)
    else:
        ntods = ntods_in if ntods_in <= ntods_available else ntods_available

    mis = repr(npass)
    nts = repr(ntods)

    #output_specific = "_".join([date_string, mis+'pass', nts+'scans'])
    #gaussdir = os.path.join("/home/scratch", username, output_base, target, "Gaussian_Fits",
    #                        gauss_folder)
    part_str = "partition"+repr(part_index+1)+"-of-"+repr(npartitions)
    output_search = "_".join(['*', mis+'pass', nts+'scans', part_str])
    #output_search = "_".join(['*', mis+'pass', nts+'scans'])
    outdir_base = os.path.join("/home/scratch", username, output_base, target, "Ring_Fits")
    base = Path(outdir_base)

    matches = base.glob(output_search)
    #print(matches)
    #print(outdir_base)
    #print(output_search)
    if ring_folder.lower() == "most_recent":
        if matches:
            most_recent = max(matches, key=lambda p: p.stat().st_mtime)
        else:
            most_recent = None
            # The entire point is to restore the files. If they don't exist, no sense continuing.
            raise SystemExit(
                f"ERROR: No output directories matching {output_search} found in {base}"
            )
        
        #raise FileNotFoundError(
        #    f"No output directories matching {output_search} found in {base}."
        #)
        outdir = os.path.join(outdir_base, most_recent)
    else:        
        outdir = os.path.join(outdir_base, ring_folder)
    
    mm1is = repr(npass-1)
    RingsOutputs = '_'.join([target,"Rings"])

    lhsm1 = numpy.load(os.path.join(outdir, RingsOutputs+'_curve'+'_pass_'+mm1is+'.npy'))
    rhsm1 = numpy.load(os.path.join(outdir, RingsOutputs+'_deriv'+'_pass_'+mm1is+'.npy'))
    lhs = numpy.load(os.path.join(outdir, RingsOutputs+'_curve'+'_pass_'+mis+'.npy'))
    rhs = numpy.load(os.path.join(outdir, RingsOutputs+'_deriv'+'_pass_'+mis+'.npy'))
    edge_files = glob.glob(os.path.join(outdir, RingsOutputs+'_edges_set*.npy') )
    edge_set = []
    for edge_file in edge_files:
        edges_saved = numpy.load(edge_file)
        edge_set.append(edges_saved)
    #edges_saved = numpy.load(os.path.join(outdir, RingsOutputs+'_edges'+'_pass_'+mm1is+'.npy'))
    cov = numpy.linalg.inv(lhs)
    val = numpy.matmul(cov,rhs)
    covm1 = numpy.linalg.inv(lhsm1)
    valm1 = numpy.matmul(covm1,rhsm1)
    
    print("Loaded numpy files")
    #import pdb; pdb.set_trace()
    rescale = 1e6                                     # If I might want to change to mK? IDK.

    starti=0
    cov_diag = numpy.diag(cov)
    unc_diag = numpy.sqrt(cov_diag)
    nclus = len(edge_set)
    
    if vals is None:
        vals=[val]
        errs=[unc_diag]
    else:
        vals.append(val)
        errs.append(unc_diag)
        
    for j, edges in enumerate(edge_set):
        nring = len(edges)-1
        myvals, slopes = minkasi.tools.rings.get_ring_slopes(edges, valm1[starti:starti+nring], decrement)
        maxedge = numpy.max(edges)
        npts = int(numpy.round(maxedge * 2.0 / pix_arcsec))+1
        radii = numpy.linspace(0, maxedge, npts)  # Presumably every 1 arcsecond.
        brightness = numpy.zeros( npts )
                                                             
        for i in range(nring):
            ii = starti+i
            annular_ind = (radii >= edges[i])*(radii < edges[i + 1])
            rfromin = radii - edges[i]
            initline = rfromin[annular_ind] * slopes[i]
            brightness[annular_ind] = val[ii] * (myvals[i] - initline) / myvals[i]
            if tshoot:
                print("Ring ", i, " prev. value: ", myvals[i], " current value: ", val[ii])
                if numpy.any(numpy.abs(brightness[annular_ind]*1e3) > 2.0):
                    print(brightness[annular_ind])
                    import pdb; pdb.set_trace()
        
        ring_amps = val[starti:starti+nring]
        ring_uncs = unc_diag[starti:starti+nring]
        cind = part_index*nclus + j
        
        myax.plot(radii, brightness*rescale, label="Cluster "+repr(j+1), color="C"+repr(cind))
        myax.set_xlabel("Radius (arcseconds)")
        myax.set_ylabel(r"Brightness ($\mu$K$_{\rm RJ}$")  # hard-coded for MUSTANG-2 data.
        #print(j, edges.shape, nring, ring_amps.shape, ring_uncs)
        #import pdb;pdb.set_trace()
        myax.errorbar(edges[:nring], ring_amps*rescale, yerr=ring_uncs*rescale, capsize=5,
                      linestyle="none", color="C"+repr(cind))
        
        starti = starti + nring

    return outdir, RingsOutputs, vals, errs
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML configuration file")
    args = parser.parse_args()

    main(args.config)    
       
