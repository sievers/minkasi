"""
Functions for loading TODs from disk.
"""
from typing import Any, Iterable, Optional

import numpy as np
from astropy.io import fits

from . import Tod, TodVec

try:
    have_qp = True
    import qpoint as qp
except:
    have_qp = False


def _get_type(nbyte):
    if nbyte == 8:
        return np.dtype("float64")
    if nbyte == 4:
        return np.dtype("float32")
    if nbyte == -4:
        return np.dtype("int32")
    if nbyte == -8:
        return np.dtype("int64")
    if nbyte == 1:
        return np.dtype("str")
    print("Unsupported nbyte " + repr(nbyte) + " in get_type")
    return None


def read_tod_from_fits_cbass(
    fname: str,
    dopol: bool = False,
    lat: float = 37.2314,
    lon: float = -118.2941,
    v34: bool = True,
    nm20: bool = False,
) -> dict:
    """
    Read a CBASS TOD from a FITS file.

    Parameters
    ----------
    fname : str
        The file to read.
    dopol : bool, default: False
        If True load Q and U for this TOD.
    lat : float, default: 37.2314
        Latitude of CBASS in degrees.
    lon : float, default: -118.2941
        Longitude of CBASS in degrees.
    v34 : bool, default: True
        Apply the v34 sign convention for polarization.
    nm20 : bool, default: False
        Try to load additional cuts from the file.
        Assumed to bo the third HDU.

    Returns
    -------
    dat : dict
        Dict containing TOD info.
        Can be passed into the Tod class.
    """
    hdul = fits.open(fname)
    raw = hdul[1].data
    ra = raw["RA"]
    dec = raw["DEC"]
    # flag=raw['FLAG']
    I = 0.5 * (raw["I1"] + raw["I2"])

    mjd = raw["MJD"]
    tvec = (mjd - 2455977.5 + 2400000.5) * 86400 + 1329696000
    # (mjd-2455977.5)*86400+1329696000;
    dt = np.median(np.diff(tvec))

    dat = {}
    dat["dx"] = np.reshape(np.asarray(ra, dtype="float64"), [1, len(ra)])
    dat["dy"] = np.reshape(np.asarray(dec, dtype="float64"), [1, len(dec)])
    dat["dt"] = dt
    dat["ctime"] = tvec
    if dopol:
        dat["dx"] = np.vstack([dat["dx"], dat["dx"]])
        dat["dy"] = np.vstack([dat["dy"], dat["dy"]])
        Q = 0.5 * (raw["Q1"] + raw["Q2"])
        U = 0.5 * (raw["U1"] + raw["U2"])
        dat["dat_calib"] = np.zeros([2, len(Q)])
        if v34:  # We believe this is the correct sign convention for V34
            dat["dat_calib"][0, :] = -U
            dat["dat_calib"][1, :] = Q
        else:
            dat["dat_calib"][0, :] = Q
            dat["dat_calib"][1, :] = U
        az = raw["AZ"]
        el = raw["EL"]
        # JLS- changing default az/el to radians and not degrees in TOD
        dat["az"] = az * np.pi / 180
        dat["el"] = el * np.pi / 180

        # dat['AZ']=az
        # dat['EL']=el
        # dat['ctime']=tvec
        dat["mask"] = np.zeros([2, len(Q)], dtype="int8")
        dat["mask"][0, :] = 1 - raw["FLAG"]
        dat["mask"][1, :] = 1 - raw["FLAG"]
        if have_qp:
            Q = qp.QPoint(accuracy="low", fast_math=True, mean_aber=True, num_threads=4)
            # q_bore = Q.azel2bore(dat['AZ'], dat['EL'], 0*dat['AZ'], 0*dat['AZ'], lon*np.pi/180, lat*np.pi/180, dat['ctime'])
            q_bore = Q.azel2bore(az, el, 0 * az, 0 * az, lon, lat, dat["ctime"])
            q_off = Q.det_offset(0.0, 0.0, 0.0)
            # ra, dec, sin2psi, cos2psi = Q.bore2radec(q_off, ctime, q_bore)
            ra, dec, sin2psi, cos2psi = Q.bore2radec(q_off, tvec, q_bore)
            tmp = np.arctan2(sin2psi, cos2psi)
            tmp = (
                tmp - np.pi / 2
            )  # this seems to be needed to get these coordinates to line up with
            # the expected, in IAU convention I believe.  JLS Nov 12 2020
            # dat['twogamma_saved']=np.arctan2(sin2psi,cos2psi)
            dat["twogamma_saved"] = np.vstack([tmp, tmp + np.pi / 2])
            # print('pointing rms is ',np.std(ra*np.pi/180-dat['dx']),np.std(dec*np.pi/180-dat['dy']))
            dat["ra"] = ra * np.pi / 180
            dat["dec"] = dec * np.pi / 180
    else:
        dat["dat_calib"] = np.zeros([1, len(I)])
        dat["dat_calib"][:] = I
        dat["mask"] = np.zeros([1, len(I)], dtype="int8")
        dat["mask"][:] = 1 - raw["FLAG"]

    dat["pixid"] = [0]
    dat["fname"] = fname

    if nm20:
        # kludget to read in bonus cuts, which should be in f[3]
        try:
            raw = hdul[3].data
            dat["nm20_start"] = raw["START"]
            dat["nm20_stop"] = raw["END"]
            print(dat.keys())
            nm20_dat = 0 * dat["mask"]
            start = dat["nm20_start"]
            stop = dat["nm20_stop"]
            for i in range(len(start)):
                nm20_dat[:, start[i] : stop[i]] = 1
            dat["mask"] = dat["mask"] * nm20_dat
        except:
            print("missing nm20 for ", fname)

    hdul.close()
    return dat


def read_tod_from_fits(
    fname: str, hdu: int = 1, branch: Optional[float] = None
) -> dict:
    """
    Read a TOD from a FITS file.
    This function nominally runs on MUSTANG TODs.

    Parameters
    ----------
    fname : str
        The path of the file to be loaded.
    hdu : int, default: 1
        The index of the HDU with the TOD.
    branch : float | None, default: None
        Branch in degrees that RA was corrected to.
        Set to None if RA wasnt correctd to a branch.

    Returns
    -------
    dat : dict
        Dict containing TOD info.
        Can be passed into the Tod class.
    """
    hdul = fits.open(fname)
    raw = hdul[hdu].data

    if raw.names is None:
        raise ValueError("TOD seems to be empty")
    # print 'sum of cut elements is ',np.sum(raw['UFNU']<9e5)
    calinfo: dict[str, Any] = {"calinfo": False}
    try:  # read in calinfo (per-scan beam volumes etc) if present
        calinfo["calinfo"] = True
        kwds = (
            "scan",
            "bunit",
            "azimuth",
            "elevatio",
            "beameff",
            "apereff",
            "antgain",
            "gainunc",
            "bmaj",
            "bmin",
            "bpa",
            "parang",
            "beamvol",
            "beamvunc",
        )  # for now just hardwired ones we want
        for kwd in kwds:
            calinfo[kwd] = hdul[hdu].header[kwd]
    except KeyError:
        print(
            "WARNING - calinfo information not found in fits file header - to track JytoK etc you may need to reprocess the fits files using mustangidl > revision 932"
        )

    dat = {}

    pixid = np.array(raw["PIXID"])
    dets = np.unique(pixid)
    ndet = len(dets)
    ndata = int(len(pixid) / len(dets))
    pixid = np.reshape(pixid, [ndet, ndata])[:, 0]
    dat["pixid"] = pixid

    # this bit of odd gymnastics is because a straightforward reshape doesn't seem to leave the data in
    # memory-contiguous order, which causes problems down the road
    # also, float32 is a bit on the edge for pointing, so cast to float64
    dx = np.array(raw["DX"])
    if not (branch is None):
        bb = branch * np.pi / 180.0
        dx[dx > bb] = dx[dx > bb] - 2 * np.pi
    dat["dx"] = np.zeros([ndet, ndata], dtype="float64")
    dat["dx"][:] = np.reshape(dx, [ndet, ndata])[:]

    dy = np.array(raw["DY"])
    dat["dy"] = np.zeros([ndet, ndata], dtype="float64")
    dat["dy"][:] = np.reshape(dy, [ndet, ndata])[:]

    if "ELEV" in raw.names:
        elev = np.array(raw["ELEV"]) * np.pi / 180
        dat["elev"] = np.zeros([ndet, ndata], dtype="float64")
        dat["elev"][:] = np.reshape(elev, [ndet, ndata])[:]

    tt = np.reshape(np.array(raw["TIME"]), [ndet, ndata])[0, :]
    dt = np.median(np.diff(tt))
    dat["dt"] = dt

    dat_calib = np.array(raw["FNU"])
    dat["dat_calib"] = np.zeros(
        [ndet, ndata], dtype="float64"
    )  # go to double because why not
    dat_calib = np.reshape(dat_calib, [ndet, ndata])
    dat["dat_calib"][:] = dat_calib[:]

    ufnu = np.array(raw["UFNU"])
    if np.sum(np.array(ufnu > 9e5)) > 0:
        dat["mask"] = np.reshape(ufnu < 9e5, dat["dat_calib"].shape)
        dat["mask_sum"] = np.sum(dat["mask"], axis=0)
    # print 'cut frac is now ',np.mean(dat_calib==0)
    # print 'cut frac is now ',np.mean(dat['dat_calib']==0),dat['dat_calib'][0,0]
    dat["fname"] = fname
    dat["calinfo"] = calinfo

    ff = 180 / np.pi
    xmin = dx.min() * ff
    xmax = dx.max() * ff
    ymin = dy.min() * ff
    ymax = dy.max() * ff
    print(
        "ndata and ndet are ",
        ndet,
        ndata,
        len(pixid),
        " on ",
        fname,
        "with lims ",
        xmin,
        xmax,
        ymin,
        ymax,
    )

    hdul.close()

    return dat


def read_octave_struct(fname: str):
    """
    Read TOD saved as octave struct.
    Presumably for compatibility with Ninkasi?

    Parameters
    ----------
    fname : str
        Path to the saved struct.

    Returns
    -------
    dat : dict
        Dict containing TOD info.
        Can be passed into the Tod class.
    """
    f = open(fname)
    nkey = np.fromfile(f, "int32", 1)[0]
    dat = {}
    for i in range(nkey):
        key = f.readline().strip()
        # print 'key is ' + key
        ndim = np.fromfile(f, "int32", 1)[0]
        dims = np.fromfile(f, "int32", ndim)
        dims = np.flipud(dims)
        # print 'Dimensions of ' + key + ' are ' + repr(dims)
        nbyte = np.fromfile(f, "int32", 1)[0]
        # print 'nbyte is ' + repr(nbyte)
        dtype = _get_type(nbyte)
        tmp = np.fromfile(f, dtype, dims.prod())
        dat[key] = np.reshape(tmp, dims)
    f.close()
    return dat


def todvec_from_files_octave(fnames: Iterable[str]) -> TodVec:
    """
    Load a TodVec from a octave structs.

    fnames : Iterable[str]
        Paths to the structs to load.

    todvec : TodVec
        TodVec with the loaded TODs.
    """
    todvec = TodVec()
    for fname in fnames:
        info = read_octave_struct(fname)
        tod = Tod(info)
        todvec.add_tod(tod)
    return todvec


def cut_blacklist(tod_names: Iterable[str], blacklist: Iterable[str]) -> list[str]:
    """
    Remove blacklisted TODs from a list of paths.

    Parameters
    ----------
    tod_names : Iterable[str]
        Paths to the TODs you intend to load.
    blacklist : Iterable[str]
        TODs that should be cut.
        Doesn't need to be a subset of tod_names.

    Returns
    -------
    uncut : list[str]
        TODs that weren't cut. Will be sorted.
    """
    # Dict mapping the TODs to their paths
    tods = {nm.split("/")[-1]: nm for nm in tod_names}
    ncut = 0
    for nm in blacklist:
        tt = nm.split("/")[-1]
        if tt in tods:
            ncut = ncut + 1
            del tods[tt]
    print("deleted ", ncut, " bad files.")
    uncut = list(tods.values())
    uncut.sort()
    return uncut
