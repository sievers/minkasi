"""
Make a simulated sky for testing twores_ex.py (--sim-sky): the cluster and point source of a WITCH model
(beam convolved, in the M2 data units, K_RJ) plus a lensed CMB realisation at the M2 beam, written as a
FITS image with a CAR WCS centred on the model. Extensions hold the components. This script uses WITCH;
twores_ex.py only reads the FITS file.
"""

import argparse
import os

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from minkasi.needlet.act import act_conversion
from minkasi.needlet.cmb import lensed_cmb_cl
from scipy.ndimage import gaussian_filter

FWHM2SIGMA = 1 / np.sqrt(8 * np.log(2))


def witch_components(config: str, ra: np.ndarray, dec: np.ndarray) -> dict:
    """
    Evaluate the M2 models of a WITCH config at sky positions, in the units of the M2 data.

    Parameters
    ----------
    config : str
        WITCH config with "cluster_model" and "ps_model" and a "mustang2" dataset.
    ra, dec : np.ndarray
        Positions in radians.

    Returns
    -------
    comps : dict
        "cluster" (beam convolved, times the dataset prefactor) and "ps" (as fitted, no beam or prefactor).
    """
    import jax.numpy as jnp
    import witch.utils as wu
    from witch.containers import Model
    from witch.containers.metamodel import _project
    from witch.dataset import BeamConvAndPrefac
    from witch.fitter import load_config

    cfg = load_config({}, config)
    dset = cfg["datasets"]["mustang2"]
    beam = wu.beam_double_gauss(
        eval(str(cfg["coords"]["dr"])),
        *(eval(str(dset["beam"][k])) for k in ("fwhm1", "amp1", "fwhm2", "amp2")),
    )
    pref = eval(str(dset["prefactor"]), {"wu": wu, "cfg": cfg, "float": float})
    x = jnp.asarray(ra * wu.rad_to_arcsec)
    y = jnp.asarray(dec * wu.rad_to_arcsec)
    out = {}
    for key, name, meta in (
        ("cluster", "cluster_model", BeamConvAndPrefac(beam, pref, exclude=())),
        ("ps", "ps_model", None),
    ):
        model = Model.from_cfg(cfg, name)
        ip = model.model if meta is None else meta.apply(model.model)
        out[key] = np.asarray(_project(ip, x, y, model.xyz))
    out["center"] = (
        np.rad2deg(eval(str(cfg["coords"]["x0"]), {"u": u})),
        np.rad2deg(eval(str(cfg["coords"]["y0"]), {"u": u})),
    )
    return out


def cmb_realisation(
    shape: tuple[int, int], pix_arcsec: float, beam_fwhm: float, seed: int, cache: str
) -> np.ndarray:
    """
    Flat sky Gaussian realisation of the lensed CMB, in uK_CMB, smoothed with a Gaussian beam.

    Parameters
    ----------
    shape : tuple[int, int]
        Map shape.
    pix_arcsec : float
        Pixel size in arcsec.
    beam_fwhm : float
        Beam FWHM in arcsec.
    seed : int
        Random seed.
    cache : str
        File caching the CAMB spectrum.

    Returns
    -------
    cmb : np.ndarray
        The realisation.
    """
    pix = np.deg2rad(pix_arcsec / 3600)
    cl = lensed_cmb_cl(cache=cache)
    ell = (
        2
        * np.pi
        * np.hypot(*np.meshgrid(*(np.fft.fftfreq(n) for n in shape), indexing="ij"))
        / pix
    )
    amp = np.sqrt(np.interp(ell, np.arange(cl.size), cl, right=0) / pix**2)
    white = np.random.default_rng(seed).normal(size=shape)
    cmb = np.fft.ifft2(np.fft.fft2(white) * amp).real
    return gaussian_filter(cmb, beam_fwhm * FWHM2SIGMA / pix_arcsec, mode="wrap")


def main(args: argparse.Namespace):
    """
    Make the sky and write it.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    n = round(args.size * 60 / args.pixsize)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    wcs.wcs.cdelt = [-args.pixsize / 3600, args.pixsize / 3600]
    wcs.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    # centre on the model: evaluate it once at the origin to read the centre from the config
    probe = witch_components(args.config, np.zeros(1), np.zeros(1))
    wcs.wcs.crval = list(probe["center"])
    yy, xx = np.mgrid[:n, :n]
    ra, dec = wcs.wcs_pix2world(xx, yy, 0)
    comps = witch_components(args.config, np.deg2rad(ra), np.deg2rad(dec))
    cmb = (
        args.cmb
        * act_conversion(args.freq, unit="K_RJ")
        * cmb_realisation((n, n), args.pixsize, args.m2_beam, args.seed, args.cmb_cache)
    )
    total = comps["cluster"] + comps["ps"] + cmb
    hdr = wcs.to_header()
    hdr["BUNIT"] = "K_RJ"
    hdus = [fits.PrimaryHDU(total, hdr)] + [
        fits.ImageHDU(m, hdr, name=k)
        for k, m in (("CLUSTER", comps["cluster"]), ("PS", comps["ps"]), ("CMB", cmb))
    ]
    fits.HDUList(hdus).writeto(args.out, overwrite=True)
    for k, m in (
        ("cluster", comps["cluster"]),
        ("ps", comps["ps"]),
        ("cmb", cmb),
        ("total", total),
    ):
        print(f"{k}: min {m.min():.3e} max {m.max():.3e} rms {m.std():.3e} K_RJ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="/home/jorlo/dev/minkasi_jax/configs/MOOJ1142_Joint/MOOJ1142_joint_M2_ACT_gNFW_rs_peak_subcluster.yaml",
        help="WITCH config with the cluster and point source models",
    )
    parser.add_argument("--out", required=True, help="Output FITS")
    parser.add_argument(
        "--size", type=float, default=40.0, help="Side of the map, arcmin"
    )
    parser.add_argument("--pixsize", type=float, default=3.0, help="Arcsec")
    parser.add_argument("--freq", type=float, default=90e9, help="Hz")
    parser.add_argument(
        "--m2-beam", type=float, default=9.735, help="FWHM for the CMB, arcsec"
    )
    parser.add_argument(
        "--cmb", type=float, default=1.0, help="Scale of the CMB (0 to leave it out)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cmb-cache",
        default="/mnt/welch/USERS/jorlo/needlets/all106/cache/cmb_cl.npy",
        help="CAMB spectrum cache",
    )
    parser.add_argument("--debug", action="store_true", help="Small map")
    args = parser.parse_args()
    if args.debug:
        args.size = 10.0
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main(args)
