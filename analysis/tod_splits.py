"""
Jackknife splits of the MUSTANG-2 TODs for twores_ex.py --tod-list: the same TODs twores_ex would use (sorted, blacklist
removed, the first --ntods-total), split in two halves by time (file order, which follows the observing sessions) and by
mean elevation. Writes one text file of TOD paths per half and prints the split.
"""

import argparse
import glob
import os

import minkasi.minkasi_all as minkasi
import numpy as np


def mean_elevation(path: str) -> float:
    """
    Mean elevation of a TOD over all detectors and samples.

    Parameters
    ----------
    path : str
        TOD FITS file.

    Returns
    -------
    float
        Mean elevation in the units of the file.
    """
    return float(np.mean(minkasi.read_tod_from_fits(path)["elev"]))


def main(args: argparse.Namespace):
    """
    Write the split lists.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    names = sorted(glob.glob(os.path.join(args.idir, "Sig*.fits")))
    names = minkasi.cut_blacklist(names, minkasi.get_bad_tods(args.name)[0])[
        : args.ntods_total
    ]
    half = len(names) // 2
    elev = np.array([mean_elevation(f) for f in names])
    order = np.argsort(elev)
    splits = {
        "time1": names[:half],
        "time2": names[half:],
        "elev_lo": [names[i] for i in sorted(order[:half])],
        "elev_hi": [names[i] for i in sorted(order[half:])],
    }
    os.makedirs(args.outdir, exist_ok=True)
    for key, files in splits.items():
        with open(os.path.join(args.outdir, f"{key}.txt"), "w") as f:
            f.write("\n".join(files) + "\n")
        idx = [names.index(x) for x in files]
        sessions = sorted({x.split("TOD-")[1].rsplit("-", 1)[0] for x in files})
        print(
            f"{key:8s} {len(files):3d} TODs, mean elevation {elev[idx].mean():6.2f} "
            f"(range {elev[idx].min():.2f}-{elev[idx].max():.2f}), sessions {', '.join(sessions)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--idir", default="/mnt/welch/MUSTANG/M2-TODs/MOOJ1142/mustang2/"
    )
    parser.add_argument("--name", default="MOOJ1142")
    parser.add_argument("--ntods-total", type=int, default=40)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--debug", action="store_true", help="Only 4 TODs")
    args = parser.parse_args()
    if args.debug:
        args.ntods_total = 4
    main(args)
