"""
Plot one final map with aplpy, blanking pixels without hits. The colour scale comes from the
well-covered pixels (hits >= --scale-hits) so the noisy edge does not set the range.
"""

import argparse

import numpy as np
from matplotlib import pyplot as plt

from plot_final_aplpy import load, panel


def main(args: argparse.Namespace):
    """
    Make the figure.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    data, hdr = load(args.final)
    hits = load(args.hits)[0]
    shown = np.where(hits < args.min_hits, np.nan, data)
    lim = np.percentile(data[hits >= args.scale_hits], [args.pct, 100 - args.pct])
    fig = plt.figure(figsize=(7, 6), dpi=80 if args.debug else 150)
    panel(
        fig,
        (1, 1, 1),
        shown,
        hdr,
        args.label,
        tuple(lim),
        beam_arcmin=args.m2_beam / 60,
    )
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", required=True, help="Final map FITS")
    parser.add_argument("--hits", required=True, help="Hits map FITS")
    parser.add_argument("--out", required=True, help="Output figure")
    parser.add_argument("--label", default="final")
    parser.add_argument(
        "--min-hits", type=float, default=1, help="Blank pixels below this"
    )
    parser.add_argument(
        "--scale-hits", type=float, default=20, help="Pixels used for the colour scale"
    )
    parser.add_argument("--pct", type=float, default=1, help="Colour scale percentile")
    parser.add_argument("--m2-beam", type=float, default=9.0, help="M2 FWHM, arcsec")
    parser.add_argument("--debug", action="store_true", help="Lower resolution figure")
    main(parser.parse_args())
