"""
Plot the CG relative residual |r|/|b| against the iteration from the logs of twores_ex.py --cg, one panel per
group of runs (each --group is "title:label=log,label=log,...").
"""

import argparse
import re

import matplotlib

matplotlib.use("Agg")  # no display needed
import numpy as np
from matplotlib import pyplot as plt

LINE = re.compile(r"CG iter (\d+): \|r\|/\|b\| = ([0-9.e+-]+)")


def read_curve(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the residual history from a log, taking the first line of each iteration (every MPI rank prints it).

    Parameters
    ----------
    path : str
        Log file of twores_ex.py.

    Returns
    -------
    it : np.ndarray
        Iteration numbers.
    res : np.ndarray
        Relative residual at each iteration.
    """
    seen = {}
    with open(path) as f:
        for line in f:
            m = LINE.search(line)
            if m:
                seen.setdefault(int(m[1]), float(m[2]))
    it = np.array(sorted(seen))
    return it, np.array([seen[i] for i in it])


def main(args: argparse.Namespace):
    """
    Make the figure.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    fig, axs = plt.subplots(
        1,
        len(args.group),
        figsize=(5.2 * len(args.group), 4),
        dpi=70 if args.debug else 130,
    )
    for ax, group in zip(np.atleast_1d(axs), args.group):
        title, runs = group.split(":", 1)
        for run in runs.split(","):
            label, path = run.split("=", 1)
            ax.plot(*read_curve(path), label=label)
        ax.set(title=title, xlabel="CG iteration", ylabel="|r| / |b|", yscale="log")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", nargs="+", required=True, help="title:label=log,label=log"
    )
    parser.add_argument("--out", required=True, help="Output figure")
    parser.add_argument("--debug", action="store_true", help="Lower resolution figure")
    main(parser.parse_args())
