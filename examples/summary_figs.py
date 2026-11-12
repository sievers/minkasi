"""
Figures for the convergence summary page: the simulation solves against the truth per scale and iteration (PNG and
a GIF of the large-scale maps), the slow edge modes from the Ritz analysis, and the effect of the unit fix.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from cg_convergence import BANDS
from sim_compare import bandpass, load_map

SIM = "/mnt/welch/USERS/jorlo/needlets/sim"
RITZ = "/mnt/welch/USERS/jorlo/needlets/ritz"
RUNS = {
    "coef (needlet CG)": "coef",
    "pixn (pixel CG, needlet precond.)": "pixn",
    "pixj (pixel CG, Jacobi)": "pixj",
    "prior (outside-footprint prior)": "prior",
    "m2only (no ACT)": "m2only",
}
ITERS = (40, 200, 300, 400, 600)


def load_runs(sim: str, runs: dict, iters: tuple) -> dict:
    """
    Load the saved CG maps of the simulation runs.

    Parameters
    ----------
    sim : str
        Directory of the simulation runs.
    runs : dict
        Label to run directory name.
    iters : tuple
        Iterations to look for.

    Returns
    -------
    dict
        Label to {iteration: map}.
    """
    out = {}
    for label, name in runs.items():
        maps = {}
        for k in iters:
            path = f"{sim}/{name}/MOOJ1142_cg{k}.npy"
            if os.path.exists(path):
                maps[k] = np.load(path)
        out[label] = maps
    return out


def footprint(sim: str, min_hits: float, edge: float):
    """
    Footprint, its interior and the bounding box.

    Parameters
    ----------
    sim : str
        Directory of the simulation runs.
    min_hits : float
        Hits threshold of the footprint.
    edge : float
        Width of the edge region in pixels.

    Returns
    -------
    good, inner : np.ndarray
        Footprint and its part more than `edge` pixels from the boundary.
    box : tuple[slice, slice]
        Bounding box of the footprint.
    """
    from scipy.ndimage import distance_transform_edt

    good = load_map(f"{sim}/coef/MOOJ1142_hits.fits") >= min_hits
    inner = good & (distance_transform_edt(good) > edge)
    ii, jj = np.where(good)
    return good, inner, (slice(ii.min(), ii.max() + 1), slice(jj.min(), jj.max() + 1))


def sim_bands(args, truth, runs, good, inner):
    """
    Plot the interior error relative to the truth in the two largest bands and the decrement, per iteration.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    truth : np.ndarray
        Input sky.
    runs : dict
        Output of load_runs.
    good, inner : np.ndarray
        Footprint and interior masks.
    """
    bands = BANDS[-2:]
    tb = {b: bandpass(truth, good, *b) for b in bands}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for label, maps in runs.items():
        it = sorted(maps)
        for a, b in zip(ax[:2], bands):
            err = [
                (bandpass(maps[k], good, *b) - tb[b])[inner].std() / tb[b][inner].std()
                for k in it
            ]
            a.plot(it, err, "o-", label=label)
        tr = [
            (bandpass(maps[k], good, *bands[-1]) * tb[bands[-1]])[inner].sum()
            / (tb[bands[-1]][inner] ** 2).sum()
            for k in it
        ]
        ax[2].plot(it, tr, "o-", label=label)
    for a, b in zip(ax[:2], bands):
        a.set_title(f"error / truth, interior, scales {b[0]}-{b[1]} px")
        a.set_xlabel("CG iteration")
        a.set_yscale("log")
    ax[2].axhline(1, color="k", lw=0.8)
    ax[2].set_title(f"transfer, interior, scales {bands[-1][0]}-{bands[-1][1]} px")
    ax[2].set_xlabel("CG iteration")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/sim_bands.png", dpi=args.dpi)


def sim_gif(args, truth, runs, good, box):
    """
    GIF of the large-scale part (Gaussian band-pass 16-64 px) of the truth and of several solves over the iterations.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    truth : np.ndarray
        Input sky.
    runs : dict
        Output of load_runs.
    good : np.ndarray
        Footprint.
    box : tuple[slice, slice]
        Bounding box to show.
    """
    show = ["coef (needlet CG)", "prior (outside-footprint prior)", "m2only (no ACT)"]
    band = (16, 64)

    def large(m):
        return np.where(good, bandpass(m, good, *band), np.nan)[box].T * 1e6

    tl = large(truth)
    lim = np.nanmax(np.abs(tl))
    frames = [
        k
        for k in (40, 200, 400, 600)
        if all(k in runs[s] or s.startswith("m2only") for s in show)
    ]
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.6))
    ims = []
    ax[0, 0].imshow(tl, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[0, 0].set_title("truth (16-64 px band)")
    ax[1, 0].axis("off")
    ax[1, 0].text(0.05, 0.5, "", transform=ax[1, 0].transAxes, fontsize=13)
    for j, s in enumerate(show):
        ims.append(
            ax[0, j + 1].imshow(
                tl * 0, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim
            )
        )
        ims.append(
            ax[1, j + 1].imshow(
                tl * 0, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim
            )
        )
        ax[0, j + 1].set_title(s.split(" (")[0] + " map")
        ax[1, j + 1].set_title(s.split(" (")[0] + " minus truth")
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(ims[0], ax=ax, shrink=0.6, label="uK_RJ")
    label = ax[1, 0].texts[0]

    def update(k):
        for j, s in enumerate(show):
            m = runs[s].get(k, runs[s][max(runs[s])])
            lm = large(m)
            ims[2 * j].set_data(lm)
            ims[2 * j + 1].set_data(lm - tl)
        label.set_text(f"CG iteration {k}\n\nm2only: last\nsaved (300)")
        return ims

    anim = FuncAnimation(fig, update, frames=frames)
    anim.save(
        f"{args.outdir}/sim_drift.gif", writer=PillowWriter(fps=1), dpi=args.dpi // 2
    )
    update(frames[-1])
    fig.savefig(f"{args.outdir}/sim_drift_last.png", dpi=args.dpi)


def ritz_fig(args):
    """
    Ritz values of the preconditioned operator without and with ACT (small problem) and the slowest modes with ACT.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    s0, s1 = (np.load(f"{args.ritz}/{n}/MOOJ1142_ritz.npz") for n in ("s0", "s1"))
    good = load_map(f"{args.ritz}/s1/MOOJ1142_hits.fits") >= 20
    fig = plt.figure(figsize=(14, 4))
    a = fig.add_subplot(1, 5, 1)
    a.semilogy(np.sort(s0["lam"]), ".", ms=3, label="M2 only")
    a.semilogy(np.sort(s1["lam"]), ".", ms=3, label="M2 + ACT")
    a.set_ylim(1e-6, 1e6)
    a.set_xlabel("index")
    a.set_title("Ritz values of the\npreconditioned operator")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    for j in range(4):
        v = s1["vecs"][j]
        a = fig.add_subplot(1, 5, j + 2)
        lim = np.abs(v).max()
        a.imshow(v.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        a.contour(good.T.astype(float), [0.5], colors="k", linewidths=1)
        a.set_title(f"slow mode {j + 1}, $\\lambda$ = {np.sort(s1['lam'])[j]:.1e}")
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/ritz_modes.png", dpi=args.dpi)


def units_fig(args):
    """
    Large-scale map change between saved iterations, 40 TODs, before and after the unit fix (numbers from the
    cg_convergence.py runs on lat40/nyq600 and lat40/krj600).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    steps = ["40 to 200", "200 to 400", "400 to 600"]
    old, new = [0.41, 0.40, 0.16], [0.23, 0.13, 0.09]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(3)
    ax[0].bar(x - 0.2, old, 0.4, label="ACT in Jy/beam (old)")
    ax[0].bar(x + 0.2, new, 0.4, label="ACT in K_RJ (fixed)")
    ax[0].set_xticks(x, steps)
    ax[0].set_ylabel("relative change, 32-64 px")
    ax[0].set_title("Large scales settle faster")
    ax[0].legend(fontsize=8)
    ax[1].bar([0, 1], [4.37e-7, 8.15e-7], color=["C0", "C1"])
    ax[1].set_xticks([0, 1], ["old factor\n(Jy/beam per uK_CMB)", "K_RJ per uK_CMB"])
    ax[1].set_title("ACT unit conversion at 90 GHz")
    for a in ax:
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/units_fix.png", dpi=args.dpi)


def main(args: argparse.Namespace):
    """
    Make the figures.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    os.makedirs(args.outdir, exist_ok=True)
    units_fig(args)
    ritz_fig(args)
    good, inner, box = footprint(args.sim, 20, 20)
    truth = load_map(f"{args.sim}/coef/MOOJ1142_truth.fits")
    runs = load_runs(args.sim, RUNS, (40, 600) if args.debug else ITERS)
    sim_bands(args, truth, runs, good, inner)
    sim_gif(args, truth, runs, good, box)
    print("wrote figures to", args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", default=SIM, help="Directory of the simulation runs")
    parser.add_argument("--ritz", default=RITZ, help="Directory of the Ritz runs")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument(
        "--debug", action="store_true", help="Two iterations only, low resolution"
    )
    args = parser.parse_args()
    if args.debug:
        args.dpi = 60
    main(args)
