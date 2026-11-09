"""
GIF of how probe_band recovers the rows of a band's response matrix G = K^T O K, with O the normal
matrix A^T N^-1 A (or the ACT analogue). Impulses further apart than the kept window are summed
into one map and passed through O together, so one application of O gives one row of G for every
impulse in the group. The groups are the ones color_groups gives probe_band.
"""

import argparse
import io
import itertools

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display needed
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from minkasi.needlet.needlet import color_groups
from PIL import Image


def draw_pass(n: int, rx: int, passes: list, k: int, dpi: int) -> Image.Image:
    """
    Draw one frame: the impulses of pass k and the rows recovered up to and including it.

    Parameters
    ----------
    n : int
        Lattice points per side (periodic).
    rx : int
        Half width of the kept window in lattice points.
    passes : list
        One (x indices, y indices) of impulses per pass, as made by itertools.product of color groups.
    k : int
        Index of the pass to draw.
    dpi : int
        Figure resolution.

    Returns
    -------
    frame : PIL.Image.Image
        The frame as a palette image.
    """
    done = np.zeros((n, n))
    for j in range(k + 1):
        done[np.ix_(*passes[j])] = 1 + (j == k)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.8), dpi=dpi)
    for ax in axs:
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    xs, ys = np.meshgrid(*passes[k], indexing="ij")
    ax = axs[0]
    ax.scatter(*np.meshgrid(range(n), range(n)), s=4, c="0.75")
    for a, b in zip(xs.ravel(), ys.ravel()):
        for dx, dy in itertools.product((-n, 0, n), repeat=2):  # periodic wrap
            ax.add_patch(
                Rectangle(
                    (a + dx - rx - 0.5, b + dy - rx - 0.5),
                    2 * rx + 1,
                    2 * rx + 1,
                    fc="tab:blue",
                    alpha=0.18,
                    ec="tab:blue",
                )
            )
    ax.scatter(xs.ravel(), ys.ravel(), s=22, c="tab:red", zorder=3)
    ax.set_title(
        f"pass {k + 1} of {len(passes)}: {xs.size} impulses summed,\n"
        "one application of the operator",
        fontsize=10,
    )
    axs[1].imshow(
        done.T,
        origin="lower",
        cmap="Blues",
        vmin=0,
        vmax=2.4,
        interpolation="nearest",
    )
    axs[1].set_title(
        f"rows of G recovered: {int((done > 0).sum())} of {n * n}\n"
        "(dark: this pass, light: earlier passes)",
        fontsize=10,
    )
    fig.suptitle(
        f"probe_band: lattice {n}x{n}, window {2 * rx + 1}x{2 * rx + 1} lattice points,"
        f" {len(passes)} passes instead of {n * n}",
        fontsize=11,
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def main(args: argparse.Namespace):
    """
    Make the GIF.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    n, rx = (10, 1) if args.debug else (args.n, args.rx)
    passes = list(
        itertools.product(color_groups(n, 2 * rx + 1), color_groups(n, 2 * rx + 1))
    )
    frames = [
        draw_pass(n, rx, passes, k, 60 if args.debug else 90)
        for k in range(len(passes))
    ]
    frames += frames[-1:] * 3  # hold the last frame
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output GIF")
    parser.add_argument("--n", type=int, default=20, help="Lattice points per side")
    parser.add_argument(
        "--rx", type=int, default=2, help="Half width of the window, lattice points"
    )
    parser.add_argument("--duration", type=int, default=450, help="ms per frame")
    parser.add_argument(
        "--debug", action="store_true", help="Small lattice, low resolution"
    )
    main(parser.parse_args())
