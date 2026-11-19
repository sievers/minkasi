"""
Check JointMapper.remap on real data. Remapping the data run() used must reproduce run()'s map (to ~1e-12), remap must be
homogeneous (2 x data -> 2 x map, with --direct), and the size of its non-additivity remap(a + b) - remap(a) - remap(b)
is reported (b = the data reversed in time). A finite PCG/CG is not additive; this measures by how much at the chosen
iteration counts.

Run as `python check_remap.py --debug` (1 TOD, small map) or under mpirun with production options, e.g.
`mpirun --bind-to none -n 8 python check_remap.py --cg 600 --lattice-tol 1e-3 --ntods-total 40 --idir ... --act-glob ...`.
Exits non-zero if remap(same data) does not reproduce run().
"""

import argparse
import glob
import os
import sys

import numpy as np
from pixell import enmap, utils

from minkasi.needlet.joint import JointMapper, options
from minkasi.parallel import myrank
from minkasi.tods.io import read_tods


def act_pairs(act_glob: str, lims: list[float], nmax: int | None) -> list[tuple]:
    """
    Read ACT (map, ivar) split pairs in a box around the field.

    Parameters
    ----------
    act_glob : str
        Glob of the ivar maps; the maps are found by replacing _ivar with _map.
    lims : list[float]
        Field limits in radians (ra0, ra1, dec0, dec1).
    nmax : int | None
        Splits to read, None for all.

    Returns
    -------
    list[tuple]
        (map, ivar) enmaps.
    """
    pad = 0.05 * utils.degree
    box = np.array([[lims[2] - pad, lims[0] - pad], [lims[3] + pad, lims[1] + pad]])
    return [
        (
            enmap.read_map(p.replace("_ivar", "_map"), box=box),
            enmap.read_map(p, box=box),
        )
        for p in sorted(glob.glob(act_glob))[:nmax]
    ]


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """
    Max relative difference of two maps.

    Parameters
    ----------
    a, b : np.ndarray
        Maps.

    Returns
    -------
    float
        max|a - b| / max|b|.
    """
    return float(np.abs(a - b).max() / np.abs(b).max())


def main() -> None:
    """Run the checks."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help='JointMapper --debug problem (1 TOD, 6 bands, 6" pixels)',
    )
    p.add_argument(
        "--idir", default=None, help="M2 TOD directory (default: JointMapper's)"
    )
    p.add_argument(
        "--act-glob", default=None, help="ACT ivar glob (default: JointMapper's)"
    )
    p.add_argument("--name", default="MOOJ1142")
    p.add_argument("--ntods", type=int, default=None, help="TODs per rank")
    p.add_argument("--ntods-total", type=int, default=None)
    p.add_argument(
        "--cg",
        type=int,
        default=5,
        help="CG iterations (0: block solve without --direct)",
    )
    p.add_argument("--cg-tol", type=float, default=1e-3)
    p.add_argument("--lattice-tol", type=float, default=None)
    p.add_argument(
        "--outdir", default="check_remap_out", help="Outputs and response caches"
    )
    args = p.parse_args()
    over = {
        "debug": args.debug,
        "name": args.name,
        "cg": args.cg,
        "cg_tol": args.cg_tol,
        "lattice_tol": args.lattice_tol,
        "cache": args.outdir,
        "outroot": os.path.join(args.outdir, "data"),
    }
    over |= {
        k: v
        for k, v in [
            ("idir", args.idir),
            ("act_glob", args.act_glob),
            ("ntods", args.ntods),
        ]
        if v
    }
    over |= {"ntods_total": args.ntods_total}
    os.makedirs(args.outdir, exist_ok=True)
    opts = options(over)
    todvec = read_tods(opts.idir, opts.ntods, opts.name, opts.ntods_total)
    dat = [tod.info["dat_calib"].copy() for tod in todvec.tods]
    jm = JointMapper(opts, todvec=todvec)
    m_run = jm.run().copy()
    pairs = act_pairs(opts.act_glob, jm.wmap.lims, 2 if opts.debug else None)
    m_same = jm.remap(dat, pairs, os.path.join(args.outdir, "same")).copy()
    rev = [d[:, ::-1].copy() for d in dat]
    rev_pairs = [(m[::-1, ::-1].copy(), iv) for m, iv in pairs]
    m_b = jm.remap(rev, rev_pairs, os.path.join(args.outdir, "b")).copy()
    sums = [(ma + mb, iv) for (ma, iv), (mb, _) in zip(pairs, rev_pairs)]
    m_ab = jm.remap(
        [a + b for a, b in zip(dat, rev)], sums, os.path.join(args.outdir, "ab")
    )
    same = rel(m_same, m_run)
    lines = [
        f"cg={opts.cg} direct={opts.direct} maxiter={opts.maxiter}: max|run| {np.abs(m_run).max():.3e}",
        f"  remap(same data) vs run: max rel diff {same:.2e}",
        f"  remap(a + b) vs remap(a) + remap(b): max rel diff {rel(m_ab, m_same + m_b):.2e}",
    ]
    if opts.direct:
        m_dbl = jm.remap(
            [2 * d for d in dat],
            [(2 * m, iv) for m, iv in pairs],
            os.path.join(args.outdir, "double"),
        )
        lines.append(
            f"  remap(2 x data) vs 2 x run: max rel diff {rel(m_dbl, 2 * m_run):.2e}"
        )
    if myrank == 0:
        print("\n".join(lines), flush=True)
    sys.exit(0 if same < 1e-8 else 1)


if __name__ == "__main__":
    main()
