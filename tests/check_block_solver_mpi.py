"""
Check that WavSkyMap.block_solver shared over MPI ranks gives the serial result.
Run as `mpirun -n N python check_block_solver_mpi.py`; exits non-zero on a mismatch.
"""

import argparse
import tempfile

import numpy as np
from scipy import sparse

from minkasi import parallel
from minkasi.maps.skymap import SkyMap
from minkasi.needlet import needlet
from minkasi.needlet.needlet import Needlet, WavSkyMap

PIXSIZE = 30.0 / 3600 * np.pi / 180


def spd(n: int, rng: np.random.Generator) -> sparse.csr_matrix:
    """Random sparse symmetric positive definite matrix of size n."""
    a = sparse.random(n, n, density=0.05, random_state=int(rng.integers(1 << 30)))
    return (a @ a.T + n * sparse.identity(n)).tocsr()


def main(args: argparse.Namespace):
    """
    Compare the shared and the serial solve for a fixed and a converged inner CG.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    lims = [0.0, args.size, 0.0, args.size]
    box = SkyMap(lims, PIXSIZE, square=True, multiple=2).map
    need = Needlet(
        np.arange(4),
        lightcone=box,
        L=box.shape[0] * 30.0 / 60 * np.sqrt(2),
        pixsize=30.0,
    )
    need.get_needlet_filters_2d(
        need.lightcone_box.get_grid_dimless_2d(return_grid=True)
    )
    wmap = WavSkyMap(need, lims, PIXSIZE, square=True, multiple=2)
    wmap.set_downsamps(np.array([2, 2, 1, 1]))
    needlet.MAX_DIRECT = args.max_direct  # make the larger bands use the split CG
    sizes = [len(x) * len(y) for x, y in zip(wmap.nx_space, wmap.ny_space)]
    rng = np.random.default_rng(0)  # the same on every rank
    resp = np.array([spd(n, rng) for n in sizes], dtype=object)
    rhs = rng.normal(size=(need.nfilt, wmap.nx, wmap.ny))
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for inner in (None, 5):
            serial = wmap.block_solver([resp], inner=inner, distribute=False)(rhs)
            shared = wmap.block_solver(
                [resp], inner=inner, distribute=True, lmin_cache=f"{tmp}/lmin.json"
            )(rhs)
            err = np.abs(shared - serial).max() / np.abs(serial).max()
            ok &= err < 1e-6
            if parallel.myrank == 0:
                print(
                    f"nproc {parallel.nproc}, lattice sizes {sizes}, inner {inner}: max rel diff {err:.2e}"
                )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=float, default=0.005, help="Map size, radians")
    parser.add_argument("--max-direct", type=int, default=600)
    parser.add_argument("--debug", action="store_true", help="Smaller map")
    args = parser.parse_args()
    if args.debug:
        args.size, args.max_direct = 0.003, 300
    main(args)
