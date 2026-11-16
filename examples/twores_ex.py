"""
Two resolution (MUSTANG-2 TODs + ACT maps) needlet mapmaking.

Reads the TODs, runs a short PCG in the needlet basis to clean the noise estimate, computes the (sparse) needlet
space response matrices of the TODs and of ACT, and solves for the combined map by CG on the full operator with
the block solve as preconditioner (--cg N), or by the block solve alone. The pipeline is
minkasi.needlet.joint.JointMapper; see --help for the options.

Baseline (106 TODs, 16 ranks):
  mpirun -n 16 python twores_ex.py --ntods 7 --direct --cg 600 --cg-tol 0 --lattice-tol 1e-3 --min-hits 0 \
      --cache <dir> --outroot <dir>/MOOJ1142
Outputs: <outroot>_final.fits (the map), _hits.fits, _act.fits (ACT as it enters the solve), _cg<N>.npy.
"""

from minkasi.needlet.joint import JointMapper, parse_args

if __name__ == "__main__":
    JointMapper(parse_args(__doc__)).run()
