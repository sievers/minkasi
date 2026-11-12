#!/bin/bash
# Joint M2 + ACT f090 needlet map of A399-401 on Trillium (see reports/A399_401_HPC.md for the memory and time model).
#
# Usage: sbatch [overrides] examples/slurm/a399_401.sbatch MODE [RADIUS]
#   dryrun  env check (--debug, one TOD) then --dry-run on all 304 TODs: prints each band's lattice and the response
#           memory per rank. Needs 1 node for < 30 min:
#             sbatch -p debug -N 1 -n 38 --ntasks-per-node=38 -c 5 -t 01:00:00 examples/slurm/a399_401.sbatch dryrun
#   smoke   production layout, --cg 5. Measures the real peak memory (mem_*.log, sacct) and the time per CG iteration,
#           and fills the response cache that the full run reuses:
#             sbatch -t 04:00:00 examples/slurm/a399_401.sbatch smoke
#   full    600 CG iterations, saves at 40 100 200 400 600:
#             sbatch examples/slurm/a399_401.sbatch full
#   post    act_consistency.py and multiscale_map.py on the full run's map, one task, a few minutes:
#             sbatch -p debug -N 1 -n 1 -c 192 -t 00:30:00 examples/slurm/a399_401.sbatch post
# RADIUS (pixels, default 14) sets the preconditioner's probe window. Each radius has its own output and cache
# directory, because the cache file names do not record it.
#
# Memory: every rank holds all the M2 response matrices, ~81 GB peak per rank at radius 14 (~140 GB at 30), plus
# ~0.08 GB per TOD. 7 ranks per node is ~630 GB of the 767 GB. At radius 30 use at most 4 ranks per node, e.g.
#   sbatch -N 10 -n 38 --ntasks-per-node=4 -c 48 examples/slurm/a399_401.sbatch full 30
# The TODs are split evenly over however many ranks the job has.
#SBATCH --job-name=a399_401_needlet
#SBATCH --account=def-rbond
#SBATCH --nodes=6
#SBATCH --ntasks=38
#SBATCH --ntasks-per-node=7
#SBATCH --cpus-per-task=27
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jorlo/needlets/a399_401/logs/%x_%j.txt
#SBATCH --mail-type=FAIL,END

set -uo pipefail
MODE=${1:?MODE is dryrun, smoke or full}
RADIUS=${2:-14}

module load StdEnv/2023 python/3.12.4 gcc/14.3 openmpi/5.0.8 mpi4py/4.1.0
source /home/jorlo/venvs/needlet/bin/activate

REPO=/home/jorlo/dev/minkasi
IDIR=/scratch/jorlo/MUSTANG/M2-TODs/bridge
ACT=/scratch/jorlo/MUSTANG/M2-TODs/A399_401/bridge/all_f090_ivar.fits # the map is found by _ivar -> _map
OUT=/scratch/jorlo/needlets/a399_401/r$RADIUS
CACHE=$OUT/cache

# Home is read-only on compute nodes. numpy's FFTs (the bulk of the time) are single threaded; numba's
# map2tod/tod2map use the rank's cores.
export MPLBACKEND=Agg MPLCONFIGDIR=/scratch/jorlo/.cache/matplotlib NUMBA_CACHE_DIR=/scratch/jorlo/.cache/numba
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 # otherwise rank 0's last lines (the --dry-run table) can be lost at exit
MEMLOG=$OUT/$MODE/mem_$SLURM_JOB_ID # one directory per job, so the summary sees only this job's samples
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$CACHE" "$MEMLOG" /scratch/jorlo/needlets/a399_401/logs
cd "$OUT/$MODE" || exit 1

ntod=$(find "$IDIR" -maxdepth 1 -name 'Sig*.fits' | wc -l)
NTODS=$(((ntod + SLURM_NTASKS - 1) / SLURM_NTASKS))
echo "$MODE, radius $RADIUS: $ntod TODs on $SLURM_NTASKS ranks ($NTODS per rank), $SLURM_NNODES nodes," \
    "$SLURM_CPUS_PER_TASK cores per rank, git $(git -C $REPO rev-parse --short HEAD)"

RUN=(python "$REPO/examples/twores_ex.py" --name A399_401 --idir "$IDIR/" --act-glob "$ACT" --ntods "$NTODS"
    --direct --cg-tol 0 --lattice-tol 1e-3 --min-hits 0 --radius "$RADIUS" --cache "$CACHE")
SRUN=(srun --cpus-per-task="$SLURM_CPUS_PER_TASK" --distribution=block:cyclic --cpu-bind=cores)

srun --overlap --ntasks-per-node=1 -N "$SLURM_NNODES" --cpus-per-task=1 --cpu-bind=none \
    bash "$REPO/examples/slurm/mem_sampler.sh" "$MEMLOG" 20 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

case $MODE in
dryrun)
    srun -n 1 -N 1 python "$REPO/examples/twores_ex.py" --debug --idir "$IDIR/" --act-glob "$ACT" \
        --direct --cg 5 --cg-tol 0 --lattice-tol 1e-3 --min-hits 0 --radius "$RADIUS" \
        --cache "$OUT/dryrun" --outroot "$OUT/dryrun/debug" || { echo "--debug failed"; exit 1; }
    "${SRUN[@]}" "${RUN[@]}" --dry-run --outroot "$OUT/dryrun/A399_401"
    ;;
smoke) "${SRUN[@]}" "${RUN[@]}" --cg 5 --cg-save 5 --outroot "$OUT/smoke/A399_401" ;;
full) "${SRUN[@]}" "${RUN[@]}" --cg 600 --cg-save 40 100 200 400 600 --outroot "$OUT/full/A399_401" ;;
post) # the full run's map against ACT, and the display map (reports/A399_401_HPC.md section 5)
    F=$OUT/full/A399_401
    srun -n 1 python "$REPO/examples/act_consistency.py" --root "$F" --act-glob "$ACT" --size 70 \
        --label "A399-401, 304 TODs, r$RADIUS, 600 CG" | tee "$OUT/post/consistency.txt"
    srun -n 1 python "$REPO/examples/multiscale_map.py" --joint "${F}_final.fits" --hits "${F}_hits.fits" \
        --act-glob "$ACT" --size 70 --smooth 10 --smooth-ref-hits 3000 --show-hits 120 --feather-hits 300 \
        --outline --title "A399-401, M2 + ACT" --out "$OUT/post/A399_401_multiscale.png" \
        --fits "$OUT/post/A399_401_multiscale.fits"
    # zoom on the M2 footprint (hits >= 120 spans 21' x 31' around this centre)
    srun -n 1 python "$REPO/examples/multiscale_map.py" --joint "${F}_final.fits" --hits "${F}_hits.fits" \
        --act-glob "$ACT" --center 44.662 13.419 --size 34 --smooth 10 --smooth-ref-hits 3000 --show-hits 120 \
        --feather-hits 300 --outline --title "A399-401, M2 + ACT (M2 footprint)" \
        --out "$OUT/post/A399_401_multiscale_zoom.png"
    # Hincks et al. 2022 Figure 1 (right) with the joint map
    srun -n 1 python "$REPO/examples/a399_401_fig1.py" --root "$F" --act-glob "$ACT" \
        --out "$OUT/post/A399_401_fig1.png"
    ;;
*) echo "unknown MODE $MODE" && exit 1 ;;
esac
rc=$?

echo "exit code $rc. Peak memory per node (GB used, from mem_*.log):"
for f in "$MEMLOG"/mem_*.log; do
    echo "  $(basename "$f" .log): $(awk '{print $4}' "$f" | sort -n | tail -1) GB;" \
        "largest python RSS $(grep -o 'max [0-9.]* GB' "$f" | awk '{print $2}' | sort -n | tail -1) GB"
done
sacct -j "$SLURM_JOB_ID" -o JobID%20,MaxRSS,MaxRSSNode,Elapsed,State
exit $rc
