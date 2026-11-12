#!/bin/bash
# Hincks et al. 2022 Figure 1 (right) with the joint A399-401 map, on a debug node (a minute):
#   sbatch analysis/slurm/a399_401_fig1.sbatch [extra a399_401_fig1.py options]
#SBATCH --job-name=a399_401_fig1
#SBATCH --account=def-rbond
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/jorlo/needlets/a399_401/logs/%x_%j.txt

module load StdEnv/2023 python/3.12.4 gcc/14.3 openmpi/5.0.8 mpi4py/4.1.0
source /home/jorlo/venvs/needlet/bin/activate
export MPLBACKEND=Agg MPLCONFIGDIR=/scratch/jorlo/.cache/matplotlib PYTHONUNBUFFERED=1
python /home/jorlo/dev/minkasi/analysis/a399_401_fig1.py \
    --out /scratch/jorlo/needlets/a399_401/r14/post/A399_401_fig1.png "$@"
