#!/bin/bash


#SBATCH --job-name=plmblast
#SBATCH -p cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24GB
#SBATCH --output=/home/nfs/kkaminski/PLMBLST/%A.out
#SBATCH --error=/home/nfs/kkaminski/PLMBLST/%A.err

set -e
# limit threads for numba/numpy/torch
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1


DBFILE="/home/nfs/kkaminski/PLMBLST/ecod70db_20220902"
EMBEDDER="esm"
RESULTS="/home/nfs/kkaminski/PLMBLST/results_allvall.p"

python="/home/nfs/kkaminski/anaconda3/envs/plmblast3/bin/python"

$python allvsall.py $DBFILE $RESULTS -workers $SLURM_CPUS_PER_TASK -win 10