#!/bin/bash

#SBATCH --job-name=conda_install
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --output=/home/nfs/kkaminski/PLMBLST/%A.out
#SBATCH --error=/home/nfs/kkaminski/PLMBLST/%A.err
#SBATCH --chdir=/home/nfs/kkaminski


CONDA_PREFFIX="/home/nfs/kkaminski/anaconda3/etc/profile.d/conda.sh"

. $CONDA_PREFFIX

which python
conda env create -f /home/nfs/kkaminski/PLMBLST/environment.yml
