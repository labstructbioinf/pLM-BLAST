#!/bin/bash

#SBATCH --job-name=esm_emb
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --output=/home/nfs/kkaminski/PLMBLST/%A.out
#SBATCH --error=/home/nfs/kkaminski/PLMBLST/%A.err


DBFILE="/home/nfs/kkaminski/PLMBLST/ecod70db_20220902.csv"
EMBEDDER="esm_msa1b_t12_100M_UR50S"
DBDESTINATION="/home/nfs/kkaminski/PLMBLST/ecod70db_esm"

python="/home/nfs/kkaminski/anaconda3/envs/plmblast3/bin/python"

srun $python ../embeddings.py -embedder $EMBEDDER -cname "sequence" -bs 32 --asdir $DBFILE $DBDESTINATION