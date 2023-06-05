#!/bin/bash

set -e
# limit threads for numba/numpy/torch
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

BASEDIR="/home/nfs/kkaminski/PLMBLST/ovo"
DBFILE="${BASEDIR}/ecod_ovo.p"
EMBFILE="${BASEDIR}/ecod_emb.npz"
EMBEDDER="pt"
RESULTS="/home/nfs/kkaminski/PLMBLST/ovo_res.p"

python="/home/users/kkaminski/anaconda3/envs/plmblast/bin/python"

if test ! -f $EMBFILE; then
    $python ../embeddings.py $DBFILE $EMBFILE -cname seq
fi
$python onvevsone.py $DBFILE $EMBFILE -workers 6 -win 10