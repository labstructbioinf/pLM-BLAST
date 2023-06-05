#!/bin/bash



DBFILE="/home/nfs/kkaminski/PLMBLST/ecod30db_20220902"
EMBEDDER="esm"
RESULTS="/home/users/kkaminski/apps/pLM-BLAST/scripts/output/allvsall.res.p"

python="/home/nfs/kkaminski/anaconda3/envs/plmblast3/bin/python"

python allvsall.py $DBFILE $RESULTS -workers 6 -win 10