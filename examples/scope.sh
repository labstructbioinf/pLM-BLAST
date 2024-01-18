#!/bin/bash

ALIGNMENT_CUTOFF="0.3"
COSINE_CUTOFF=70

DBDIR=/home/nfs/kkaminski/PLMBLST/db/scop40-201
RESULTS="/home/nfs/kkaminski/PLMBLST/results/scope40-201_${ALIGNMENT_CUTOFF}_${COSINE_CUTOFF}.csv"

NUM_WORKERS=6
# Return hits with scores >=0.3
# no cutoff because our database is small we dont need additional filtering
# Run plm-blast
python ../scripts/plmblast.py \
	$DBDIR \
	$DBDIR \
	$RESULTS \
	-cosine_percentile_cutoff $COSINE_CUTOFF \
	-alignment_cutoff $ALIGNMENT_CUTOFF \
	-workers $NUM_WORKERS \
	--use_chunks
