#!/bin/bash


DBDIR=/home/nfs/kkaminski/PLMBLST/scope40
RESULTS=data/output/scope40_results.csv

NUM_WORKERS=10
# Return hits with scores >=0.3
ALIGNMENT_CUTOFF="0.3"
COSINE_CUTOFF=95
SIGMA=2
# no cutoff because our database is small we dont need additional filtering
# Run plm-blast
python ../scripts/plmblast.py \
	$DBDIR \
	$DBDIR \
	$RESULTS \
	-cosine_percentile_cutoff $COSINE_CUTOFF \
	-alignment_cutoff $ALIGNMENT_CUTOFF \
	-workers $NUM_WORKERS \
		-sigma_factor $SIGMA \
	--use_chunks
