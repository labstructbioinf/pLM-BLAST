#!/bin/bash
set -e
# remamber to activate appropriate python environment


INPUT="data/input/rossmannsdb.fas"
DBDIR=/home/nfs/kkaminski/PLMBLST/test_data/rossmanns
DB=/home/nfs/kkaminski/PLMBLST/ecod30db_20220902
RESULTS=allvsall.csv
# create database directory

NUM_WORKERS=6
# Return hits with scores >=0.3
ALIGNMENT_CUTOFF="0.3"
COSINE_CUTOFF=98
SIGMA=2
# no cutoff because our database is small we dont need additional filtering
# Run plm-blast
python scripts/plmblast.py \
	$DB \
	$DBDIR \
	$RESULTS \
	-cosine_percentile_cutoff $COSINE_CUTOFF \
	-alignment_cutoff $ALIGNMENT_CUTOFF \
	-workers $NUM_WORKERS \
		-sigma_factor $SIGMA \
	--use_chunks \
	--enh
