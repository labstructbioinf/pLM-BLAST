#!/bin/bash
set -e
export MKL_DYNAMIC=FALSE
# remamber to activate appropriate python environment


INPUT="data/input/rossmannsdb.fas"
DBDIR="data/output/rossmannsdb"
RESULTS=allvsall.csv
# create database directory
mkdir -p data/output
# Calculate embeddings if they are not created yet
if [ ! -d $DBDIR ]; then
	echo "calculate Rossmanns embeddings"
	python ../embeddings.py start $INPUT $DBDIR -bs 0 --asdir
	cp $INPUT $DBDIR.fas
fi

NUM_WORKERS=6
# Return hits with scores >=0.3
ALIGNMENT_CUTOFF="0.3"
COSINE_CUTOFF=0
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
