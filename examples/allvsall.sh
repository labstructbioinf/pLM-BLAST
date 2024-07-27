#!/bin/bash
set -e
# remamber to activate appropriate python environment


INPUT="data/input/rossmannsdb"
DBDIR="data/output/rossmannsdb"
RESULTS=allvsall.csv
# create database directory
mkdir -p data/output
# Calculate embeddings if they are not created yet
if [ ! -d $DBDIR ]; then
	echo "calculate Rossmanns embeddings"
	python ../embeddings.py start $INPUT.fas $DBDIR -bs 0 --asdir
	cp $INPUT.fas $DBDIR.fas
fi

NUM_WORKERS=6
# Return hits with scores >=0.3
ALIGNMENT_CUTOFF="0.3"
# we are comparing only 30 sequences vs 30 sequence there is no need for additional prescreening
COSINE_CUTOFF=0
# Run plm-blast
python ../scripts/plmblast.py \
	$DBDIR \
	$DBDIR \
	$RESULTS \
	-alignment_cutoff $ALIGNMENT_CUTOFF \
	-workers $NUM_WORKERS \
	-cpc $COSINE_CUTOFF
