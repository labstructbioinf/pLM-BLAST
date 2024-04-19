#!/bin/bash
set -e

# Query is a cupredoxin sequence (see the manuscript for the details)
case='histone'

# data paths
INDIR="./data/input"
OUTDIR="./data/output"
OUTFILE="$OUTDIR/${case}.hits.csv"

# Replace with a path to the database
# Pre-calculated ECOD databased can be obtained from http://ftp.tuebingen.mpg.de/pub/protevo/toolkit/databases/plmblast_dbs
DB_PATH="/home/users/sdunin/db/plmblast/data/ecod30db_20231201"

# Return hits with scores >=0.3 (max score is 1)
ALIGNMENT_CUTOFF="0.3"
COSINE_CUTOFF=70

# Customize according to your system specifications
NUM_WORKERS=10

mkdir -p $OUTDIR

# Calculate query embedding
if [ ! -f $OUTDIR/$case.pt ]; then
	echo "calculate query embedding"
	python ../embeddings.py start $INDIR/$case.fas $OUTDIR/$case.pt -bs 0
	cp $INDIR/$case.fas $OUTDIR/$case.fas
fi

# Run plm-blast
python ../scripts/plmblast.py \
	$DB_PATH \
	$OUTDIR/$case \
	$OUTFILE \
	-cosine_percentile_cutoff $COSINE_CUTOFF \
	-alignment_cutoff $ALIGNMENT_CUTOFF \
	-workers $NUM_WORKERS \

# Plotting works for single queries only
python ../scripts/plot.py $OUTFILE $OUTDIR/$case.fas $OUTDIR/$case.hits_score_ecod.png -mode score -ecod
python ../scripts/plot.py $OUTFILE $OUTDIR/$case.fas $OUTDIR/$case.hits_qend_ecod.png -mode qend -ecod

