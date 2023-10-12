#!/bin/bash

set -e
# limit threads for numba/numpy/torch
#export MKL_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1
#export OMP_NUM_THREADS=1

#case='cupredoxin'
case='rossmanns'
# data paths
INDIR="./input"
OUTDIR="./output"

QUERY_INDEX="$OUTDIR/${case}.csv"
OUTFILE="$OUTDIR/${case}.hits.csv"

# Replace with a path to the database
DB_PATH="/home/nfs/kkaminski/PLMBLST/ecod30db_20220902"

ALIGNMENT_CUTOFF="0.3"
COSINE_CUTOFF=90
SIGMA=2

NUM_WORKERS=8

mkdir -p $OUTDIR

# Calculate query embedding
if [ ! -f $OUTDIR/$case.pt ]; then
	echo "calculate query embedding"
	python ../embeddings.py start $INDIR/$case.fas $OUTDIR/$case.pt
	cp $INDIR/$case.fas $OUTDIR/$case.fas
fi

# Run plm-blast
if [ ! -f $OUTFILE ]; then
	python plmblast.py \
		$DB_PATH \
		$OUTDIR/$case \
		$OUTFILE \
		-cosine_percentile_cutoff $COSINE_CUTOFF \
		-alignment_cutoff $ALIGNMENT_CUTOFF \
		-workers $NUM_WORKERS \
        -sigma_factor $SIGMA \
		-use_chunks
fi

# plot hits
python plot.py $OUTFILE $QUERY_INDEX $OUTDIR/$case.hits_score_ecod.png -mode score -ecod
python plot.py $OUTFILE $QUERY_INDEX $OUTDIR/$case.hits_qend_ecod.png -mode qend -ecod

