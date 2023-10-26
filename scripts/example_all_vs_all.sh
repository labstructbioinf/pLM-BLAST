#!/bin/bash
set -e
export MKL_DYNAMIC=FALSE


OUTFILE="allvsall.hits.csv"
# Replace with a path to the database
DB_PATH="/home/nfs/kkaminski/PLMBLST/ecod30db_20220902"

NUM_WORKERS=6
# Return hits with scores >=0.3
ALIGNMENT_CUTOFF="0.3"
COSINE_CUTOFF=90
SIGMA=2

# Run plm-blast
if [ ! -f $OUTFILE ]; then
	python plmblast.py \
		$DB_PATH \
		$DB_PATH \
		$OUTFILE \
		-cosine_percentile_cutoff $COSINE_CUTOFF \
		-alignment_cutoff $ALIGNMENT_CUTOFF \
		-workers $NUM_WORKERS \
        	-sigma_factor $SIGMA \
		--use_chunks
fi

# Plotting works for single queries only
#python plot.py $OUTFILE $OUTDIR/$case.fas $OUTDIR/$case.hits_score_ecod.png -mode score -ecod
#python plot.py $OUTFILE $OUTDIR/$case.fas $OUTDIR/$case.hits_qend_ecod.png -mode qend -ecod

