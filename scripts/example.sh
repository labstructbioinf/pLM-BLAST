#!/bin/bash

set -e
# limit threads for numba/numpy/torch
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Databases consist of two files, an index file (.csv) containing sequences and their descriptions, and an embedding file (.pt_emb.p)
# containing PT5 (or other model) embeddings of sequences listed in the index.

# To create a database, first use makeindex.py to create an index from a FASTA file,
# and then use `embeddings.py` to compute embeddings based on this index file.

# The query sequence must be represented as a one-element database.

# See the example below for the complete pipeline from the FASTA sequence to searching the ECOD database.

case='cupredoxin'

# data paths
INDIR="./input"
OUTDIR="./output"

QUERY_INDEX="$OUTDIR/${case}.csv"
OUTFILE="$OUTDIR/${case}.hits.csv"
# Replace with a path to the database
DB_PATH="/home/nfs/kkaminski/PLMBLST/ecod30db_20220902"

ALIGNMENT_CUTOFF="0.35"
COSINE_CUTOFF=95
SIGMA=2

NUM_WORKERS=10

mkdir -p $OUTDIR

# calculate index
python makeindex.py ./input/$case.fas ./output/$case.csv

# calculate embeddings
if [ ! -f $OUTDIR/$case.pt_emb.p ]; then
	echo "calculate query embedding"
	python ../embeddings.py $INDIR/$case.fas $OUTDIR/$case.pt_emb.p
fi

# run plm-blast
if [ ! -f $OUTFILE ]; then
	# search pre-calculated ECOD database
	python run_plm_blast.py \
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

