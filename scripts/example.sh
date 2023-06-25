#!/bin/bash

set -e
# limit threads for numba/numpy/torch
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Databases comprise two files, an index file (.csv) containing sequences and their descriptions,
# and an embeddings file (.pt_emb.p) containing PT5 embeddings of sequences listed in the index.

# To create a database, first, use makeindex.py to create an index from a FASTA file, and then
# use `embeddings.py` to calculate embeddings based on this index file.

# Query sequence needs to be represented as a one-item database. To create such a query database
# from a one-sequence FASTA file use `query_emb.py`

# example cases `A9A4Y8`, `cupredoxin`
case='cupredoxin'

# data paths
INDIR="./input"
OUTDIR="./output"

QUERY_INDEX="$OUTDIR/${case}.csv"
OUTFILE="$OUTDIR/${case}.hits.csv"
#OUTFILE_MERGED="$OUTDIR/${case}.hits_merged.csv"
DB_PATH="/home/nfs/kkaminski/PLMBLST/ecod30db_20220902"

ALIGNMENT_CUTOFF="0.3"
COSINE_CUTOFF=95

NUM_WORKERS=10

mkdir -p $OUTDIR

if [ ! -f $OUTDIR/$case.pt_emb.p ]; then
	echo "calculate query embedding"
	python query_emb.py $INDIR/$case.fas $OUTDIR/$case.pt_emb.p $QUERY_INDEX
fi

if [ ! -f $OUTFILE ]; then
	# search pre-calculated ECOD database
	python run_plm_blast.py \
		$DB_PATH \
		$OUTDIR/$case \
		$OUTFILE \
		-cosine_percentile_cutoff $COSINE_CUTOFF \
		-alignment_cutoff $ALIGNMENT_CUTOFF \
		-workers $NUM_WORKERS \
		-use_chunks
fi

# plot hits
python plot.py $OUTFILE $QUERY_INDEX $OUTDIR/$case.hits_score_ecod.png -mode score -ecod
python plot.py $OUTFILE $QUERY_INDEX $OUTDIR/$case.hits_qend_ecod.png -mode qend -ecod

