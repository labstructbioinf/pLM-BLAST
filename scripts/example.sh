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
case='A9A4Y8'

# data paths
INDIR="./input"
OUTDIR="./output"

QUERY_INDEX="$OUTDIR/${case}.csv"
OUTFILE="$OUTDIR/${case}.hits.csv"
OUTFILE_MERGED="$OUTDIR/${case}.hits_merged.csv"
DB_PATH="/ssd/users/sdunin/db/localaln/ecod70db_20220902"
NUM_WORKERS=5

mkdir -p $OUTDIR

if [ ! -f $OUTDIR/$case.pt_emb.p ]; then
	echo "calculate query embedding"
	python ../embeddings.py $INDIR/$case.fas $OUTDIR/$case.pt_emb.p
fi

if [ ! -f $OUTFILE ]; then
	# search pre-calculated ECOD database
	python plm_blast.py \
		$DB_PATH \
		$OUTDIR/$case \
		$OUTFILE \
		-cosine_percentile_cutoff 99 \
		-alignment_cutoff 0.35 \
		-workers $NUM_WORKERS
fi

# pLM-BLAST tends to yield rather short hits therefore it is beneficial to merge those associated
# with a single database sequence; additionally, a more strict score cut-off is used
python merge.py $OUTFILE $OUTFILE_MERGED -score 0.35 # 0.39

# plot hits
python plot.py $OUTFILE_MERGED $QUERY_INDEX $OUTDIR/$case.hits_merged_score_ecod.png -mode score -ecod
python plot.py $OUTFILE_MERGED $QUERY_INDEX $OUTDIR/$case.hits_merged_qend_ecod.png -mode qend -ecod

