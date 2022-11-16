# Databases comprise two files, an index file (.csv) containing sequences and their descriptions,
# and an embeddings file (.pt_emb.p) containing PT5 embeddings of sequences listed in the index.

# To create a database, first, use makeindex.py to create an index from a FASTA file, and then
# use `embeddings.py` to calculate embeddings based on this index file.

# Query sequence needs to be represented as a one-item database. To create such a query database
# from a one-sequence FASTA file use `query_emb.py`

# example cases `A9A4Y8`, `cupredoxin`
case='apaf1'

# data paths
INDIR="./input"
OUTDIR="./output"

QUERY_INDEX="$OUTDIR/${case}.csv"
OUTFILE="$OUTDIR/${case}.hits.csv"
OUTFILE_MERGED="$OUTDIR/${case}.hits_merged.csv"

if [ ! -f $QUERY_INDEX ]; then
	# calculate query embedding
	python query_emb.py $INDIR/$case.fas $OUTDIR/$case.pt_emb.p $QUERY_INDEX
fi

if [ ! -f $OUTFILE ]; then
	# search pre-calculated ECOD database
	python plm_blast.py \
		/ssd/users/sdunin/db/localaln/ecod70db_20220902 \
		$OUTDIR/$case \
		$OUTFILE \
		-cosine_percentile_cutoff 99 \
		-alignment_cutoff 0.35
fi

# pLM-BLAST tends to yield rather short hits therefore it is beneficial to merge those associated
# with a single database sequence; additionally, a more strict score cut-off is used
python merge.py $OUTFILE $OUTFILE_MERGED -score 0.35 # 0.39

# plot hits
python plot.py $OUTFILE_MERGED $QUERY_INDEX $OUTDIR/$case.hits_merged_score_ecod.png -mode score -ecod
python plot.py $OUTFILE_MERGED $QUERY_INDEX $OUTDIR/$case.hits_merged_qend_ecod.png -mode qend -ecod

