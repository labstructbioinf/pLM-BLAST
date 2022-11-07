
# Databases comprise two files, an index file (.csv) containing sequences and their descriptions,
# and an embeddings file (.pt_emb.p) containing PT5 embeddings of sequences listed in the index.

# To create a database, first, use makeindex.py to create an index from a FASTA file, and then
# use `embeddings.py` to calculate embeddings based on this index file.

# Query sequence needs to be represented as a one-item database. To create such a query database
# from a one-sequence FASTA file use `query_emb.py`

# example cases A9A4Y8, adhx
case='A9A4Y8'
# data paths
INDIR="./input"
OUTDIR="./output"


OUTFILE="$OUTDIR/${case}.csv"
OUTFILE_MERGED="$OUTDIR/${case}.hits_merged.csv"

if [ ! -f $OUTDIR/$case.pt_emb.p ]; then
	# calculate query embedding
	python query_emb.py $INDIR/$case.fas $OUTDIR/$case.pt_emb.p $OUTDIR/$case.csv
fi

if [ ! -f .$OUTDIR/$case.hits.csv ]; then
	# search pre-calculated ECOD database
	python plm_blast.py \
		/ssd/users/sdunin/db/localaln/ecod70db_20220902 \
		$OUTDIR/$case \
		$OUTDIR/$case.hits.csv \
		-cosine_percentile_cutoff 90 \
		-alignment_cutoff 0.35
fi

# pLM-BLAST tends to yield rather short hits therefore it is beneficial to merge those associated
# with a single database sequence
python merge.py $OUTDIR/$case.hits.csv $OUTFILE_MERGED

# plot hits
python plot.py $OUTFILE_MERGED $OUTFILE $OUTDIR/$case.hits_merged_score.png -mode score
python plot.py $OUTFILE_MERGED $OUTFILE $OUTDIR/$case.hits_merged_score_ecod.png -mode score -ecod

python plot.py $OUTFILE_MERGED $OUTFILE $OUTDIR/$case.hits_merged_qend.png -mode qend
python plot.py $OUTFILE_MERGED $OUTFILE $OUTDIR/$case.hits_merged_qend_ecod.png -mode qend -ecod

