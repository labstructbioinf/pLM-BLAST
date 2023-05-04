#!/bin/bash


case='A9A4Y8'

# data paths
INDIR="./input"
OUTDIR="./output"

QUERY_INDEX="$OUTDIR/${case}.csv"
OUTFILE="$OUTDIR/${case}.hits.csv"
OUTFILE_MERGED="$OUTDIR/${case}.hits_merged.csv"
DBPATH="/home/nfs/kkaminski/PLMBLST/"
ALIGNMENT_CUTOFF="0.30"
COSINE_CUTOFF=99
NUM_WORKERS=6

datestart=`date`
datestart_sec=$(date -d "$datestart" '+%s')
python devel_plm_blast.py \
		$DBPATH/ecod100db_20220902 \
		$OUTDIR/$case \
		$OUTFILE \
		-cosine_percentile_cutoff $COSINE_CUTOFF \
		-alignment_cutoff $ALIGNMENT_CUTOFF \
		-workers $NUM_WORKERS \
		-use_chunkcs

dateend=`date`
dateend_sec=$(date -d "$dateend" '+%s')
datediff=$(($dateend_sec - $datestart_sec))
printf "${datediff}\n"

datestart=`date`
datestart_sec=$(date -d "$datestart" '+%s')
python devel_plm_blast.py \
		$DBPATH/ecod70db_20220902 \
		$OUTDIR/$case \
		$OUTFILE \
		-cosine_percentile_cutoff $COSINE_CUTOFF \
		-alignment_cutoff $ALIGNMENT_CUTOFF \
		-workers $NUM_WORKERS \
		-use_chunkcs


dateend=`date`
dateend_sec=$(date -d "$dateend" '+%s')
datediff=$(($dateend_sec - $datestart_sec))
printf "${datediff}\n"

datestart=`date`
datestart_sec=$(date -d "$datestart" '+%s')
python devel_plm_blast.py \
		$DBPATH/pdb30db_20220902 \
		$OUTDIR/$case \
		$OUTFILE \
		-cosine_percentile_cutoff $COSINE_CUTOFF \
		-alignment_cutoff $ALIGNMENT_CUTOFF \
		-workers $NUM_WORKERS \
		-use_chunkcs

dateend=`date`
dateend_sec=$(date -d "$dateend" '+%s')
datediff=$(($dateend_sec - $datestart_sec))
printf "${datediff}\n"


