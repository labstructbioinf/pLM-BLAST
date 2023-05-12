#!/bin/bash

TMVECBASE="/home/users/kkaminski/apps/tm-vec"
TMBIN="${TMVECBASE}/scripts/tmvec-search"
TMVECDATA="/home/nfs/sdunin/calc/localaln/tmvec/ver1"
PLMBLAST="/home/users/kkaminski/apps/pLM-BLAST/scripts/input/A9A4Y8.fas"

DBFILE="/home/nfs/kkaminski/PLMBLST/tmvec/tmvec/ecod30db_20220902/db.npy"
DBFASTA="/home/nfs/kkaminski/PLMBLST/tmvec/tmvec/ecod30db_20220902/meta.npy"
OUTFILE="/home/nfs/kkaminski/data/tmvecout"
#"${TMVECDATA}/query2.fas" \
mkdir -p $OUTFILE
export PYTHONPATH=$PYTHONPATH:$TMVECBASE
datestart_sec=$(date -d "$datestart" '+%s')
python $TMBIN \
    --query $PLMBLAST \
    --database $DBFILE \
    --database-fasta $DBFASTA \
    --metadata $DBFASTA \
    --tm-vec-model "${TMVECDATA}/model/tm_vec_cath_model.ckpt" \
    --tm-vec-config "${TMVECDATA}/model/tm_vec_cath_model_params.json" \
    --output "${OUTFILE}/embeddings" \
    --k-nearest-neighbors 20000 \
    --device cpu

dateend=`date`
dateend_sec=$(date -d "$dateend" '+%s')
datediff=$(($dateend_sec - $datestart_sec))
printf "${TMBIN} sec: ${datediff}\n"