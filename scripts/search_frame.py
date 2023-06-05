import sys
import os
import argparse
# add one level up directory
sys.path.append('..')
print(os.path.dirname(__file__))
import torch
import pandas as pd
from tqdm import tqdm

from alntools.base import Extractor
import alntools.density as ds

# prequisitions
# how many records from dataframe should be processed in negative - uses all
limit_records = 20
# minimal length of alignment
MIN_SPAN_LEN = 20
# border indices step
BFACTOR = 2
# moving average window size
MA_WINDOW = 1
# standard deviation threshold applied when scoring alignment route
# the bigger value the more conservative algorithm is
SIGMA_FACTOR = 1
# measure alignment hits with available `pdb_list` column
# in this example `pdb_list` contain indices of the Rossmann core
MEASURE_ALIGNMENT_WITH_TEMP = True
RAW = True
VERBOSE = False
INPUTFILE="/home/nfs/kkaminski/PLMBLST/ecod70db_20220902"
db = INPUTFILE
#PATH_DATAFRAME = f"{INPUTFILE}.p"


def main():

    rtbdf = pd.read_csv(INPUTFILE + '.csv')
    print(f'input frame records: {rtbdf.shape[0]}')
    num_records = rtbdf.shape[0]
    num_iterations = int(num_records*(num_records-1)/2)
    # load embeddings
    module = Extractor()
    module.LIMIT_RECORDS = limit_records
    module.MIN_SPAN_LEN = MIN_SPAN_LEN
    module.WINDOW_SIZE = MA_WINDOW
    module.BFACTOR = BFACTOR
    module.SIGMA_FACTOR = SIGMA_FACTOR
    head = 100
    filelist = [os.path.join(db, f'{fileid}.emb') for fileid in range(0, 59990)]
    filelist = filelist[:head]
    filedict = {i : file for i, file in enumerate(filelist)}
    #assert dbsize == len(filelist)
    embedding_list = ds.load_full_embeddings(filelist=filelist, poolfactor=4)
    embedding_list = [emb.numpy() for emb in embedding_list]
    records_stack = list()
    with tqdm(total=num_iterations) as progress_bar:
        for query_idx, query_row in rtbdf.iterrows():
            if limit_records <= query_idx: break
            query_embedding = embedding_list[query_idx]
            for target_idx, target_row in rtbdf.iterrows():
                if query_idx <= target_idx: break
                # TODO Wrap into function
                target_embedding = embedding_list[target_idx]
                # generate similarity matrix
                results = module.full_compare(query_embedding, target_embedding, query_idx, 'unknown', 0.2)
                # find unique alignments
                # if no alignment is found move to the next iteration
                ################################
                # core overlap factor
                records_stack.append(results)
                progress_bar.update(1)
    
        if len(records_stack) == 0:
            if VERBOSE:
                print(f'no matches for {query_row.pdb_chain}  ({query_idx}) - {target_row.pdb_chain} ({target_idx})')
        results = pd.DataFrame(records_stack)
    print('total matches: ', results.shape[0])

if __name__ == "__main__":
    main()