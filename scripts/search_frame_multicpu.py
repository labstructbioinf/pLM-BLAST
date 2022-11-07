import sys
import os
import concurrent
# add one level up directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(os.path.dirname(__file__))
import torch
import pandas as pd
from tqdm import tqdm

from alntools.base import Extractor
from alntools.postprocess import measure_aln_overlap_with_pdblist, filter_result_dataframe

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
INPUTFILE = "datafull"
PATH_DATAFRAME = f"{INPUTFILE}.p"
PATH_EMBEDDINGS = f"{INPUTFILE}.emb.prottrans"
PATH_RESULTS = f"{INPUTFILE}.results.csv"

MAX_WORKERS = 6
WITH_SCORE = True
rtbdf = pd.read_pickle(PATH_DATAFRAME)
rtbdf['idx'] = list(range(rtbdf.shape[0]))
rtbdf['len_chain'] = rtbdf['seq_chain'].apply(len)
rtbdf = rtbdf.head(limit_records)
num_records = rtbdf.shape[0]
num_iterations = int(num_records*(num_records-1)/2)
print('config:')
print('WITH_SCORE: ', WITH_SCORE)
# load embeddings
if not os.path.isfile(PATH_EMBEDDINGS):
    raise FileNotFoundError('wrong embeddings path: ', PATH_EMBEDDINGS)
embeddings = torch.load(PATH_EMBEDDINGS)
print('embeddings loaded')
module = Extractor()
module.LIMIT_RECORDS = limit_records
module.MIN_SPAN_LEN = MIN_SPAN_LEN
module.WINDOW_SIZE = MA_WINDOW

def filter_spans(results, query_row, target_row) -> list:
    records_stack = list()
    if results.shape[0] != 0:
        results = filter_result_dataframe(results)
        if results.shape[0] == 0:
            assert 'error'
        # core overlap factor
        # calculate alignemnt - core pdb_list cover
        for _, sample in results.iterrows():
            cover_scores = measure_aln_overlap_with_pdblist(
                seq1_true=query_row.pdb_list,
                seq2_true=target_row.pdb_list,
                alignment=sample.indices)
            record = {
                'query_pdbchain' : query_row.pdb_chain,
                'target_pdbchain': target_row.pdb_chain,
                'query_cof': query_row.simplified_cofactor,
                'target_cof': target_row.simplified_cofactor,
                'path_score': sample.score,
                'len' : sample.len,
                **sample,
                **cover_scores
                }
            records_stack.append(record)
    return records_stack

def all_at_once(X, Y, query_row, target_row) -> list:
    global WITH_SCORE
    results = module.embedding_to_span(X, Y)
    if WITH_SCORE:
        results_with_score = filter_spans(results, query_row, target_row)
        return results_with_score
    else:
        return results


with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    job_stack = {}
    records_stack = []
    iter_id = 0
    for query_idx, query_row in rtbdf.iterrows():
        if limit_records <= query_idx: break
        query_embedding = embeddings[query_idx]
        for target_idx, target_row in rtbdf.iterrows():
            if query_idx <= target_idx: break
            target_embedding = embeddings[target_idx]
            job = executor.submit(all_at_once, query_embedding, target_embedding, query_row, target_row)
            job_stack[job] = iter_id

    with tqdm(total=num_iterations) as progress_bar:
        for job in concurrent.futures.as_completed(job_stack):
            data = job.result()
            records_stack.extend(data)
            progress_bar.update(1)

if VERBOSE:
    num_matches = len(records_stack)
    print('matches: ', num_matches)
datastack = pd.DataFrame(records_stack)
datastack.to_pickle(PATH_RESULTS)