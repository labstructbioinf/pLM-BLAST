import os
import gc
import argparse
from typing import List, Dict, Optional

from tqdm import tqdm
import torch

from ..filehandle import DataObject
from ..density.local import chunk_cosine_similarity, calculate_pool_embs, unfold_large_db
from ..density import load_and_score_database
from ..density.parallel import load_embeddings_parallel_generator
from ..density.iterate import slice_iterator_with_seqlen
from ..settings import EMB64_EXT
from .reduce_duplicates import reduce_duplicates_query_filedict


def apply_database_screening(args: argparse.Namespace,
                            querydata: DataObject,
                            dbdata: DataObject,
                            stride: int = 10) -> Dict[int, List[str]]:
    '''
    apply pre-screening for database search
    Args:
        args: (namespace)
        query_embs: (list[torch.Tensor]) query embeddings
        dbsize: (int) size of database - number of sequences
    Returns:
        (dict) each key is query_id, and values are embeddings above threshold
    '''
    kernel_size = 30
    num_workers_loader = 0
    num_queries = querydata.size
    percentile_factor = args.COS_PER_CUT/100
    embdim: int = 64
    torch.set_num_threads(args.workers)
    if 0 < args.COS_PER_CUT < 100 and dbdata.size > 10:
        print(f"Pre-screening with {args.COS_PER_CUT} quantile")
        query_filedict = dict()
        if not os.path.isfile(dbdata.poolpath):
            print(
                f'''missing pooled embedding file {dbdata.poolpath} for given database, it will be generated on fly,
                and saved. Depending on run specification this may decrease performence of the first run,
                especially for larger databases. It can be created manually by scripts/dbtofile.py''')
            # load regular database and pool
            # find db structure
            if dbdata.datatype == "file":
                db_embs = torch.load(dbdata.embeddingpath)
                db_embs = calculate_pool_embs(db_embs)
            else:
                # generator version to reduce RAM usage
                db_embs = list()
                print('loading embeddings')
                for embs in load_embeddings_parallel_generator(dbdata.embeddingpath,
                                                               num_records=dbdata.size,
                                                               num_workers=num_workers_loader):
                    db_embs.extend(calculate_pool_embs(embs))
            # try to write emb.64 file
            try:
                torch.save(db_embs, dbdata.poolpath)
            except Exception as e:
                print(f'cannot write {dbdata.poolpath} due to: {e}')
        else:
            db_embs: List[torch.Tensor] = torch.load(dbdata.poolpath)
    
        if args.verbose:
            print('Loading database for chunk cosine similarity screening...')
        # look for pooled emb file
        if querydata.poolpath is None:
            if querydata.datatype == "dir":
                query_embs_chunkcs = list()
                for embs in load_embeddings_parallel_generator(querydata.embeddingpath,
                                                                num_records=querydata.size,
                                                                num_workers=0):
                    query_embs_chunkcs.extend(calculate_pool_embs(embs))
            # file mode - pool file is not available
            else:
                query_embs_chunkcs = torch.load(querydata.embeddingpath)
                query_embs_chunkcs = calculate_pool_embs(query_embs_chunkcs)
        # file exists
        else:
            query_embs_chunkcs = torch.load(querydata.poolpath)
        # loop over all query embeddings
        index = 0
        seqlen_query = [q.shape[0] for q in query_embs_chunkcs]
        seqlen_db = [q.shape[0] for q in db_embs]
        # check if embdim is same
        seq_embdim = [q.shape[1] for q in db_embs]
        if len(set(seq_embdim)) > 1:
            raise ValueError(f'db embedding has multiple sizes of embdim {set(seq_embdim)}')
        embdim = embdim if embdim < seq_embdim[0] else seq_embdim[0]
        # change kernel size if the shortest sequence in targets is smaller then kernel size
        kernel_size = min(min(seqlen_query + seqlen_db), kernel_size)
        # create unfolded db once per run - this will increase performence when dealing
        # with multiquery mode

        batchdb = unfold_large_db(db_embs, kernel_size=kernel_size, stride=stride, embdim=embdim)
        del db_embs
        print(f'kernel set to: {kernel_size}')
        with tqdm(total=num_queries, desc='screening seqences') as pbar:
            for embslice in slice_iterator_with_seqlen(seqlen_query):
                filedict_batch = chunk_cosine_similarity(
                                                    query=query_embs_chunkcs[embslice],
                                                    targets=batchdb,
                                                    quantile=percentile_factor,
                                                    dataset_files=dbdata.dirfiles,
                                                    stride=stride,
                                                    kernel_size=kernel_size)
                for filedict in filedict_batch:
                    query_filedict[index] = filedict
                    index += 1
                    pbar.update(1)
                gc.collect()
        #avg_hits = [len(v) for v in query_filedict.values()]
        #avg_hits = int(sum(avg_hits)/len(avg_hits))
        #print(f"{avg_hits} alignment candidates per query")
        del batchdb
        del query_embs_chunkcs
    else:
        # no screening case
        print("Pre-screening skipped")
        filedict: Dict[int, int] = {
            dbid: dict(file=file, condition=True, score=1) 
                for dbid, file in zip(range(dbdata.size), dbdata.dirfiles)
                }
        query_filedict = {queryid : filedict.copy() for queryid in range(num_queries)}
    # remove redundancy from search space only usable when query is the same as db
    if args.reduce_duplicates:
        print("removing duplicated entires")
        query_filedict = reduce_duplicates_query_filedict(query_filedict)
    return query_filedict
