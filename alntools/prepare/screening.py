import os
import argparse
from typing import List, Dict, Optional
from typing import Union

from tqdm import tqdm
import torch
from torch.nn.functional import avg_pool1d

from ..filehandle import DataObject
from ..density.local import chunk_cosine_similarity, calculate_pool_embs
from ..density import load_and_score_database
from ..density.parallel import load_embeddings_parallel_generator
from ..density import batch_slice_iterator
from ..settings import EMB64_EXT, SCR_BATCH_SIZE


def apply_database_screening(args: argparse.Namespace,
                            querydata: DataObject,
                            dbdata: DataObject,
                            stride: int = 10) -> Dict[int, List[str]]:
    '''
    apply pre-screening for database search
    Args:
        args (namespace):
        query_embs (list[torch.Tensor]) query embeddings
        dbsize (int) size of database - number of sequences
    Returns:
        (dict) each key is query_id, and values are embeddings above threshold
    '''
    # set torch num CPU limit
    num_workers_loader = 0
    num_queries = querydata.size
    percentile_factor = args.COS_PER_CUT/100
    torch.set_num_threads(args.workers)
    if 0 < args.COS_PER_CUT < 100 and dbdata.size > 10:
        print(f"Pre-screening with {args.COS_PER_CUT} quantile")
        query_filedict = dict()
        dbpath = os.path.join(args.db, EMB64_EXT)
        querypath = os.path.join(args.query, EMB64_EXT)
        if not os.path.isfile(dbpath):
            print(
                f'''missing pooled embedding file {dbpath} for given database, it will be generated on fly,
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
                torch.save(db_embs, dbpath)
            except Exception as e:
                print(f'cannot write {dbpath} due to: {e}')
        else:
            db_embs: List[torch.Tensor] = torch.load(dbpath)
        if args.use_chunks:
            if args.verbose:
                print('Loading database for chunk cosine similarity screening...')
            # look for pooled emb file
            if not os.path.isfile(querypath):
                if querydata.datatype == "dir":
                    query_embs_chunkcs = list()
                    for embs in load_embeddings_parallel_generator(querydata.embeddingpath,
                                                                   num_records=querydata.size,
                                                                   num_workers=0):
                        query_embs_chunkcs.extend(calculate_pool_embs(embs))
                else:
                    query_embs_chunkcs = torch.load(querydata.embeddingpath)
                    query_embs_chunkcs = calculate_pool_embs(query_embs_chunkcs)
            else:
                query_embs_chunkcs = torch.load(querypath)
            # loop over all query embeddings
            index = 0
            with tqdm(total=num_queries, desc='screening seqences') as pbar:
                for embslice in batch_slice_iterator(num_queries, batchsize=SCR_BATCH_SIZE):
                    filedict_batch = chunk_cosine_similarity(
                                                        query=query_embs_chunkcs[embslice],
                                                        targets=db_embs,
                                                        quantile=percentile_factor,
                                                        dataset_files=dbdata.dirfiles,
                                                        stride=stride)
                    for filedict in filedict_batch:
                        query_filedict[index] = filedict
                        index += 1
                        pbar.update(1)
            avg_hits = [len(v) for v in query_filedict.values()]
            avg_hits = int(sum(avg_hits)/len(avg_hits))
            print(f"{avg_hits} alignment candidates per query")
        else:
            if args.verbose:
                print('Using regular cosine similarity screening...')
            # TODO make sure that regular screening works
            for i, emb in tqdm(enumerate(load_embeddings_parallel_generator(querydata.embeddingpath,
                                                                             num_records=querydata.size,
                                                                             num_workers=0)),
                                                                               total=num_queries, desc='screening seqences'):
                filedict = load_and_score_database(emb,
                                                    dbpath=dbdata.embeddingpath,
                                                    num_records=dbdata.size,
                                                    quantile=percentile_factor,
                                                    num_workers=2)
                query_filedict[i] = filedict
    else:
        # no screening case
        print("Pre-screening skipped")
        filedict = {k: v for k, v in zip(range(dbdata.size), dbdata.dirfiles)}
        query_filedict = {queryid : filedict for queryid in range(num_queries)}
    return query_filedict
