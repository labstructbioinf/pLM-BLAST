import os
import argparse
from typing import List, Dict, Optional
from typing import Union

from tqdm import tqdm
import torch
from torch.nn.functional import avg_pool1d

from filehandle import DataObject
from ..density.local import chunk_cosine_similarity
from ..density import load_and_score_database
from ..density.parallel import load_embeddings_parallel_generator
from ..density.local import batch_slice_iterator


def apply_database_screening(args: argparse.Namespace,
                            querydata: DataObject,
                            dbdata: DataObject,
                            batchsize: int = 256,
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
    torch.set_num_threads(args.workers)
    num_workers_loader = 0
    num_queries = querydata.size
    if 0 < args.COS_PER_CUT < 100 and dbdata.size > 1:
        query_filedict = dict()
        dbpath = os.path.join(args.db, 'emb.64')
        querypath = os.path.join(args.query, 'emb.64')
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
                for embs in load_embeddings_parallel_generator(args.db, num_records=dbdata.size, num_workers=num_workers_loader):
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
            if not os.path.isfile(querypath):
                if querydata.datatype == "dir":
                    query_embs_chunkcs = list()
                    for embs in load_embeddings_parallel_generator(args.query, num_records=querydata.size, num_workers=0):
                        query_embs_chunkcs.extend(calculate_pool_embs(embs))
                    else:
                        query_embs_chunkcs = torch.load(querydata.embeddingpath)
                        query_embs_chunkcs = calculate_pool_embs(query_embs_chunkcs)
            else:
                query_embs_chunkcs = torch.load(querypath)
            # loop over all query embeddings
            index = 0
            with tqdm(total=num_queries, desc='screening seqences') as pbar:
                for embslice in batch_slice_iterator(num_queries, batchsize):
                    filedict_batch = chunk_cosine_similarity(
                                                        query=query_embs_chunkcs[embslice],
                                                        targets=db_embs,
                                                        quantile=args.COS_PER_CUT/100,
                                                        dataset_files=filelist,
                                                        stride=stride)
                    for filedict in filedict_batch:
                        query_filedict[index] = filedict
                        index += 1
                        pbar.update(1)
        else:
            if args.verbose:
                print('Using regular cosine similarity screening...')
            # TODO make sure that regular screening works
            for i, emb in tqdm(enumerate(load_embeddings_parallel_generator(args.query,num_records=dbsize,num_workers=0)), total=num_queries, desc='screening seqences'):
                filedict = load_and_score_database(emb,
                                                    dbpath=args.db,
                                                    num_records=dbsize,
                                                    quantile=args.COS_PER_CUT/100,
                                                    num_workers=2)
                query_filedict[i] = filedict
    else:
        # no screening case
        print("screening skipped")
        filelist = [os.path.join(args.db, f'{f}.emb') for f in range(0, dbsize)]  # db_df is a database index
        filedict = {k: v for k, v in zip(range(len(filelist)), filelist)}
        query_filedict = {queryid : filedict for queryid in range(num_queries)}
    return query_filedict


@torch.jit.script
def calculate_pool_embs(embs: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    convert embeddings to torch.float32 and [seqlen, 64]
    """
    if len(embs) == 0:
        raise ValueError('target database is empty')
    return [avg_pool1d(emb.float().unsqueeze(0), 16).squeeze() for emb in embs]