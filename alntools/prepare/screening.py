import os
import argparse
import warnings
from typing import List, Dict, Optional

from tqdm import tqdm
import torch
from torch.nn.functional import avg_pool1d

from ..density.local import chunk_cosine_similarity
from ..density import load_and_score_database
from ..density.parallel import load_embeddings_parallel_generator

def apply_database_screening(args: argparse.Namespace,
                            query_embs: List[torch.Tensor],
                            dbsize: Optional[str]) -> Dict[int, List[str]]:
    '''
    apply pre-screening for database search
    Args:
        args (namespace):
        query_embs (list[torch.Tensor]) query embeddings
        dbsize (int) size of database - number of sequences
    Returns:
        (dict) each key is query_id, and values are embeddings above threshold
    '''
    assert len(query_embs) > 0
    assert isinstance(query_embs, list)
    # set torch num CPU limit
    torch.set_num_threads(args.MAX_WORKERS)
    num_queries = len(query_embs)
    if 0 < args.COS_PER_CUT < 100 and dbsize > 1:
        query_filedict = dict()
        dbfile = os.path.join(args.db, 'emb.64')
        if not os.path.isfile(dbfile):
            print(
                f'''missing pooled embedding file {dbfile} for given database, it will be generated on fly,
                and saved. Depending on run specification this may decrease performence of the first run,
                especially for larger databases. It can be created manually by scripts/dbtofile.py''')
            # load regular database and pool
            # find db structure
            if os.path.isfile(args.db + ".pt"):
                db_embs = torch.load(args.db + ".pt")
                db_embs = calculate_pool_embs(db_embs)
            else:
                # generator version to reduce RAM usage
                db_embs = list()
                for embs in load_embeddings_parallel_generator(args.db, num_records=dbsize):
                    db_embs.extend(calculate_pool_embs(embs))
            # try to write emb.64 file
            try:
                torch.save(db_embs, dbfile)
            except Exception as e:
                print(f'cannot write {dbfile} due to: {e}')
        else:
            db_embs: List[torch.Tensor] = torch.load(dbfile)
        if args.use_chunks:
            if args.verbose:
                print('Loading database for chunk cosine similarity screening...')
            filelist = [os.path.join(args.db, f'{f}.emb') for f in range(0, dbsize)]
            query_embs_chunkcs = calculate_pool_embs(query_embs)
            # loop over all query embeddings
            for i, emb in tqdm(enumerate(query_embs_chunkcs), total=num_queries, desc='screening seqences'):
                filedict = chunk_cosine_similarity(
                    query=emb,
                    targets=db_embs,
                    quantile=args.COS_PER_CUT/100,
                    dataset_files=filelist,
                    stride=10)
                query_filedict[i] = filedict
        else:
            if args.verbose:
                print('Using regular cosine similarity screening...')
            # TODO make sure that regular screening works
            for i, emb in tqdm(enumerate(query_embs), total=num_queries, desc='screening seqences'):
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