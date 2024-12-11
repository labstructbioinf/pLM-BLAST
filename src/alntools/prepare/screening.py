import os
import gc
import argparse
from typing import List, Dict, Optional

from tqdm import tqdm
import torch

from ..filehandle import DataObject
from ..density.local import chunk_cosine_similarity, calculate_pool_embs, unfold_large_db
from ..density.parallel import load_embeddings_parallel_generator
from ..density.iterate import slice_iterator_with_seqlen
from ..settings import DBTYPE
from embedders.dataset import NPHandle
from .reduce_duplicates import reduce_duplicates_query_filedict


def read_embeddings_for_screening(
        datahandle: DataObject,
        workers: int = 4,
        ) -> List[torch.Tensor]:
    if not os.path.isfile(datahandle.poolpath):
        print(
            f'''missing pooled embedding file {datahandle.poolpath} for given database, it will be generated on fly,
            and saved. Depending on run specification this may decrease performence of the first run,
            especially for larger databases. It can be created manually by scripts/dbtofile.py''')
        # load regular database and pool
        # find db structure
        if datahandle.datatype == DBTYPE.file:
            embeddings_pooled = torch.load(datahandle.embeddingpath)
            embeddings_pooled = calculate_pool_embs(embeddings_pooled)
        elif datahandle.datatype == DBTYPE.dir:
            # generator version to reduce RAM usage
            embeddings_pooled = []
            for embs in load_embeddings_parallel_generator(datahandle.embeddingpath,
                                                            record=datahandle.plmblast_ids,
                                                            num_workers=workers):
                embeddings_pooled.extend(calculate_pool_embs(embs))
        elif datahandle.datatype == DBTYPE.npy:
            npyhandle = NPHandle(dbpath=datahandle.pathdata)
            embeddings_pooled = [ calculate_pool_embs(npyhandle.read(pidx)) 
                            for pidx in datahandle.plmblast_ids ]
        # try to write emb.64 file
        if datahandle.datatype != DBTYPE.file and datahandle.alias is None:
            try:
                torch.save(embeddings_pooled, datahandle.poolpath)
            except Exception as e:
                print(f'cannot write {datahandle.poolpath} due to: {e}')
    else:
        print(f"loading database pre-screening cache from: {datahandle.poolpath}")
        embeddings_pooled: List[torch.Tensor] = torch.load(datahandle.poolpath)
        if datahandle.alias is not None:
            embeddings_pooled = [embeddings_pooled[pidx] for pidx in datahandle.plmblast_ids]
    return embeddings_pooled


def apply_database_screening(
        args: argparse.Namespace,
        querydata: DataObject,
        dbdata: DataObject) -> Dict[int, List[str]]:
    '''
    apply pre-screening for database search
    
    Args:
        args: (namespace)
        query_embs: (list[torch.Tensor]) query embeddings
        dbsize: (int) size of database - number of sequences
    Returns:
        (dict) each key is query_id, and values are embeddings above threshold
    '''
    num_workers_loader = 0
    num_queries = querydata.size
    percentile_factor = args.COS_PER_CUT/100
    embdim: int = 64
    torch.set_num_threads(args.workers)
    if (0 < percentile_factor < 1 and dbdata.size > 10) or args.only_scan:
        print(f"Pre-screening params: {args.COS_PER_CUT} quantile, kernel size: {args.cpc_kernel_size}, stride: {args.cpc_stride}")
        query_filedict = dict()
        db_embs = read_embeddings_for_screening(dbdata, 4)
        if args.verbose:
            print('Loading database for chunk cosine similarity screening...')
        query_embs_chunkcs = read_embeddings_for_screening(querydata, 4)
        seqlen_query: list[int] = [q.shape[0] for q in query_embs_chunkcs]
        seqlen_db: list[int] = [q.shape[0] for q in db_embs]
        # check if embdim is same
        seq_embdim = [q.shape[1] for q in db_embs]
        if len(set(seq_embdim)) > 1:
            raise ValueError(f'db embedding has multiple sizes of embdim {set(seq_embdim)}')
        embdim = embdim if embdim < seq_embdim[0] else seq_embdim[0]
        # change kernel size if the shortest sequence in targets is smaller then kernel size
        kernel_size = min(min(seqlen_query + seqlen_db), args.cpc_kernel_size)
        # create unfolded db once per run - this will increase performence when dealing
        # with multiquery mode
        batchdb = unfold_large_db(db_embs, kernel_size=kernel_size, stride=args.cpc_stride, embdim=embdim)
        del db_embs
        # loop over all query embeddings
        with tqdm(total=num_queries, desc='screening seqences') as pbar:
            for embslice in slice_iterator_with_seqlen(seqlen_query):
                filedict_batch = chunk_cosine_similarity(
                                                    query=query_embs_chunkcs[embslice],
                                                    targets=batchdb,
                                                    quantile=percentile_factor,
                                                    dataset_files=dbdata.plmblast_ids,
                                                    stride=args.cpc_stride,
                                                    kernel_size=kernel_size)
                for index, filedict in zip(querydata.plmblast_ids[embslice], filedict_batch):
                    query_filedict[index] = filedict
                pbar.update(len(filedict_batch))
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
            dbid: {"file": file, "score": 1} 
                for dbid, file in zip(dbdata.plmblast_ids, dbdata.plmblast_ids)
                }
        query_filedict = {queryid : filedict.copy() for queryid in querydata.plmblast_ids}

    # remove redundancy from search space only usable when query is the same as db
    if args.reduce_duplicates:
        print("removing duplicated entires")
        query_filedict = reduce_duplicates_query_filedict(query_filedict)
    return query_filedict