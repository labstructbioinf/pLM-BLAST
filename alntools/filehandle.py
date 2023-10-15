import os
import math
from typing import List, Dict, Tuple
import itertools

import torch

from .density import load_full_embeddings

extensions = ['.csv', '.p', '.pkl', '.fas', '.fasta']


def find_file_extention(infile: str) -> str:
    '''search for extension for query or index files'''
    assert isinstance(infile, str)
    infile_with_ext = infile
    for ext in extensions:
        if os.path.isfile(infile + ext):
            infile_with_ext = infile + ext
            break
    if infile_with_ext == "":
        raise FileNotFoundError(f'no matching index file {infile}')
    return infile_with_ext

class BatchLoader:
    
    def __init__(self, query_ids: List[str],
                  query_seqs: List[str],
                    filedict: Dict[int, Dict[int, str]],
                      batch_size: int = 300,
                      mode='emb'):

        assert len(query_ids) == len(query_seqs)
        assert len(query_seqs) > 0
        assert batch_size > 0
        assert isinstance(mode, str)
        assert mode in {"emb", "file"}

        self.mode = mode
        self.query_ids = query_ids
        self.query_seqs = query_seqs
        self.batch_size = batch_size
        self.filedict = filedict
        self.num_records = len(self.filedict)
        # total number of embeddings to load per each query
        self._files_per_record = dict()
        self._indices_per_record = dict()
        self._iteratons_per_record = dict()
        # calculate batch items for each query_id
        for qid in self.query_ids:
            batch_index_per_qid, batch_files_per_qid = self._query_file_to_slice(query_id=qid) 
            self._iteratons_per_record[qid] = len(batch_files_per_qid)
            self._files_per_record[qid] = batch_files_per_qid
            self._indices_per_record[qid] = batch_index_per_qid
        # calc iterator len
        self.num_iterations = sum(self._iteratons_per_record.values())
        # iterations/batches per query without need of knowing qid
        # each list element should be list of files for certain batch
        self._query_data_to_iteration = list()
        self._query_flatten: List[List[str]] = list(itertools.chain(*self._files_per_record.values()))
        self._query_flatten_id: List[List[int]] = list(itertools.chain(*self._indices_per_record.values()))
        for qid, qseq in zip(self.query_ids, self.query_seqs):
             self._query_data_to_iteration += [(qid, qseq)]*self._iteratons_per_record[qid]
        # checks
        assert len(self._query_data_to_iteration) == self.num_iterations, \
            f'{len(self._query_data_to_iteration)} != {self.num_iterations}'
        assert len(self._query_flatten) == self.num_iterations
        self.current_iteration = 0
   
    def __len__(self):
         return self.num_iterations
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_iteration < self.num_iterations:
             # find correct query id
             query_id, query_seq = self._query_data_to_iteration[self.current_iteration]
             query_files = self._query_flatten[self.current_iteration]
             query_indices = self._query_flatten_id[self.current_iteration]
             # return embeddings
             if self.mode == 'emb':
                embeddings = self._load_batch(query_files)
            # return files
             else:
                 embeddings = query_files
             self.current_iteration += 1
             return query_id, query_indices, embeddings
        else:
             raise StopIteration

    def _query_file_to_slice(self, query_id: int) -> Tuple[List[List[int]], List[List[str]]]:
        '''
        calculate file slices for each batch for given query_id
        '''
        files_per_qid: Dict[int, str] = self.filedict[query_id]
        file_list = list(files_per_qid.values())
        index_list = list(files_per_qid.keys())
        assert isinstance(files_per_qid, dict)
        num_files_per_qid = len(files_per_qid)
        num_batch = math.ceil(num_files_per_qid/self.batch_size)
        batch_start = 0
        batch_list = list()
        batch_index = list()
        for _ in range(num_batch):
            batch_end = batch_start + self.batch_size
            # clip value
            batch_end = min(batch_end, num_files_per_qid)
            batchslice = slice(batch_start, batch_end, 1)
            batch_filelist = file_list[batchslice]
            batch_indexlist = index_list[batchslice]
            batch_list.append(batch_filelist)
            batch_index.append(batch_indexlist)
            # update batch start position
            batch_start = batch_end
        return batch_index, batch_list 
    
    def _load_batch(self, filelist: List[str]) -> List[torch.FloatTensor]:
         
         embeddings = load_full_embeddings(filelist)
         return embeddings