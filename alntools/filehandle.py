import os
import math
from typing import List, Dict, Tuple
from collections import namedtuple
import itertools

from Bio import SeqIO
import pandas as pd
import torch

from .density import load_full_embeddings

extensions = ['.csv', '.p', '.pkl', '.fas', '.fasta']
record = namedtuple('record', ['id', 'file'])

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


def read_input_file(file: str, cname: str = "sequence") -> pd.DataFrame:
	'''
	read sequence file in format (.csv, .p, .pkl, .fas, .fasta)
	'''
	# gather input file
	if file.endswith('csv'):
		df = pd.read_csv(file)
	elif file.endswith('.p') or file.endswith('.pkl'):
		df = pd.read_pickle(file)
	elif file.endswith('.fas') or file.endswith('.fasta'):
		# convert fasta file to dataframe
		data = SeqIO.parse(file, 'fasta')
		# unpack
		data = [[i, record.description, str(record.seq)] for i, record in enumerate(data)]
		df = pd.DataFrame(data, columns=['id', 'description', 'sequence'])
		df.set_index('description', inplace=True)
	elif file == "":
		raise FileNotFoundError("empty string passed as input file")
	else:
		raise FileNotFoundError(f'''
                          could not find input query or database file with name `{file}`
                          expecting one of the extensions .csv, .p, .pkl, .fas or .fasta
                          make sure that both embeddings storage and sequence files are
                          in the same catalog with the same names
                          ''')
	
	if cname != '' and not (file.endswith('.fas') or file.endswith('.fasta')):
		if cname not in df.columns:
			raise KeyError(f'no column: {cname} available in file: {file}, columns: {df.columns}')
		else:
			print(f'using column: {cname} as sequence source')
			if 'seq' in df.columns and cname != 'seq':
				df.drop(columns=['seq'], inplace=True)
			df.rename(columns={cname: 'sequence'}, inplace=True)
	return df


class BatchLoader:
    asdir = True
    qdata = None
    _files_per_record = dict()
    _indices_per_record = dict()
    _iteratons_per_record = dict()
    def __init__(self, query_ids: List[str],
                  querypath: List[str],
                    filedict: Dict[int, Dict[int, str]],
                      batch_size: int = 300,
                      mode='emb'):

        assert len(query_ids)
        assert batch_size > 0
        assert isinstance(mode, str)
        assert mode in {"emb", "file"}
        
        self.mode = mode
        self.query_ids = query_ids
        self.querypath = querypath
        if not os.path.isdir(self.querypath):
            self.asdir = False
            self.qdata = torch.load(self.querypath + ".pt")

        self.batch_size = batch_size
        self.filedict = filedict
        self.num_records = len(self.filedict)
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
        for qid in self.query_ids:
             self._query_data_to_iteration += [qid]*self._iteratons_per_record[qid]
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
             query_id = self._query_data_to_iteration[self.current_iteration]
             query_files = self._query_flatten[self.current_iteration]
             query_dbindices = self._query_flatten_id[self.current_iteration]
             # load query embeddings
             if self.qdata is None:
                qembedding = os.path.join(self.querypath, f"{query_id}.emb")
                qembedding = self._load_batch([qembedding])[0]
             else:
                qembedding = self.qdata[query_id]
             # return embeddings
             if self.mode == 'emb':
                dbembeddings = self._load_batch(query_files)
            # return files
             else:
                 dbembeddings = query_files
             self.current_iteration += 1
             return query_id, query_dbindices, qembedding, dbembeddings
        else:
             raise StopIteration

    def _query_file_to_slice(self, query_id: int) -> Tuple[List[List[int]], List[List[str]]]:
        '''
        calculate file slices for each batch for given query_id
        '''
        files_per_qid: Dict[int, str] = self.filedict[query_id]
        assert isinstance(files_per_qid, dict)
        file_list = list(files_per_qid.values())
        index_list = list(files_per_qid.keys())
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
         
         embeddings = load_full_embeddings(filelist, poolfactor=None)
         return embeddings