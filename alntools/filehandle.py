import os
import math
from typing import List, Dict, Tuple, Union
from collections import namedtuple
import itertools
import warnings

import numpy as np
from Bio import SeqIO
import pandas as pd
import torch


extensions = ['.csv', '.p', '.pkl', '.fas', '.fasta']
record = namedtuple('record', ['qid', 'qdbids' , 'dbfiles'])


class DataObject:
     """
     core object to handle pLM-Blast script calls for either query or database
     """
     size: int = 0
     indexfile: str
     indexdata: pd.DataFrame
     datatype: str = "dir"
     embeddingpath: str = ""
     pathdata: str
     ext: str = ".emb"
     objtype: str = "query"

     def __init__(self, indexdata: pd.DataFrame, pathdata: str, objtype: str = "query"):

        self.pathdata = pathdata
        self.indexdata = indexdata
        self.objtype = objtype
        self.size = indexdata.shape[0]
        self._find_datatype()
        print(f"loaded {self.objtype}: {self.pathdata} - in {self.datatype} mode")
     
     @classmethod
     def from_dir(cls, pathdata: str, objtype: str = "query"):
        """
        path to data
        """
        infile_with_extention = find_file_extention(pathdata)
        indexfile = read_input_file(infile_with_extention)
        indexfile['run_index'] = list(range(0, indexfile.shape[0]))
        return cls(indexdata=indexfile, pathdata=pathdata, objtype=objtype)
     
     def _find_datatype(self):
          
        self.embeddingpath = self.pathdata
        if os.path.isdir(self.pathdata):
            self.datatype = 'dir'
        elif os.path.isfile(self.pathdata + ".pt"):
            self.datatype = 'file'
            self.embeddingpath += ".pt"
        else:
             FileNotFoundError(f'''no valid database in given location: {self.pathdata},
                                make sure it contain {self.pathdata}.pt file or it is a 
                                directory with .emb files''')
    
     @property
     def dirfiles(self) -> Union[List[str], List[int]]:
          """
          return all files availabe for this dataobj
          """
          if self.datatype == "dir":
                return [os.path.join(self.embeddingpath, f"{idx}{self.ext}") for idx in self.indexdata['run_index'].tolist()]
          else:
                return self.indexdata['run_index'].tolist()
          
               
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
    Returns:
        pd.DataFrame: with columns: sequence, id and optionally description
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
		data = [[record.id, record.description, str(record.seq).upper()] for record in data]
		df = pd.DataFrame(data, columns=['id', 'description', 'sequence'])
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
			if 'seq' in df.columns and cname != 'seq':
				df.drop(columns=['seq'], inplace=True)
			df.rename(columns={cname: 'sequence'}, inplace=True)
	if 'id' not in df.columns or not df["id"].is_unique:
		df["id"] = list(range(0, df.shape[0]))
		warnings.warn("Id column is not unique, using index as id")
	return df


class BatchLoader:
    qasdir = True
    dbasdir = True
    # this will be always nan if datatype = file
    qdata = None
    dbdata = None
    queryfiles: List[str]
    _files_per_record = dict()
    _indices_per_record = dict()
    _iteratons_per_record = dict()
    _qdata_record: List[record] = list()
    current_iteration = 0
    def __init__(self,
                 querydata: DataObject,
                 dbdata: DataObject, 
                 filedict: Dict[int, Dict[int, str]],
                 batch_size: int = 300,
                 mode='emb'):

        assert batch_size > 0
        assert isinstance(mode, str)
        assert mode in {"emb", "file"}
        
        self.mode = mode
        self.query_ids = querydata.indexdata['run_index'].tolist()
        if querydata.datatype == "file":
            self.qasdir = False
            self.qdata = self._load_single(querydata.embeddingpath)
        else:
             self.queryfiles = querydata.dirfiles
        if dbdata.datatype == "file":
             self.dbasdir = False
             self.dbdata =  self._load_single(dbdata.embeddingpath)
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
        _query_data_to_iteration = list()
        _query_flatten: List[List[str]] = list(itertools.chain(*self._files_per_record.values()))
        _query_flatten_id: List[List[int]] = list(itertools.chain(*self._indices_per_record.values()))
        for qid in self.query_ids:
            _query_data_to_iteration += [qid]*self._iteratons_per_record[qid]
        # merge all needed data into single object
        for itr in range(self.num_iterations):
             self._qdata_record.append(record(qid=_query_data_to_iteration[itr],
                                       dbfiles=_query_flatten[itr],
                                       qdbids=_query_flatten_id[itr]))
        # checks
        assert len(self._qdata_record) == self.num_iterations, \
            f'{len(self._qdata_record)} != {self.num_iterations}'
        assert len(_query_flatten) == self.num_iterations
   
    def __len__(self):
         return self.num_iterations
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[int, List[int], np.ndarray, List[np.ndarray]]:
        if self.current_iteration < self.num_iterations:
             # get id
             qdata = self._qdata_record[self.current_iteration]
             # load query embeddings
             if self.qdata is None:
                qembedding = self._load_single(self.queryfiles[qdata.qid]).pop()
             else:
                qembedding = self.qdata[qdata.qid]
             # return embeddings
             if self.mode == 'emb':
                if self.dbdata is None:
                    dbembeddings = self._load_batch(qdata.dbfiles)
                else:
                     # if dbdata is single file
                    if len(self.dbdata) == 1:
                          dbembeddings = [self.dbdata[qdata.qdbids[0]]]
                    else:
                        dbembeddings = [self.dbdata[qdb] for qdb in qdata.qdbids]
            # return files
             else:
                 dbembeddings = qdata.dbfiles
             self.current_iteration += 1
             return qdata.qid, qdata.qdbids, qembedding, dbembeddings
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
         
         embeddings = [torch.load(f).float().numpy() for f in filelist]
         return embeddings
    
    def _load_single(self, f) -> List[np.ndarray]:
        """
        load torch file content
        Returns:
            list(np.ndarray) or np.ndarray
        """
        emb = torch.load(f)
        if isinstance(emb, list):
            emb = [e.float().numpy() for e in emb]
        else:
            emb = [emb.float().numpy()]
        return emb