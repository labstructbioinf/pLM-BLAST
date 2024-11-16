import os
import math
from typing import List, Dict, Tuple, Union, Optional, Literal
from collections import namedtuple
import itertools
import warnings

import numpy as np
from Bio import SeqIO
import pandas as pd
import torch

from alntools.settings import DBTYPE, DataType
from alntools.settings import EXTENSIONS, DBNPY, DBNPY_INDEX, EMB64_EXT
from embedders.dataset import NPHandle

ObjType = Literal["query", "database"]
DBType = Literal['dir', 'file', 'npy']
record = namedtuple('record', ['qid', 'qdbids' , 'dbfiles'])


class PLMBlastDBError(Exception):
    '''error coresponding to database format and its content'''
    pass


class DataObject:
     """
     core object to handle pLM-Blast script calls for either query or database
     """
     size: int = 0
     indexfile: str
     indexdata: pd.DataFrame
     datatype: DBTYPE = "dir"
     embeddingpath: str = ""
     # none if not exists
     poolpath: Optional[str] = None
     pathdata: str
     ext: str = ".emb"
     objtype: DataType = "query"

     def __init__(self, indexdata: pd.DataFrame, pathdata: str, objtype: DataType):

        self.pathdata = pathdata
        self.indexdata = indexdata
        self.objtype = objtype
        self.size = indexdata.shape[0]
        self._find_datatype()
        if objtype == DataType.db and self.dbtype == DBTYPE.file:
            raise PLMBlastDBError('''
                                  db argument must be a npy or directory database 
                                  (--npy or --asdir in embeddings.py) it looks like you 
                                  passed as file datatabase'''
                                  )
        print(f"loaded {self.objtype}: {self.pathdata} - in {self.datatype.value} mode  ({self.size}) seq total")
     
     @classmethod
     def from_dir(cls, pathdata: str, objtype: DataType):
        """
        find embeddings storage type
        """
        infile_with_extention = find_file_extention(pathdata)
        indexfile = read_input_file(infile_with_extention)
        if 'plmblastid' not in indexfile.columns:
            indexfile['plmblastid'] = list(range(0, indexfile.shape[0]))
        return cls(indexdata=indexfile, pathdata=pathdata, objtype=objtype)
     
     def _find_datatype(self):
        """
        determine input data format: dir, file or npy mode
        """
        self.embeddingpath = self.pathdata
        _dbnpy = os.path.join(self.pathdata, DBNPY)
        _dbnpy_index = os.path.join(self.pathdata, DBNPY_INDEX)
        _dbdir = os.path.join(self.pathdata, "0.emb") # at least one embedding in a directory
        _dbdir_index = self.pathdata + ".csv"
        #breakpoint()
        if os.path.isfile(_dbnpy) and os.path.isfile(_dbnpy_index):
            self.datatype = DBTYPE.npy
        elif os.path.isdir(self.pathdata) and \
            os.path.isfile(_dbdir) and os.path.isfile(_dbdir_index):
            self.datatype = DBTYPE.dir
        elif os.path.isfile(self.pathdata + ".pt"):
            self.datatype = DBTYPE.file
            self.embeddingpath += ".pt"
        elif os.path.isfile(self.pathdata + ".emb"):
            self.datatype = DBTYPE.file
            self.embeddingpath += ".emb"
        else:
             FileNotFoundError(f'''no valid database in given location: {self.pathdata},
                                make sure it contain {self.pathdata}.pt file or is a 
                                directory with .emb files or .npy file with .index.csv''')
        if self.datatype != DBTYPE.file:
            # only present in dir/npy mode
            self.poolpath = os.path.join(self.pathdata, EMB64_EXT)
    
     @property
     def dirfiles(self) -> Union[List[str], List[int]]:
          """
          return all files availabe for this dataobj
          """
          plmblastid = self.indexdata['plmblastid'].tolist()
          # find file locations
          if self.datatype == DBTYPE.dir:
                return [os.path.join(self.embeddingpath, f"{idx}{self.ext}") for idx in plmblastid]
          else:
                return plmblastid
          
               
def find_file_extention(infile: str) -> str:
    '''search for extension for query or index files'''
    assert isinstance(infile, str)
    infile_with_ext = infile
    for ext in EXTENSIONS:
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
	elif file.endswith(('.p', '.pkl')):
		df = pd.read_pickle(file)
	elif file.endswith(('.fas', '.fasta')):
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
    npyhandle = None
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
        # prepare query data
        self.query_ids = querydata.indexdata['plmblastid'].tolist()
        if dbdata.datatype == DBTYPE.file:
             self.dbasdir = False
             self.dbdata =  self._load_single_dir(dbdata.embeddingpath)
        elif dbdata.datatype == DBTYPE.npy:
            self.npyhandle = NPHandle(dbdata.pathdata, mode="r+")
        if querydata.datatype == DBTYPE.file:
            self.qasdir = False
            self.qdata = self._load_single_dir(querydata.embeddingpath)
        else:
             self.queryfiles = querydata.dirfiles
        # overwrite load methods depending on db type
        if querydata.datatype == DBTYPE.dir:
            setattr(self, "_load_single_query", self._load_single_dir)
            setattr(self, "_load_batch_query", self._load_batch_dir)
        elif querydata.datatype == DBTYPE.npy:
            setattr(self, "_load_single_query", self._load_single_npy)
            setattr(self, "_load_batch_query", self._load_batch_npy)
        if dbdata.datatype == DBTYPE.dir:
            setattr(self, "_load_single_db", self._load_single_dir)
            setattr(self, "_load_batch_db", self._load_batch_dir)
        elif dbdata.datatype == DBTYPE.npy:
            setattr(self, "_load_single_db", self._load_single_npy)
            setattr(self, "_load_batch_db", self._load_batch_npy)
        #breakpoint()
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
                qembedding = self._load_single_query(self.queryfiles[qdata.qid]).pop()
             else:
                qembedding = self.qdata[qdata.qid]
             # return embeddings
             if self.mode == 'emb':
                if self.dbdata is None:
                    dbembeddings = self._load_batch_db(qdata.dbfiles)
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
    
    def _load_batch_dir(self, filelist: List[str]) -> List[torch.FloatTensor]:
         
         embeddings = [torch.load(f).float().numpy() for f in filelist]
         return embeddings
    
    def _load_single_dir(self, f) -> List[np.ndarray]:
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
    
    def _load_single_npy(self, idx: int):
        '''
        Args:
            idx: (int) position in npyfile
        '''
        return [self.npyhandle.read(idx)]
    
    def _load_batch_npy(self, indexlist: List[int]):
        return [self.npyhandle.read(idx) for idx in indexlist]
