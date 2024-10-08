import os
import time
from typing import List, Any, Union, Tuple

import pandas as pd
import h5py 
import numpy as np
import torch


class HDF5Handle:
    dataset_attrs = {
        'compression' : 'lzf',
        'dtype': np.float32
        }
    groupname: str = 'embeddings'
    preffix: str = 'emb_'
    direct_read: bool = True
    filename: str
    wait_time: float = 0.1
    def __init__(self, filename : Union[str, os.PathLike]):
        self.filename = filename
        # TODO create if not exists

    def write_batch(self, emb_list: List[Any], start_index: Union[int, List[int]]):
        assert isinstance(emb_list, list)
        if isinstance(start_index, list):
            batch_iter = zip(start_index, emb_list)
        else:
            batch_iter = enumerate(emb_list, start_index)
        with h5py.File(self.filename, "a") as hf:
            emb_group = hf.require_group(self.groupname)
            for index, emb in batch_iter:
                if isinstance(emb, torch.FloatTensor):
                    emb = emb.numpy()
                emb_group.create_dataset(
                    name=f'{self.preffix}{index}',
                      shape=emb.shape, data=emb,
                        **self.dataset_attrs)
    
    def write_batch_mp(self, emb_list: List[Any], start_index: Union[int, List[int]]):
        '''
        handle multiprocess writing (wait until other process finishes writing)
        '''
        is_saved = False
        attemps = 0
        while not is_saved:
            try:
                self.write_batch(emb_list, start_index)
                is_saved = True
            except:
                attemps += 1
                time.sleep(self.wait_time*attemps)


    def read_batch(self, start, size = None) -> List[np.ndarray]:
        '''
        if size is none read all record from start to the end
        '''
        emb_list = list()
        with h5py.File(self.filename, 'r') as hf:
            if not 'embeddings' in hf.keys():
                raise KeyError('missing embedding group, probably the file is empty')
            else:
                emb_group = hf['embeddings']
            if size is None:
                size = len(emb_group.keys())
            if start > size:
                raise ValueError(f'start >= then dataset size {start} >= {size}')
            for index in range(start, start+size):
                dataset_name = f'{self.preffix}{index}'
                emb_list.append(emb_group[dataset_name][:])
        return emb_list
    
    
class NPHandle:
    '''
    handle db operations on memmap numpy object
    db structure
    ```
    ├── dbpath.csv # sequences and metadata
    └── dbpath
        ├── db.npy 
        ├── db.index.csv
    ```
    '''
    dtype = np.float16
    index_file_name = "db.index.csv"
    dbfile_name = "db.npy"
    index_columns = ['startindex', 'seqlen']
    embdim: int = 1024
    dbfile: str = None
    dbfile_index: str = None
    cursor: np.ndarray = None
    # position of last embedding batch 
    _last_save_location: int = 0
    shape: Tuple[int, int] = None
    
    def __init__(self, dbpath: str, mode='r+', seqlens: List[int] = None):
        
        assert mode in {'r+', 'w+'}
        self.dbfile = os.path.join(dbpath, self.dbfile_name)
        self.dbfile_index = os.path.join(dbpath, self.index_file_name)
        # read index
        # read shape
        # set pointer/cursor
        if mode == 'r+':
            assert os.path.isdir(dbpath)
            assert os.path.isfile(self.dbfile)
            assert os.path.isfile(self.dbfile_index)
            self.index = pd.read_csv(self.dbfile_index)
            self.read_shape()
            self.fp = np.memmap(filename=self.dbfile,
                    dtype=self.dtype,
                    mode="r+",
                    shape=self.shape)
        
        # create index if not exists
        # calculate shape if file is not present
        if mode == 'w+':
            if not (os.path.isfile(self.dbfile) and os.path.isfile(self.dbfile_index)):
                if isinstance(seqlens, list):
                    self.initialize(seqlens)
                else:
                    raise ArgumentError("""
                        if you want to create a database you must first supply
                        seqlens
                        """)
      
    def read_shape(self):
        '''
        shape of database is equal to position of last protein + its len
        '''
        lastindex = self.index['startindex'].values[-1]
        lastlen = self.index['seqlen'].values[-1]
        self.shape = (lastindex + lastlen, self.embdim)
    
    def initialize(self, seqlens: List[int], embdim: int = 1024):
        '''
        create instance .npy and index.csv if not exists
        '''
        startindex = np.cumsum([0] + seqlens)
        self.shape = (startindex[-1] + seqlens[-1], embdim)
        # write index
        tmp = pd.DataFrame(data=zip(startindex, seqlens), columns=self.index_columns)
        os.makedirs(os.path.dirname(self.dbfile), exist_ok=True)
        tmp.to_csv(self.dbfile_index, index=False)
        fp = np.memmap(filename=self.dbfile, 
                       dtype=self.dtype,
                       mode="w+",
                       shape=self.shape)
        del fp
        
    def write(self, embbatch: torch.Tensor):
        """
        write single sequence
        """
        embbatch = embbatch.half().numpy()
        num_residues = embbatch.shape[0]
        fp = np.memmap(filename=self.dbfile,
                       dtype=self.dtype,
                       mode="w+",
                       shape=self.shape)
        fp[self._last_save_location:self._last_save_location + num_residues, :] = num_residues
        # apply changes to disk and remove
        del fp
    
    def readraw(self, startindex, seqlen) -> np.ndarray:
        '''
        read chunk of the database
        '''
        return self.fp[startindex:startindex+seqlen, :]
        
    def read(self, idx: int) -> np.ndarray:
        idrow = self.index.iloc[idx]
        return self.fp[idrow.startindex:idrow.startindex + idrow.seqlen, :]