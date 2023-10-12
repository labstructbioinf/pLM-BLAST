import os
import time
from typing import List, Any, Union

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