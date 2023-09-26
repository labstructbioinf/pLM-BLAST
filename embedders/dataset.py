import os
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
    filename: str
    def __init__(self, filename : str):
        self.filename = filename

    def write_batch(self, emb_list: List[Any], start_index: Union[int, List[int]]):
        assert isinstance(emb_list, list)
        if isinstance(start_index, list):
            start_index = start_index[0]
        with h5py.File(self.filename, "w") as hf:
            if 'embeddings' in hf.keys():
                emb_group = hf['embeddings']
            else:
                emb_group = hf.create_group("embeddings")

            for index, emb in enumerate(emb_list, start=start_index):
                if isinstance(emb, torch.FloatTensor):
                    emb = emb.numpy()
                emb_group.create_dataset(
                    name=f'{self.preffix}{index}',
                      shape=emb.shape, data=emb,
                        **self.dataset_attrs)
    
    def read_batch(self, start, size) -> List[np.ndarray]:
        emb_list = list()
        with h5py.File(self.filename, 'r') as hf:
            if not 'embeddings' in hf.keys():
                raise KeyError('missing embedding group, probably the file is empty')
            else:
                emb_group = hf['embeddings']
            for i in range(start, size):
                dataset_name = f'{self.preffix}{i}'
                emb_list.append(emb_group[dataset_name])
        return emb_list