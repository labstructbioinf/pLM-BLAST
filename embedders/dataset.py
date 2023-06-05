import h5py 
import numpy as np


class CreateDB:
    dataset_attrs = {'compression' : 'gzip'}
    def __init__(self, filename : str, num_files : int):

        self.file = h5py.File(filename, "w")
        self.dataset = self.file.create_dataset("embeddings", (num_files), **self.dataset_attrs)