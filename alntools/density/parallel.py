import sys
import os
from typing import Union

from tqdm import tqdm
import pandas as pd
import torch


class Database(torch.utils.data.Dataset):
    def __init__(self, dbpath : os.PathLike, device : torch.device = torch.device('cpu')):

        self.device = device
        dirname = os.path.dirname(dbpath)
        if not (dirname == ''):
            if not os.path.isdir(dirname):
                raise FileExistsError(f'directory: {dirname} is bad')
        self.basedata = pd.read_csv(dbpath + '.csv')
        num_records = self.basedata.shape[0]
        self.embedding_files = [f'{i}.pt.emb' for i in range(self.basedata.shape[0])]

    def __len__(self):
        return self.basedata.shape[0]

    def __getitem__(self, idx):
        embedding = torch.load(self.embedding_files[idx])
        return embedding

def load_and_score_database(query_emb, dbpath, threshold = 0.2):

    assert 0 < threshold < 1
    
    dataset = Database(dbpath=dbpath)
    dataloader = torch.utils.data.DataLoader(dataset, batchsize=64)

    database_embeddings = []
    for batch in dataloader:
        
        batch_summed = [emb.sum(0) for emb in batch]
        batch_stack = torch.stack(batch_summed)
        score = torch.nn.functional.cosine_similarity(query_emb.view(-1, 1), batch_stack)
        score = score.tolist()
        batch_positive = [batch[idx] for idx, sc in enumerate(batch) if sc > threshold]
        database_embeddings.extend(batch_positive)

    return database_embeddings