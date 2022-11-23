import sys
import os
from typing import Union

from tqdm import tqdm
import pandas as pd
import torch
from torch.nn.functional import avg_pool1d

class Database(torch.utils.data.Dataset):
    def __init__(self, dbpath : os.PathLike, query_emb : torch.Tensor, suffix :str = '.emb', device : torch.device = torch.device('cpu')):

        self.query_emb =  query_emb.sum(0, keepdim=True)
        self.query_emb = avg_pool1d(self.query_emb, 2).T
        self.device = device
        dirname = os.path.dirname(dbpath)
        if not (dirname == ''):
            if not os.path.isdir(dirname):
                raise FileExistsError(f'directory: {dirname} is bad')
        self.basedata = pd.read_csv(dbpath + '.csv')
        num_records = self.basedata.shape[0]
        self.embedding_files = [f'{i}{suffix}' for i in range(num_records)]
        # add preffix
        self.embedding_files = [os.path.join(dbpath, ind) for ind in self.embedding_files]

    def __len__(self):
        return self.basedata.shape[0]

    def __getitem__(self, idx):
        embedding = torch.load(self.embedding_files[idx])
        #embedding = embedding.sum(0, keepdim=True)
        #embedding = avg_pool1d(embedding, 2).T
        #score = torch.nn.functional.cosine_similarity(self.query_emb, embedding, dim=0)
        return embedding

def collate_fn(batch):
    (emb, score) = zip(*batch)
    return emb, score

def load_and_score_database(query_emb : torch.Tensor, dbpath: os.PathLike, threshold : float = 0.2, device : torch.device = torch.device('cpu')):

    assert 0 < threshold < 1
    batch_size = 32
    pooling = 1
    num_workers = 2
    dataset = Database(dbpath=dbpath, query_emb=query_emb)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn = lambda x: list(x), num_workers = num_workers)
    num_batches = int(len(dataset)/batch_size)
    # equivalent to math.ceil
    num_batches = num_batches if num_batches*batch_size <= len(dataset) else num_batches + 1
    if query_emb.shape[0] != 1:
        query_emb = query_emb.sum(0, keepdim=True).T
    if query_emb.ndim == 1:
        query_emb = query_emb.view(-1, 1)
    if pooling > 1:
        query_emb = avg_pool1d(query_emb.T, pooling).T
    database_embeddings = []
    with tqdm(total = num_batches) as pbar:
        for i, batch in enumerate(dataloader):
            batch_summed = [emb.sum(0, keepdim=True).T for emb in batch]
            batch_stack = torch.cat(batch_summed, dim=1).to(device)
            score = batch_cosine_similarity(query_emb, batch_stack, poolfactor=pooling)
            score = score.tolist()
            #batch_positive = [batch[idx] for idx, sc in enumerate(score) if sc > threshold]
            #database_embeddings.extend(batch_positive)
            pbar.update(1)
    return database_embeddings

torch.set_num_threads(6)
@torch.jit.script
def batch_cosine_similarity(x : torch.Tensor, B : torch.Tensor, poolfactor: int):

    if poolfactor > 1:
        B = avg_pool1d(B.T, poolfactor).T
    score = torch.nn.functional.cosine_similarity(x, B, dim=0)
    return score
