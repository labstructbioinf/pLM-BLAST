import os
from typing import Union, List, Dict
import warnings

from tqdm import tqdm
import pandas as pd
import torch
from torch.nn.functional import avg_pool1d


class Database(torch.utils.data.Dataset):
    '''
    handle loading database composed from single files
    '''
    def __init__(self, dbpath : os.PathLike, suffix :str = '.emb', device : torch.device = torch.device('cpu')):

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
        # exclude missing proteins
        self.embedding_files = [file for file in self.embedding_files if os.path.isfile(file)]
        num_missed = int(num_records - len(self.embedding_files))
        if num_missed < num_missed:
            warnings.warn(f'{num_records - len(self.embedding_files)} missing protein embeddings')
        elif num_missed == num_records:
            raise FileNotFoundError('no embeddings found')

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        embedding = torch.load(self.embedding_files[idx])
        return embedding

class DatabaseChunk(torch.utils.data.Dataset):
    '''
    handle loading database composed from single files
    '''
    def __init__(self, dbpath : List[os.PathLike], device : torch.device = torch.device('cpu')):

        self.device = device
        assert isinstance(dbpath, list)
        dirname = os.path.dirname(dbpath[0])
        if not (dirname == ''):
            if not os.path.isdir(dirname):
                raise FileExistsError(f'directory: {dirname} is bad')
        self.embedding_files = dbpath

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        embedding = torch.load(self.embedding_files[idx])
        return embedding

'''
def collate_fn(batch):
    (file, emb) = zip(*batch)
    emb = torch.cat(emb, dim=1).T
    filelist = list(file)
    return filelist, emb
'''

def load_and_score_database(query_emb : torch.Tensor,
                            dbpath: os.PathLike,
                            quantile : float = 0.9,
                            num_workers: int = 1,
                            device : torch.device = torch.device('cpu')) -> Dict[int, os.PathLike]:

    assert 0 < quantile < 1
    batch_size = 256
    pooling = 1
    num_workers = 1
    verbose = False
    threshold = 0.3
    # setup database
    dataset = Database(dbpath=dbpath, suffix='.emb.sum')
    dataloader = torch.utils.data.DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn = lambda x: torch.cat(x, dim=1),
                                num_workers = num_workers,
                                drop_last = False)
    num_batches = int(len(dataset)/batch_size)
    # equivalent to math.ceil
    num_batches = num_batches if num_batches*batch_size <= len(dataset) else num_batches + 1
    if query_emb.shape[0] != 1:
        query_emb = query_emb.sum(0, keepdim=True).T
    if query_emb.ndim == 1:
        query_emb = query_emb.view(-1, 1)
    if pooling > 1:
        query_emb = avg_pool1d(query_emb.T, pooling).T
    scorestack = []
    dataset_files = dataset.embedding_files
    with tqdm(total = num_batches) as pbar:
        for i, batch in enumerate(dataloader):
            if batch.shape[1] != 256:
                print(batch.shape)
            batch = batch.to(device)
            score = batch_cosine_similarity(query_emb, batch, poolfactor=pooling)
            scorestack.append(score)
            pbar.update(1)
    # quantile filter
    scorestack = torch.cat(scorestack, dim=0)
    quantile_threshold = torch.quantile(scorestack, quantile)
    scoremask = (scorestack >= quantile_threshold)
    scoreidx = torch.nonzero(scoremask, as_tuple=False).tolist()
    filedict = { i : file for i, (file, condition) in enumerate(zip(dataset_files, scoremask)) if condition }

    if verbose:
        print(f'{len(scoreidx)}/{len(dataset_files)}')
        print(len(dataset))
        print(scoremask.shape)
    assert scorestack.shape[0] == len(dataset_files)
    return filedict


@torch.jit.script
def batch_cosine_similarity(x : torch.Tensor, B : torch.Tensor, poolfactor: int):
    '''
    calculate cosine similarity for a batch of embeddings
    '''
    if poolfactor > 1:
        B = avg_pool1d(B.T, poolfactor).T
    score = torch.nn.functional.cosine_similarity(x, B, dim=0)
    return score


def load_full_embeddings(filelist : List[os.PathLike],
                            batch_size : int = 32,
                            num_workers : int = 1):

    stack = []
    dataset = DatabaseChunk(dbpath=filelist)
    dataloader = torch.utils.data.DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn = lambda x: x,
                                num_workers = num_workers,
                                drop_last = False)
    stack = []
    for batch in tqdm(dataloader):
        stack.extend(batch)
    return stack


