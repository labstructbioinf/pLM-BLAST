import os
from typing import Union, List
import warnings

from tqdm import tqdm
import pandas as pd
import torch
from torch.nn.functional import avg_pool1d


class Database(torch.utils.data.Dataset):
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
        if len(self.embedding_files) != num_records:
            warnings.warn(f'{num_records - len(self.embedding_files)} missing protein embeddings')

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
                            threshold : float = 0.2,
                            device : torch.device = torch.device('cpu')) -> List[str]:

    assert 0 < threshold < 1
    batch_size = 256
    pooling = 1
    num_workers = 1
    verbose = True
    threshold = 0.2
    # setup database
    dataset = Database(dbpath=dbpath)
    dataloader = torch.utils.data.DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn = lambda x: torch.cat(x, dim=1),
                                num_workers = num_workers)
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
            batch = batch.to(device)
            score = batch_cosine_similarity(query_emb, batch, poolfactor=pooling)
            scorestack.append(score)
            pbar.update(1)
    scorestack = torch.cat(scorestack, dim=0)
    scoremask = (scorestack > threshold)
    scoreidx = torch.nonzero(scoremask, as_tuple=False).tolist()
    filelist = [file for (file, cond) in zip(dataset_files, scoreidx) if cond]
    if verbose:
        print(f'{len(scoreidx)}/{len(dataset_files)}')
        print(len(dataset))
        print(scoremask.sum())
    assert scorestack.shape == len(dataset_files)
    return filelist


@torch.jit.script
def batch_cosine_similarity(x : torch.Tensor, B : torch.Tensor, poolfactor: int):

    if poolfactor > 1:
        B = avg_pool1d(B.T, poolfactor).T
    score = torch.nn.functional.cosine_similarity(x, B, dim=0)
    return score
