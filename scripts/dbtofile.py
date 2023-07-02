import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

dbpath = sys.argv[1]
dbfile = pd.read_csv(dbpath + '.csv')
factor = 64
kernel = int(1024/factor)

embfilelist = [f'{i}.emb' for i in range(0, dbfile.shape[0])]
embfilelist = [os.path.join(dbpath, embfile) for embfile in embfilelist]
embpoollist = []

for file in tqdm(embfilelist):
    emb = torch.load(file).float()
    if emb.ndim != 2:
        raise ValueError(f'embedding file dim is invalid: {emb.ndim}')
    embpool = torch.nn.functional.avg_pool1d(emb.unsqueeze(0), kernel).squeeze()
    embpoollist.append(embpool)

torch.save(embpoollist, os.path.join(dbpath, f'emb.{factor}'))
print('done')

