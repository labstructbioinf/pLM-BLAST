import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from glob import glob

# Directory with embeddings stored as separate files
dbpath = sys.argv[1]

# Adjustable (see publication for details)
factor = 64
kernel = int(1024/factor)

embfilelist = glob(os.path.join(dbpath, '*.emb'))
embpoollist = []

for file in tqdm(embfilelist):
    emb = torch.load(file).float()
    if emb.ndim != 2:
        raise ValueError(f'embedding file dim is invalid: {emb.ndim}')
    embpool = torch.nn.functional.avg_pool1d(emb.unsqueeze(0), kernel).squeeze()
    embpoollist.append(embpool)

outfile = os.path.join(dbpath, f'emb.{factor}')
torch.save(embpoollist, outfile)
print(f'Done! {outfile} created')

