import os
import sys
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from glob import glob


def get_parser() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser('''
                                     create chunk cosine similarity embeddings with reducted size for faster screening.
                                     Script will create .emb64 file within input directory''')
    parser.add_argument('inputdir', help='directory with embeddings')
    args = parser.parse_args()
    return args

# Directory with embeddings stored as separate files
dbpath = get_parser().inputdir
if not os.path.isdir(dbpath):
    raise NotADirectoryError(f'given path is not a directory: {dbpath}')


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

