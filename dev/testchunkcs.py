# %%
import os
import sys
from time import perf_counter as timeit
sys.path.append('..')
import torch
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm

import alntools as aln
import alntools.density as ds

# %%
example_emb = "/home/nfs/kkaminski/PLMBLST/ecod70db_20220902"
dbfile = example_emb + '/emb.64'
def padlast(x):
    '''
    [seqlen, emb]
    '''
    x = torch.nn.functional.avg_pool1d(x, 8)
    return x
# %%
if not os.path.isfile(dbfile):
    filelist = [os.path.join(example_emb, f'{f}.emb') for f in range(0, 59990)]
    embedding_list = ds.load_full_embeddings(filelist, poolfactor=16)
    torch.save(embedding_list, dbfile)
else:
    embedding_list = torch.load(dbfile)
print(len(embedding_list))
print(embedding_list[0].shape)
# %%
X = embedding_list[20000]
scorelist = []

scorelist = ds.local.chunk_score(X, embedding_list, stride = 3, kernel_size = 20)
#print(scorelist)
print(pd.Series(scorelist).quantile(np.arange(0, 1.02, 0.02)))
# %%



