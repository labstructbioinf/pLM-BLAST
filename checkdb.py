import os
import torch
import pandas as pd
from tqdm import tqdm


pathdb = "/home/nfs/kkaminski/PLMBLST/ecod70db_20220902"
dbdata = pd.read_csv(pathdb + '.csv')
dbdata = dbdata.reset_index()

num_miss = 0
num_miss_shape = 0
sizediff_list = []
for idx, row in tqdm(dbdata.iterrows()):

    file = os.path.join(pathdb, str(idx) + '.emb.256')
    if not os.path.isfile(file):
        print('missing file ', file)
        num_miss += 1
    else:
        emb = torch.load(file)
        if emb.shape[0] != len(row.sequence):
            sizediff = len(row.sequence) - emb.shape[0]
            sizediff_list.append(sizediff)
            num_miss_shape += 1

    if idx > 10:
        break
print('missing embs:' ,num_miss)
print('num_miss_shape: ', num_miss_shape)
print(pd.Series(sizediff_list).describe())