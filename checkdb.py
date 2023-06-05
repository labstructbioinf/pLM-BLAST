import os
import argparse

import torch
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description =  
		"""
		validate database cohesion
		""")
parser.add_argument('--db', type=str, required=True, dest='db')
args = parser.parse_args()

pathdb = "/home/nfs/kkaminski/PLMBLST/ecod70db_20220902"
dbdata = pd.read_csv(pathdb + '.csv')
#dbdata = dbdata.reset_index()
print(dbdata.columns)
print(dbdata.head(10))

print(pd.Series(dbdata.sequence.apply(len)).tolist()[:100])
num_miss = 0
num_miss_shape = 0
sizediff_list = []
for idx, row in dbdata.iterrows():

    file = os.path.join(pathdb, str(idx) + '.emb')
    if not os.path.isfile(file):
        print('missing file ', file)
        num_miss += 1
    else:
        emb = torch.load(file)
        sequence = str(row.sequence)
        #print(emb.shape, len(sequence), file, idx)
        if emb.shape[0] != len(sequence):
            if len(sequence) <= 600:
                sizediff = len(sequence) - emb.shape[0]
                sizediff_list.append(sizediff)
                num_miss_shape += 1
                print(sizediff, emb.shape, file)
        if idx % 1000 == 0:
            print(idx)

print('missing embs:' ,num_miss)
print('num_miss_shape: ', num_miss_shape)