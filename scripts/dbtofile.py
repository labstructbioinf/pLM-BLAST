import os
import sys
import argparse

import torch
import pandas as pd
from tqdm import tqdm

from ..alntools.filehandle import DataObject
from ..alntools.density.local import calculate_pool_embs
from ..alntools.density.parallel import load_embeddings_parallel_generator

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


dbdata = DataObject.from_dir(dbpath, "db")
db_embs = []
for embs in load_embeddings_parallel_generator(dbdata.embeddingpath,
                                                num_records=dbdata.size,
                                                num_workers=2):
    db_embs.extend(calculate_pool_embs(embs))
    # try to write emb.64 file
try:
    torch.save(db_embs, dbpath)
except Exception as e:
    print(f'cannot write {dbpath} due to: {e}')

outfile = os.path.join(dbpath, f'emb.64')
torch.save(db_embs, outfile)
print(f'Done! {outfile} created')

