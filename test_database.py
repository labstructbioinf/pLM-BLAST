import time
import os
import torch
import pytest

from alntools.density import load_and_score_database
from alntools.density import load_full_embeddings


pathdb = "/home/nfs/kkaminski/PLMBLST/ecod70db_20220902"
path_query_emb = "/home/nfs/kkaminski/PLMBLST/ecod70db_20220902/1.emb"

timestart = time.perf_counter()
def test_database_scan():

	query_emb = torch.load(path_query_emb)
	files = load_and_score_database(query_emb, pathdb, quantile=0.8)
	print('resulting embeddings', len(files))

	embs = load_full_embeddings(files)
	print(len(embs))

test_database_scan()
timestop = time.perf_counter() - timestart
print(f'{timestop:.2f}')
