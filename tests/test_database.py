import os
import torch
import pytest

from alntools.density import load_and_score_database



pathdb = "/home/nfs/kkaminski/PLMBLST/ecod70db_20220902"
path_query_emb = "/home/nfs/kkaminski/PLMBLST/ecod70db_20220902/1.emb"
def test_database_scan(path, query):

    query_emb = torch.load(path_query_emb)
    embeddings = load_and_score_database(query_emb, path_query_emb, threshold=0.2)
    print('resulting embeddings', len(embeddings))
