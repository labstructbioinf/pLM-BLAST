'''test full flow'''
import os
#os.environ['NUMBA_DEBUGINFO'] = '1'
#os.environ['NUMBA_DISABLE_JIT'] = '1'
import pytest
import numpy as np
import torch

from alntools.base import Extractor


# only zero gap penalty will produce normalized results
@pytest.mark.parametrize("GAP_OPEN", [0])
@pytest.mark.parametrize("GAP_EXT", [0])
@pytest.mark.parametrize("WINDOW_SIZE",  [1, 5, 10, 20])
@pytest.mark.parametrize("BFACTOR", [1, 2, 3, 4])
@pytest.mark.parametrize("SIGMA_FACTOR", [1, 1.5, 2, 3])
def test_results(GAP_OPEN, GAP_EXT, WINDOW_SIZE, BFACTOR, SIGMA_FACTOR):

    preffix = 'tests/test_data/embeddings'
    files = os.listdir(preffix)
    # add preffix
    files = [os.path.join(preffix, file) for file in files]
    if len(files) == 0:
        raise FileNotFoundError(f'no test embeddings in {preffix}')
    embedding_list = []
    for file in files:
        tmp = torch.load(file)
        # convert to numpy array
        tmp = tmp.cpu().float().numpy()
        embedding_list.append(tmp)
    module = Extractor()
    module.SIGMA_FACTOR = SIGMA_FACTOR
    module.WINDOW_SIZE = WINDOW_SIZE
    module.GAP_OPEN = GAP_OPEN
    module.GAP_EXT = GAP_EXT
    module.BFACTOR = BFACTOR
    for emb1 in embedding_list:
        for emb2 in embedding_list:
            res = module.embedding_to_span(emb1, emb2)
            if len(res) != 0:
                # check results
                assert res['score'].max() <= 1.01, 'score is higher then one'


