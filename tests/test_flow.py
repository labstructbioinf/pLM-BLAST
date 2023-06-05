'''test full flow'''
import os
#os.environ['NUMBA_DEBUGINFO'] = '1'
#os.environ['NUMBA_DISABLE_JIT'] = '1'
import pytest
import numpy as np
import torch

from alntools.base import Extractor
import alntools as aln

PATH_SYMMETRIC_TEST = 'tests/test_data/asymetric'
ATOL=1e-6
# only zero gap penalty will produce normalized results
@pytest.mark.parametrize("GAP_OPEN", [0])
@pytest.mark.parametrize("GAP_EXT", [0])
@pytest.mark.parametrize("WINDOW_SIZE",  [1, 5, 20])
@pytest.mark.parametrize("BFACTOR", [1, 2, 3])
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
                res['indices_seq1'] = res['indices'].apply(lambda aln : [x[1] for x in aln])
                res['indices_seq2'] = res['indices'].apply(lambda aln : [x[0] for x in aln])
                # check alignment borders
                res_seq1_max = res['indices_seq1'].apply(max).max()
                res_seq1_min = res['indices_seq1'].apply(min).min()
                res_seq2_max = res['indices_seq2'].apply(max).max()
                res_seq2_min = res['indices_seq2'].apply(min).min()

                assert res_seq1_min >= 0 and res_seq1_max < emb1.shape[0], f'seq1 ({emb1.shape[0]}) aln indices exeeds seqlen {res_seq1_min} - {res_seq1_max}'
                assert res_seq2_min >= 0 and res_seq2_max < emb2.shape[0], f'seq2 ({emb2.shape[0]}) aln indices exeeds seqlen {res_seq2_min} - {res_seq2_max}'


@pytest.mark.parametrize("WINDOW_SIZE",  [20, 30])
def test_result_symmetry(WINDOW_SIZE):
    '''
    check if results are symmetric and non empty: results for x,y and y,x should be the same
    '''
    file = PATH_SYMMETRIC_TEST + '.pt'
    if not os.path.isfile(file):
        raise FileNotFoundError(f'missing embedding symmetricity test file: {file}')
    embs = torch.load(file)
    X, Y = embs[0].numpy(), embs[1].numpy()
    module = Extractor()
    module.WINDOW_SIZE = WINDOW_SIZE
    res12, density12, _, scorematrix12 = module.embedding_to_span(Y, X, mode='all')
    res21, density21, _, scorematrix21 = module.embedding_to_span(X, Y, mode='all')
    # draw path masks for both
    if res12.shape[0] != res21.shape[0]:
        raise ValueError('result dataframe is asymmetric')
    if res12.shape[0] == 0 or res21.shape[0] == 0:
        raise ValueError(f'empty result dataframe {res12.shape[0]}, {res21.shape[0]} for window: {WINDOW_SIZE}')
    mask12 = aln.prepare.mask_like(densitymap=density12, paths=res12['indices'])
    mask21 = aln.prepare.mask_like(densitymap=density21, paths=res21['indices'])
    if not np.allclose(density12, density21.T, atol=ATOL):
        max_diff = density12 - density21.T
        raise ValueError(f'density matrix is asymmetric, max diff: {max_diff.max()}')
    if not np.allclose(scorematrix12, scorematrix21.T, atol=ATOL):
        raise ValueError('scorematrix matrix is asymmetric')
    if not np.allclose(mask12, mask21.T, atol=ATOL):
        raise ValueError('mask matrix is asymmetric')
    if res12['score'].max() > 1.01 or res12['score'].max() > 1.01:
        raise ValueError('score is higher then one')

