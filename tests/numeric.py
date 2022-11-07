'''numerical function tests'''
import os
from xml.etree.ElementTree import TreeBuilder
os.environ['NUMBA_DEBUGINFO'] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '1'
import sys
import time
import faulthandler 
sys.path.append('..')
import pytest
import numpy as np
import torch

from alntools.numeric import move_mean
from alntools.numeric import find_alignment_span
from alntools.numeric import fill_score_matrix
from alntools.alignment import border_argmaxpool

faulthandler.enable()

densitymap_test = torch.load('test_data/densitymap_example.pt')

noise_low = -0.1
noise_high = 0.1


@pytest.mark.parametrize("arr", [
    np.random.rand(10),
    np.random.rand(15),
    np.random.rand(20),
    np.random.rand(100),
    np.random.rand(1000),
    np.random.rand(2999)])
@pytest.mark.parametrize("window", [
    1,
    5,
    10,
    11,
    13,
    20])
def test_move_mean(arr, window):
    
    result = move_mean(arr, window)
    assert arr.shape[0] == result.shape[0], 'invalid shape'
    assert not np.isnan(result).any(), 'nan values'


@pytest.mark.parametrize("arr", [
    np.random.uniform(noise_low, noise_high, 35),
    np.random.uniform(noise_low, noise_high, 50),
    np.random.uniform(noise_low, noise_high, 100),
    np.random.uniform(noise_low, noise_high, 200)
])
@pytest.mark.parametrize("spans", [
    [(10, 30), (50, 80)],
    [(0, 20), (25, 100)]
])
def test_path_validpoints(arr, spans):
    # apply spans to input array
    # filtering out of range 
    spans_in_range = []
    for sp1, sp2 in spans:
        if sp1 < arr.size and sp2 < arr.size:
            arr[sp1:sp2+1] += 1
            spans_in_range.append((sp1, sp2))
    spans_results = find_alignment_span(arr, mthreshold=0.2)
    # if no span was detected
    assert len(spans_results) > 0, f'no spans found for arr {arr.size} with {spans_in_range} and {spans_results}'
    # if detected they should match `spans` param
    for sp1_pred, sp2_pred in spans_in_range:
        path_found = False
        for sp1, sp2 in spans:
            if sp1 == sp1_pred and sp2 == sp2_pred:
                path_found = True
        if not path_found:
            raise AssertionError(f'missing results for arr {arr.size} with {spans_in_range} and {spans_results}') 
    

def test_fill_score_matrix():

    # convert to numpy
    densitymap = densitymap_test['densitymap']

    score_matrix = fill_score_matrix(densitymap)
    assert not np.isnan(score_matrix).any(), 'nan values in score_matrix'

    #score_matrix_with_penalty = fill_score_matrix(densitymap, gap_penalty=0.1)
    #assert not np.isnan(score_matrix_with_penalty).any(), 'nan values in score_matrix'
    #assert not (score_matrix == score_matrix_with_penalty).all(), 'penalty score dont affect score_matrix'



@pytest.mark.parametrize("arr", [np.random.rand(s1, s2) for s1, s2 in [(25, 25), (25, 20), (30, 50), (100, 50)]])
@pytest.mark.parametrize("cutoff", [0, 1, 5, 10])
@pytest.mark.parametrize("factor", [1, 2, 3])
def test_borderline_extraction(arr, cutoff, factor):
    borders = border_argmaxpool(array=arr, cutoff=cutoff, factor=factor)
    if factor == 1:
        assert borders.shape[0] == (arr.shape[0] + arr.shape[1] - 2*cutoff - 1), 'border size mismatch'
        # diagonal location should be always here
        bottom_right_diag = np.array([[arr.shape[0], arr.shape[1]]]) - 1
        assert (borders == bottom_right_diag).all(1).any(), 'missing last diagnal index'



