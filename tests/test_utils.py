import pytest
import sys
import os
import itertools

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from alntools.prepare import reduce_duplicates_query_filedict


@pytest.mark.parametrize("size", [100, 500, 1000])
def test_remove_duplicates(size):
    '''
    test if function properly filter redundant hits eg. a-b, b-a, a-c func should exclude b-a
    '''
    
    filedict = {k: v for k, v in zip(range(size), range(size))}
    data = {idx: filedict.copy() for idx in range(size)}
    # func
    data_red = reduce_duplicates_query_filedict(data)
    # expected number of samples is size*(size - 1)/2 + size
    expected_size = size*(size - 1)/2 + size
    actual_size = sum([len(v) for v in data_red.values()])

    # flatten and check if all samples exists
    data_flatten = set(itertools.chain(*[list(v.values()) for v in data_red.values()]))
    for key in filedict:
        assert key in data_flatten, f"missing {key}"
    # check if any keys exists
    for key in data:
        assert key in data_red
    # check if all keys are not empty
    for key, val in data_red.items():
        assert len(val) > 0, f'on entires for {key}'
    # check total size
    assert actual_size == expected_size


@pytest.mark.parametrize("size", [100, 500, 1001])
@pytest.mark.parametrize("cos_cut", [0.1, 0.5, 0.9])
def test_remove_duplicates_cos_cut(size: int, cos_cut: float):
    '''
    test if function properly filter redundant hits eg. a-b, b-a, a-c func should exclude b-a
    symulate cosine similarity cutoff
    '''
    cos_cut_size = int(cos_cut*size)
    data = dict()
    for idx in range(size):
        qfiles = np.random.choice(range(size), cos_cut_size).tolist()
        data[idx] = { k : k for k in qfiles}
    data_red = reduce_duplicates_query_filedict(data)
    # flatten and check if all samples exists
    data_red_flatten = set(itertools.chain(*[list(v.values()) for v in data_red.values()]))
    data_flatten = set(itertools.chain(*[list(v.values()) for v in data.values()]))
    for key in data_flatten:
        assert key in data_flatten, f"missing {key}"
    for key, val in data_red.items():
        assert len(val) > 0, f'on entires for {key}'