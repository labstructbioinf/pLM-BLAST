import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from alntools.density.local import reduce_duplicates_query_filedict


@pytest.mark.parametrize("size", [10, 50, 100])
def test_remove_duplicates(size):
    '''
    test if function properly filter redundant hits eg. a-b, b-a, a-c func should exclude b-a
    '''
    
    filedict = {k: v for k, v in zip(range(size), range(size))}
    data = {idx: filedict.copy() for idx in range(size)}
    # func
    reduce_duplicates_query_filedict(data)
    # expected number of samples is size*(size - 1)/2 + size
    expected_size = size*(size - 1)/2 + size
    actual_size = sum([len(v) for v in data.values()])

    assert actual_size == expected_size
