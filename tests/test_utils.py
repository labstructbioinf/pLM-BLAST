import pytest




@pytest.mark.parametrize("size", [10, 50, 100])
def test_remove_duplicates(size):
    '''
    test if function properly filter redundant hits eg. a-b, a-b, a-c func should exclude b-a
    '''

    data = {idx: list(range(size)) for idx in range(size)}
    # func

    # expected number of samples is size*(size - 1)/2
    expected_size = size*(size - 1)/2
    actual_size = sum([len(v) for v in data.values()])

    assert actual_size == expected_size
