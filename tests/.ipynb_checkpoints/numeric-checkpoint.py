'''numerical function tests'''
import os
os.environ['NUMBA_DEBUGINFO'] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '1'
import sys
import time
import faulthandler 
sys.path.append('..')
import pytest
import numpy as np

from alntools.numeric import move_mean

faulthandler.enable()

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
    assert arr.shape[0] == result.shape[0]
    time.sleep(0.1)