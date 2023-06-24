import os
import sys
import time
import timeit
import faulthandler 
sys.path.append('..')
import numpy as np
import pandas as pd


from alntools.numeric import move_mean
from alntools.numeric import find_validpoints2_opt
from alntools.postprocess import calc_aln_sim


data = pd.read_pickle('raw_data.p')

def test_calc_aln_sim():
	global data
	for g, chunk in data.groupby(['query_pdbchain','target_pdbchain']):
		alnlist = chunk.indices.apply(np.array).tolist()
		indices = calc_aln_sim(alnlist)

import timeit
print(timeit.timeit('test_calc_aln_sim()', globals=globals(), number=3))
