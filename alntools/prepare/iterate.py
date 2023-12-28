'''functions to prepare local alignment paths'''
from typing import Union, List, Tuple, Dict

import numpy as np
import pandas as pd

from .. numeric import move_mean, find_alignment_span
from .. settings import AVG_EMBEDDING_STD


def search_paths(submatrix: np.ndarray,
		 paths: List[np.ndarray],
		 window: int = 10,
		 min_span: int = 20,
		 sigma_factor: float = 1.0,
		 globalmode: bool = False,
		 as_df: bool = False) -> Union[Dict[str, Dict], pd.DataFrame]:
	'''
	iterate over all paths and search for routes matching alignmnet criteria
	Args:
		submatrix: (np.ndarray) density matrix
		paths: (list) list of paths to scan
		window: (int) size of moving average window
		min_span: (int) minimal length of alignment to collect
		sigma_factor: (float) standard deviation threshold
		globalmode: (bool) if True global alignemnt is extrted instead of local
		as_df: (bool) when True, instead of dictionary dataframe is returned
	Returns:
		(dict): alignment paths
	'''
	assert isinstance(submatrix, np.ndarray)
	assert isinstance(paths, list)
	assert isinstance(window, int) and window > 0
	assert isinstance(min_span, int) and min_span > 0
	assert isinstance(sigma_factor, (int, float))
	assert isinstance(globalmode, bool)
	assert isinstance(as_df, bool)

	mode = "global" if globalmode else "local"
	min_span = max(min_span, window)
	if not np.issubsctype(submatrix, np.float32):
		submatrix = submatrix.astype(np.float32)
	# force sigma to be not greater then average std of embeddings
	# also not too small
	path_threshold = sigma_factor*AVG_EMBEDDING_STD
	spans_locations = dict()
	# iterate over all paths
	for ipath, path in enumerate(paths):
		if path.size < min_span:
			continue
		# remove one index push
		path -= 1
		# revert indices and and split them into x, y
		y, x = path[::-1, 0].ravel(), path[::-1, 1].ravel()
		pathvals = submatrix[y, x].ravel()
		if not globalmode:
			# smooth values in local mode
			if window != 1:
				line_mean = move_mean(pathvals, window)
			else:
				line_mean = pathvals
			spans = find_alignment_span(means=line_mean,
										mthreshold=path_threshold,
										minlen=min_span)
		else:
			spans = [(0, len(path))]
		# check if there is non empty alignment
		if len(spans) > 0:
			for idx, (start, stop) in enumerate(spans):
				alnlen = stop - start
				# to short alignment
				if alnlen < min_span:
					continue
				if globalmode:
					y1, x1 = y[start:stop], x[start:stop]
				else:
					y1, x1 = y[start:stop-1], x[start:stop-1]
				ylen = y1[-1] - y1[0]
				# to short alignment
				if ylen < min_span:
					continue
				arr_values = submatrix[y1, x1]
				arr_indices = np.stack([y1, x1], axis=1)
				keyid = f'{ipath}_{idx}'
				spans_locations[keyid] = {
					'pathid': ipath,
					'spanid': idx,
					'span_start': start,
					'span_end': stop,
					'indices': arr_indices,
					'score': arr_values.mean(),
					"len": alnlen,
					"mode": mode
				}
	if as_df:
		return pd.DataFrame(spans_locations.values())
	else:
		return spans_locations
