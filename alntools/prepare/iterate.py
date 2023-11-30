'''functions to prepare local alignment paths'''
from typing import Union, List, Tuple, Dict

import numpy as np
import pandas as pd

from .. numeric import move_mean, find_alignment_span
from ..settings import AVG_EMBEDDING_STD


def mask_like(densitymap: np.array,
				paths: Union[List[List[int]], List[Tuple[int, int]]]) -> np.ndarray:
	'''
	create densitymap mask for visualization
	Args:
		densitymap (np.ndarray):
		paths (list of paths):
	Returns:
		(np.ndarray) binary mask
	'''
	mask = np.zeros_like(densitymap)
	for path in paths:
		for (y, x) in path:
			assert x >= 0 and y >= 0
			mask[y, x] = 1
	return mask


def search_paths(submatrix: np.ndarray,
		 paths: Tuple[list, list],
		 window: int = 10,
		 min_span: int = 20,
		 sigma_factor: float = 1.0,
		 global_mode: bool = False,
		 as_df: bool = False) -> Union[Dict[str, Dict], pd.DataFrame]:
	'''
	iterate over all paths and search for routes matching alignmnet criteria

	Args:
		submatrix (np.ndarray): density matrix
		paths (list): list of paths to scan
		window (int): size of moving average window
		min_span (int): minimal length of alignment to collect
		sigma_factor (float): standard deviation threshold
		as_df (bool): when True, instead of dictionary dataframe is returned
	Returns:
		(dict): alignment paths
	'''
	assert isinstance(submatrix, np.ndarray)
	assert isinstance(paths, list)
	assert isinstance(window, int) and window > 0
	assert isinstance(min_span, int) and min_span > 0
	assert isinstance(sigma_factor, (int, float))
	assert isinstance(global_mode, bool)
	assert isinstance(as_df, bool)

	min_span = max(min_span, window)
	if not np.issubsctype(submatrix, np.float32):
		submatrix = submatrix.astype(np.float32)

	# force sigma to be not greater then average std of embeddings
	# also not too small
	#arr_sigma = submatrix.std()
	#arr_sigma = max(arr_sigma, AVG_EMBEDDING_STD)
	arr_sigma = AVG_EMBEDDING_STD
	path_threshold = sigma_factor*arr_sigma
	spans_locations = dict()
	# iterate over all paths
	for ipath, path in enumerate(paths):
		# remove one index push
		diag_ind = path - 1
		if diag_ind.size < min_span:
			continue
		# revert indices and and split them into x, y
		y, x = diag_ind[::-1, 0].ravel(), diag_ind[::-1, 1].ravel()
		pathvals = submatrix[y, x].ravel()
		if not global_mode:
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
		if any(spans):
			for idx, (start, stop) in enumerate(spans):
				alnlen = stop - start
				if alnlen < min_span:
					continue
				y1, x1 = y[start:stop-1], x[start:stop-1]
				arr_values = submatrix[y1, x1]
				arr_indices = np.stack([y1, x1], axis=1)
				spans_locations[idx] = {
					'pathid': ipath,
					'spanid': idx,
					'span_start': start,
					'span_end': stop,
					'indices': arr_indices,
					'score': arr_values.mean(),
					"len": alnlen,
					"mode": "global" if global_mode else "local"
				}
	if as_df:
		return pd.DataFrame(spans_locations.values())
	else:
		return spans_locations
