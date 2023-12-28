'''functions used to find and handle redundant alignments'''
from typing import List, Union
import numpy as np
import pandas as pd


def calc_aln_sim(aln_xy: List[np.ndarray]) -> np.ndarray:
	'''
	calculare overlap between local alignment
	Returns:
		np.ndarray (2d) [num_aln, num_aln]
	'''
	num_alns = len(aln_xy)
	aln_similarity = np.zeros((num_alns, num_alns), dtype=np.float32)
	for i, xyi in enumerate(aln_xy):
		xyi_view = xyi[None, :].copy()
		for j, xyj in enumerate(aln_xy):
			if j >= i:
				break
			# if x,y coords are the same
			similarity = (xyi_view == xyj[:, None]).all(-1)
			# if index in one aln appear in other
			similarity = similarity.any(-1).mean()
			aln_similarity[i, j] = similarity
	return aln_similarity


def unique_aln(simgrid: np.ndarray, tolerance: float = 0.8) -> np.ndarray:
	'''
	create mask indicating unique alignment over alignment similarity matrix
	Args:
		simgrid: (np.ndarray) 2D matrix with alignment similarity
		tolerance: (float) similarity cutoff, alignment with similarity lower
		 then `tolerance` will be treated as unique
	Returns:
		mask (np.ndarray)
	'''
	assert isinstance(tolerance, float)
	assert isinstance(simgrid, np.ndarray)

	index_mask = np.zeros(simgrid.shape[0], dtype=bool)
	num_rows = simgrid.shape[0]
	index = list()
	for i in range(num_rows):
		# is unique
		row = simgrid[i, :]
		if not (row > 0).any():
			index_mask[i] = True
			index.append(i)
		# not unique
		else:
			# indices which current index is similar to
			indices = np.flatnonzero(row > tolerance)
			is_unique = True
			# is any of them already in `index`
			for ind in index:
				# not it isnt
				if ind in indices:
					is_unique = False
					break
			if is_unique:
				index_mask[i] = True
				index.append(i)
	return index_mask


def filter_aln(aln_list: List[np.array], tolerance: float = 0.8,
			with_similarity=False) -> np.ndarray:
	'''
	Args:
		aln_list: (list of np.ndarrays) list of alignments
		tolerance: (float)
	Returns:
		mask (np.ndarray[bool])
	'''
	aln_matrix = calc_aln_sim(aln_list)
	mask = unique_aln(aln_matrix, tolerance=tolerance)
	if with_similarity:
		return mask, aln_matrix
	else:
		return mask


def filter_result_dataframe(data: pd.DataFrame,
							column: Union[str, List[str]] = ['score']) -> \
								pd.DataFrame:
	'''
	keep spans with biggest score and len
	and remove heavily overlapping hits
	Args:
		data (pd.DataFrame): columns required (dbid)
	Returns:
		filtred frame sorted by score
	'''
	if isinstance(column, str):
		column = [column]
	if 'dbid' not in data.columns:
		data['dbid'] = 0

	data = data.sort_values(by=['len'], ascending=False)
	indices = data.indices.tolist()
	data['y1'] = [yx[0][0] for yx in indices]
	data['x1'] = [yx[0][1] for yx in indices]
	#data['y2'] = [yx[-1][0] for yx in indices]
	#data['x2'] = [yx[-1][1] for yx in indices]
	#xy1 = np.concatenate((data['x1'].values, data['y1'].values))
	#data['score'] = data['score'].round(3)

	resultsflt = list()
	iterator = data.groupby(['y1', 'x1'])
	for col in column:
		for groupid, group in iterator:
			tmp = group.nlargest(1, [col], keep='first')
			resultsflt.append(tmp)
	resultsflt = pd.concat(resultsflt)
	# drop duplicates sometimes
	resultsflt = resultsflt.drop_duplicates(
		subset=['pathid', 'dbid', 'len', 'score'])
	# filter
	resultsflt = resultsflt.sort_values(by=['score'], ascending=False)
	return resultsflt
