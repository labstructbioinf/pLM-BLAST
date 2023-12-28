
from typing import (Tuple,
					List,
					Union,
					Optional)

import numpy as np
import torch
import pandas as pd

from .numeric import fill_score_matrix, traceback_from_point_opt2


ACIDS_ORDER = 'ARNDCQEGHILKMFPSTWYVX'
ACID_DICT = {r: i for i,r in enumerate(ACIDS_ORDER)}
ACID_DICT['-'] = ACID_DICT['X']
ACID_DICT[' '] = ACID_DICT['X']

HTML_HEADER = """
<html>
<head>
<style>
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
th, td {
  padding: 1px;
}
</style>
</head>"""


def sequence_to_number(seq: List[str]):
	encoded = [ACID_DICT[res] for res in seq]
	return torch.LongTensor(encoded)


def list_to_html_row(data: List[str]) -> str:
	output = ""
	for letter in data:
		output += f"<td>{letter}</td>"
	return output


def mask_like(paths: Union[List[List[int]], List[Tuple[int, int]]],
			   densitymap: Optional[np.ndarray] = None) -> np.ndarray:
	'''
	create densitymap mask for visualization
	Args:
		paths: (list of paths)
		densitymap: (np.ndarray) plot background if not supplied it will be automatically created based on paths
	Returns:
		mask: (np.ndarray) binary mask
	'''
	if densitymap is None:
		maxseq1 = max([coords[-1][0] for coords in paths]) + 1
		maxseq2 = max([coords[-1][1] for coords in paths]) + 1
		mask = np.zeros((maxseq1, maxseq2), dtype=np.int32)
	else:
		mask = np.zeros_like(densitymap)
	for path in paths:
		for (y, x) in path:
			assert x >= 0 and y >= 0
			mask[y, x] = 1
	return mask


def draw_alignment(coords: List[Tuple[int, int]], seq1: str, seq2: str, output: Union[None, str]) -> str:
	'''
	draws alignment based on input coordinates
	Args:
		coords: (list) result of align list of tuple indices
		seq1: (str) full residue sequence 
		seq2: (str) full residue sequence
		output: (str or bool) if None output is printed
	'''

	assert isinstance(seq1, str) or isinstance(seq1[0], str), 'seq1 must be sting like type'
	assert isinstance(seq2, str)or isinstance(seq1[0], str), 'seq2 must be string like type'
	assert len(seq1) > 1 and len(seq2), 'seq1 or seq1 is too short'

	# check whether alignment indices exeed sequence len
	last_position = coords[-1]
	lp1, lp2 = last_position[0], last_position[1]
	if lp1 >= len(seq1):
		raise KeyError(f'mismatch between seq1 length and coords {lp1} - {len(seq1)} for seq2 {lp2} - {len(seq2)}')
	if lp2 >= len(seq2):
		raise KeyError(f'mismatch between seq1 length and coords {lp2} - {len(seq2)}')

	if output != 'html':
		newline_symbol = "\n"
	else:
		newline_symbol = "<br>"
	# container
	alignment = dict(up=[], relation=[], down=[])
	c1_prev, c2_prev = -1, -1
	
	for c1, c2 in coords:
		# check if gap occur
		up_increment   = True if c1 != c1_prev else False
		down_increment = True if c2 != c2_prev else False
		
		if up_increment:
			up = seq1[c1]
		else:
			up = '-'

		if down_increment:
			down = seq2[c2]
		else:
			down = '-'

		if up_increment and down_increment:
			relation = '|'
		else:
			relation = ' '
			
		alignment['up'].append(up)
		alignment['relation'].append(relation)
		alignment['down'].append(down)
			
		c1_prev = c1
		c2_prev = c2
	# merge into 3 line string
	if output != 'html':
		string = ''.join(alignment['up']) + '\n'
		string += ''.join(alignment['relation']) + '\n'
		string += ''.join(alignment['down'])
		if output is not None:
			return string
		else:
			print(string)
	
	# format as html table
	if output == "html":
		html_string = HTML_HEADER + '<body>\n<table>\n'
		html_string +=  "<tr>" + list_to_html_row(alignment['up']) + "</tr>\n"
		html_string += "<tr>" + list_to_html_row(alignment['relation']) + "</tr>\n"
		html_string += "<tr>" + list_to_html_row(alignment['down']) + "</tr>\n"
		html_string += "</table>\n<body>\n"
		return html_string


def get_borderline(a: np.array, cutoff_h: int = 10, cutoff_w: int = 10) -> np.ndarray:
	'''
	extract all possible border indices (down, right) for given 2D matrix
	for example: \n
		A A A A A X\n
		A A A A A X\n
		A A A A A X\n
		A A A A A X\n
		A A A A A X\n
		X X X X X X\n
	\n
	result will contain indices of `X` values starting from upper right to lower left
	Args:
		a (np.ndarray):
		cutoff_h (int): control how far stay from edges - the nearer the edge the shorter diagonal for first dimension
		cutoff_w (int): control how far stay from edges - the nearer the edge the shorter diagonal for second dimension
	Returns:
		np.ndarray: border coordinates with shape of [len, 2] 
	'''
	# width aka bottom
	height, width = a.shape
	height -= 1; width -= 1
	# clip values		

	if height < cutoff_h:
		hstart = 0
	else:
		hstart = cutoff_h

	if width < cutoff_w:
		bstart = 0
	else:
		bstart = cutoff_w
	# arange with add syntetic dimension
	# height + 1 is here for diagonal
	hindices = np.arange(hstart, height+1)[:, None]
	# add new axis
	hindices = np.repeat(hindices, 2, axis=1)
	hindices[:, 1] = width

	# same operations for bottom line
	# but in reverted order
	bindices = np.arange(bstart, width)[::-1, None]
	# add new axis
	bindices = np.repeat(bindices, 2, axis=1)
	bindices[:, 0] = height
	
	borderline = np.vstack((hindices, bindices))
	return borderline


def border_argmaxpool(array: np.ndarray,
					cutoff: int = 10,
					factor: int = 2) -> np.ndarray:
	"""
	Get border indices of an array satysfing cutoff and factor conditions.

	Args:
		array (np.ndarray): embedding-based scoring matrix.
		cutoff (int): parameter to control border cutoff.
		factor (int): stride-like control of indices returned similar to path[::factor].

	Returns:
		(np.ndarray) path indices

	"""
	assert factor >= 1
	assert cutoff >= 0
	assert isinstance(factor, int)
	assert isinstance(cutoff, int)
	assert array.ndim == 2
	# case when short embeddings are given
	cutoff_h = cutoff if cutoff < array.shape[0] else 0
	cutoffh_w = cutoff if cutoff < array.shape[1] else 0
		
	boderindices = get_borderline(array, cutoff_h=cutoff_h, cutoff_w=cutoffh_w)
	if factor > 1:
		y, x = boderindices[:, 0], boderindices[:, 1]
		bordevals = array[y, x]
		num_values = bordevals.shape[0]	
		# make num_values divisible by `factor` 
		num_values = (num_values - (num_values % factor))
		# arange shape (num_values//factor, factor)
		# argmax over 1 axis is desired index over pool 
		arange2d = np.arange(0, num_values).reshape(-1, factor)
		arange2d_idx = np.arange(0, num_values, factor, dtype=np.int32)
		borderargmax = bordevals[arange2d].argmax(1)
		# add push factor so values  in range (0, factor) are translated
		# into (0, num_values)
		borderargmax += arange2d_idx
		return boderindices[borderargmax, :]
	else:
		return boderindices


def border_argmaxlenpool(array: np.ndarray,
					cutoff: int = 10,
					factor: int = 2) -> np.ndarray:
	'''
	get border indices of an array satysfing
	Args:
		array: (np.ndarray)
		cutoff: (int)
		factor: (int)
	Returns:
		borderindices: (np.ndarray)
	'''
	assert factor > 0
	assert cutoff >= 0
	assert isinstance(factor, int)
	assert cutoff*2 < (array.shape[0] + array.shape[1]), 'cutoff exeed array size'

	boderindices = get_borderline(array, cutoff=cutoff)
	if factor > 1:
		y, x = boderindices[:, 0], boderindices[:, 1]
		bordevals = array[y, x]
		num_values = bordevals.shape[0]	
		# make num_values divisible by `factor` 
		num_values = (num_values - (num_values % factor))
		# arange shape (num_values//factor, factor)
		# argmax over 1 axis is desired index over pool 
		arange2d = np.arange(0, num_values).reshape(-1, factor)
		arange2d_idx = np.arange(0, num_values, factor, dtype=np.int32)
		borderargmax = bordevals[arange2d].argmax(1)
		# add push factor so values  in range (0, factor) are translated
		# into (0, num_values)
		borderargmax += arange2d_idx
		return boderindices[borderargmax, :]
	else:
		return boderindices


def gather_all_paths(array: np.ndarray,
					minlen: int = 10,
					norm: bool = True,
					bfactor: Union[int, str] = 1,
					gap_penalty: float = 0,
					with_scores: bool = False) -> List[np.ndarray]:
	'''
	calculate scoring matrix from input substitution matrix `array`
	find all Smith-Waterman-like paths from bottom and right edges of scoring matrix
	Args:
		array (np.ndarray): raw subtitution matrix aka densitymap
		norm_rows (bool, str): whether to normalize array per row or per array
		bfactor (int): use argmax pooling when extracting borders, bigger values will improve performence but may lower accuracy
		gap_penalty: (float) default to zero
		with_scores (bool): if True return score matrix
	Returns:
		list: list of all valid paths through scoring matrix
		np.ndarray: scoring matrix used
	'''
	
	if not isinstance(array, np.ndarray):
		array = array.numpy().astype(np.float32)
	if not isinstance(norm, bool):
		raise ValueError(f'norm_rows arg should be bool type, but given: {norm}')
	if not isinstance(bfactor, (str, int)):
		raise TypeError(f'bfactor should be int/str but given: {type(bfactor)}')
	# standarize embedding
	if norm:
		array = (array - array.mean())/(array.std() + 1e-3)
	# set local or global alignment mode
	globalmode = True if bfactor == "global" else False
	# get all edge indices for left and bottom
	# score_matrix shape = array.shape + 1
	score_matrix = fill_score_matrix(array, gap_penalty=gap_penalty, globalmode=globalmode)
	# local alignment mode
	if isinstance(bfactor, int):
		indices = border_argmaxpool(score_matrix, cutoff=minlen, factor=bfactor)
	# global alignment mode
	elif globalmode:
		indices = [(array.shape[0], array.shape[1])]
	paths = list()
	for ind in indices:
		path = traceback_from_point_opt2(score_matrix, ind, gap_opening=0)
		paths.append(path)
	if with_scores:
		return (paths, score_matrix)
	else:
		return paths
