'''utilities to compare calculated alignment with ground truth'''
from typing import List, Union, Tuple
import numpy as np


def measure_aln_overlap_with_pdblist(seq1_true: Union[np.array, List[int]],
									 seq2_true: Union[np.array, List[int]],
									 alignment: Union[np.array, List[int]]) -> dict:
	'''
	measure cover of given pdb indices (seq1_true and seq2_true)
	with alignment extracted by plm-SEARCH
	Args:
		seq1_true: (list)
		seq2_true: (list)
		alignment: (list)
	Returns:
		record: (dict)
	'''
	if isinstance(seq1_true, (list, tuple)):
		seq1_true = np.array(seq1_true, dtype=np.int32)

	if isinstance(seq2_true, (list, tuple)):
		seq2_true = np.array(seq2_true, dtype=np.int32)

	# unpack alignment to seq1 seq2 pdb indices
	ycoords, xcoords = map(np.asarray, zip(*alignment))

	# create matrix
	seq1_core = seq1_true[np.newaxis, :] == xcoords[:, np.newaxis]
	seq2_core = seq2_true[np.newaxis, :] == ycoords[:, np.newaxis]

	seq1_cover, seq1_core_cover = mean_over_trace(seq1_core)
	seq2_cover, seq2_core_cover = mean_over_trace(seq2_core)
	seq1_aln_in_core = seq1_cover/seq1_true.size
	seq2_aln_in_core = seq2_cover/seq2_true.size

	seq1_core_cover = seq1_core_cover/ycoords.size
	seq2_core_cover = seq2_core_cover/xcoords.size

	record = {
		'seq1_aln_in_core': seq1_aln_in_core,
		'seq2_aln_in_core': seq2_aln_in_core,
		'seq1_core_cover': seq1_core_cover,
		'seq2_core_cover': seq2_core_cover
	}
	return record


def mean_over_trace(arr: np.ndarray) -> Tuple[float, float]:
	offset_down, offset_up = arr.shape
	offset_down_max, offset_up_max = 0, 0
	iterator = np.arange(-offset_down, offset_up, 1)
	trace_stack = np.zeros(iterator.size, dtype=np.float32)
	for i, offset in enumerate(iterator):
		trace = np.trace(arr, offset=offset).sum()
		trace_stack[i] = trace

	offset_down_max = trace_stack.max()
	offset_up_max = offset_down_max
	return offset_down_max, offset_up_max
