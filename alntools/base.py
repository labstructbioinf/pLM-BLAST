'''module merging all extraction steps into user friendly functions'''
import itertools

import pandas as pd
from typing import List, Tuple, Union
import numpy as np
import torch

from .numeric import embedding_local_similarity, signal_enhancement
from .alignment import gather_all_paths
from .prepare import search_paths
from .postprocess import filter_result_dataframe


class Extractor:
	'''
	main class for handling alignment extaction
	'''
	MIN_SPAN_LEN: int = 20
	WINDOW_SIZE: int = 20
	# NORM rows/cols whould make this method asymmmetric a(x,y) != a(y,x).T
	NORM: Union[bool, str] = True
	BFACTOR: int = 1
	SIGMA_FACTOR: float = 1
	GAP_OPEN: float = 0.0
	GAP_EXT: float = 0.0
	FILTER_RESULTS: bool = False
	enh: bool = False

	# TODO add proper agument handling here
	def __init__(self, enh: bool = False, *args, **kw_args):
		assert isinstance(enh, bool)
		self.enh = enh

	def embedding_to_span(self, X: np.ndarray, Y: np.ndarray, mode : str = 'results' ) -> pd.DataFrame:
		'''
		convert embeddings of given X and Y tensors into dataframe
		Args:
			X: (np.ndarray)
			Y: (np.ndarray)
			mode: (str) if set to `all` densitymap and alignment paths are returned
		Returns:
			results: (pd.DataFrame) alignment hits frame
			densitymap: (np.ndarray)
			paths: (list[np.array])
			scorematrix: (np.ndarray)
		'''
		if not np.issubdtype(X.dtype, np.float32):
			X = X.astype(np.float32)
		if not np.issubdtype(Y.dtype, np.float32):
			Y = Y.astype(np.float32)
		if mode not in {'results', 'all'}:
			raise AttributeError(f'mode must me results or all, but given: {mode}')
		densitymap = embedding_local_similarity(X, Y)
		if self.enh:
			densitymap = signal_enhancement(densitymap)
		paths = gather_all_paths(densitymap,
								 norm=self.NORM,
								 minlen=self.MIN_SPAN_LEN,
								 bfactor=self.BFACTOR,
								 gap_opening=self.GAP_OPEN,
								 gap_extension=self.GAP_EXT,
								 with_scores = True if mode == 'all' else False)
		if mode == 'all':
			scorematrix = paths[1]
			paths = paths[0]
		results = search_paths(densitymap,
							   paths=paths,
							   window=self.WINDOW_SIZE,
							   min_span=self.MIN_SPAN_LEN,
							   sigma_factor=self.SIGMA_FACTOR,
							   mode='global' if isinstance(self.BFACTOR, str) else 'local',
							   as_df=True)
		if mode == 'all':
			return (results, densitymap, paths, scorematrix)
		else:
			return results


	def full_compare(self, emb1: np.ndarray, emb2: np.ndarray,
					 qid: int = 0, dbid: int = 0) -> pd.DataFrame:
		'''
		Args:
			emb1: (np.ndarray) sequence embedding [seqlen x embdim]
			emb2: (np.ndarray) sequence embedding [seqlen x embdim]
			idx: (int) identifier used when multiple function results are concatenated
			file: (str) embedding/sequence source file may be omitted
		Returns:
			data: (pd.DataFrame) frame with alignments and their scores
		'''
		res = self.embedding_to_span(emb1, emb2)
		if len(res) > 0:
			# add referece index to each hit
			res['queryid'] = qid
			res['dbid'] = dbid
			# filter out redundant hits
			if self.FILTER_RESULTS:
				res = filter_result_dataframe(res)
			return res
		return []


	@staticmethod
	def validate_argument(X: np.ndarray) -> bool:
		raise NotImplementedError()

