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


class PlmBlastParamError(Exception):
	pass


class Extractor:
	'''
	main class for handling alignment extaction
	'''
	MIN_SPAN_LEN: int = 20
	WINDOW_SIZE: int = 20
	FILTER_RESULTS: bool = False
	# NORM rows/cols whould make this method asymmmetric a(x,y) != a(y,x).T
	norm: bool = False
	enh: bool = False
	globalmode: bool = False
	sigma_factor: Union[int, float] = 2
	bfactor: int = 2

	# TODO add proper agument handling here
	def __init__(self, enh: bool = False,
			   norm: bool = False,
				bfactor: Union[str, bool] = 2,
				sigma_factor: Union[int, float] = 2,
				gap_penalty: float = 0.0):
		
		# validate arguments
		assert isinstance(enh, bool)
		assert isinstance(norm, bool)
		if isinstance(bfactor, str):
			if bfactor != "global":
				raise PlmBlastParamError(f'invalid bfactor value: {bfactor}')
		elif isinstance(bfactor, int):
			if bfactor <= 0:
				raise PlmBlastParamError(f'invalid bfactor value: {bfactor} should be > 0 or str: global')
		else:
			raise PlmBlastParamError(f'invalid bfactor type: {type(bfactor)}')
		if not isinstance(sigma_factor, (float, int)) or sigma_factor <= 0:
			raise PlmBlastParamError(f'sigma factor must be positive valued number, not: {type(sigma_factor)} with value: {sigma_factor}')

		self.enh = enh
		self.norm = norm
		self.globalmode = True if bfactor == 'global' else False
		self.bfactor = bfactor
		self.sigma_factor = sigma_factor
		self.gap_penalty = gap_penalty


	def embedding_to_span(self, X: np.ndarray, Y: np.ndarray, mode: str = 'results' ) -> pd.DataFrame:
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
		if not isinstance(mode, str) or mode not in {'results', 'all'}:
			raise PlmBlastParamError(f'mode must me results or all, but given: {mode}')
		densitymap = embedding_local_similarity(X, Y)
		if self.enh:
			densitymap = signal_enhancement(densitymap)
		paths = gather_all_paths(densitymap,
								 norm=self.norm,
								 minlen=self.MIN_SPAN_LEN,
								 bfactor=self.bfactor,
								 gap_penalty=self.gap_penalty,
								 with_scores = True if mode == 'all' else False)
		if mode == 'all':
			scorematrix = paths[1]
			paths = paths[0]
		results = search_paths(densitymap,
							   paths=paths,
							   window=self.WINDOW_SIZE,
							   min_span=self.MIN_SPAN_LEN,
							   sigma_factor=self.sigma_factor,
							   globalmode=self.globalmode,
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
		return None


	@staticmethod
	def validate_argument(X: np.ndarray) -> bool:
		raise NotImplementedError()

