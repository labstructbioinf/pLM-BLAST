'''module merging all extraction steps into user friendly functions'''
import itertools

import pandas as pd
from typing import Union, Optional
import numpy as np
import torch

from .numeric import embedding_local_similarity
from .alignment import gather_all_paths
from .prepare import search_paths
from .postprocess import filter_result_dataframe


class Extractor:
	'''
	main class for handling alignment extaction
	'''
	min_spanlen: int = 20
	window_size: int = 20
	# NORM rows/cols whould make this method asymmmetric a(x,y) != a(y,x).T
	NORM: Union[bool, str] = True
	bfactor: int = 1
	sigma_factor: float = 1
	GAP_OPEN: float = 0.0
	GAP_EXT: float = 0.0
	filter_results: bool = False
	mode: str = 'local'


	def __init__(self,
			   	min_spanlen: int = 20,
			    window_size: int = 20,
				sigma_factor: float = 2,
				filter_results: bool = False,
				bfactor: int = 1,
				**kw_args):
		'''
		pLM-BLAST module
		Args:
			min_spanlen (int): minimal lenght of requested alignment
			window_size (int): size of averaging window, used when extracting alignments
			sigma_factor (float): 
			filter_results (bool): whether to remove redundant hits from results
			bfactor (str, int): if int it refers to density of alignment searching (1 is maximal density),
			 	increasing this parameter will reduce the number of overlaping alignments in results
				  and will increase searching speed. When bfactor == global, will perform global alignment search
		'''
		assert isinstance(min_spanlen, int)
		assert min_spanlen > 0
		assert isinstance(window_size, int)
		assert window_size > 0
		assert isinstance(sigma_factor, (int, float))
		assert sigma_factor > 0
		assert isinstance(filter_results, bool)
		assert isinstance(bfactor, (str, int))
		if isinstance(bfactor, int):
			assert 1 <= bfactor <= 10
		if isinstance(bfactor, str):
			assert bfactor == 'global'


		self.min_spanlen = min_spanlen
		self.window_size = window_size
		self.sigma_factor = sigma_factor
		self.filter_results = filter_results
		self.bfactor = bfactor
		self.mode = 'global' if self.bfactor == 'global' else 'local'

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
		paths = gather_all_paths(densitymap,
								 norm=self.NORM,
								 minlen=self.min_spanlen,
								 bfactor=self.bfactor,
								 gap_opening=self.GAP_OPEN,
								 gap_extension=self.GAP_EXT,
								 with_scores = True if mode == 'all' else False)
		if mode == 'all':
			scorematrix = paths[1]
			paths = paths[0]
		results = search_paths(densitymap,
							   paths=paths,
							   window=self.window_size,
							   min_span=self.min_spanlen,
							   sigma_factor=self.sigma_factor,
							   mode=self.mode,
							   as_df=True)
		if mode == 'all':
			return (results, densitymap, paths, scorematrix)
		else:
			return results


	def full_compare(self, emb1: np.ndarray, emb2: np.ndarray,
					 dbid: int = 0, file: Optional[str] = '') -> Optional[pd.DataFrame]:
		'''
		perform comparison of two sequence embeddings

		Args:
			emb1 (np.ndarray): sequence embedding [seqlen, embdim]
			emb2 (np.ndarray): sequence embedding [seqlen, embdim]
			dbid (int): typically database protein index, identifier used when multiple function results are concatenated
			file (str): embedding/sequence source file may be omitted
		Returns:
			data: (pd.DataFrame) frame with alignments and their scores
		'''
		res = self.embedding_to_span(emb1, emb2)
		if len(res) > 0:
			# add referece index to each hit
			res['dbid'] = dbid
			res['dbfile'] = file
			# filter out redundant hits
			if self.filter_results and self.bfactor != 'global':
				res = filter_result_dataframe(res)
			return res


	@staticmethod
	def validate_argument(X: np.ndarray) -> bool:
		raise NotImplementedError()

