'''module merging all extraction steps into user friendly functions'''
import itertools

import pandas as pd
from typing import List, Tuple, Union
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
    MIN_SPAN_LEN: int = 20
    WINDOW_SIZE: int = 20
    NORM: bool = 'rows'
    LIMIT_RECORDS: int = 20
    BFACTOR: float = 1
    SIGMA_FACTOR: float = 1
    GAP_OPEN: float = 0.0
    GAP_EXT: float = 0.0
    FILTER_RESULTS: bool = False

    def __init__(self, *args, **kw_args):
        pass

    def nested_frame_generator(self, dataframe: pd.DataFrame,
                               embeddings: List[torch.Tensor]) -> \
                                Tuple[torch.Tensor, pd.Series]:
        '''
        yields (query_embedding, target_embedding, query_row, target_row)
        '''
        for query_idx, query_row in dataframe.iterrows():
            if self.LIMIT_RECORDS <= query_idx:
                break
            query_embedding = embeddings[query_idx]
            for target_idx, target_row in dataframe.iterrows():
                if query_idx <= target_idx:
                    break
                target_embedding = embeddings[target_idx]
                yield (
                    query_embedding, target_embedding, query_row, target_row)

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
                               as_df=True)
        if mode == 'all':
            return (results, densitymap, paths, scorematrix)
        else:
            return results

    def full_compare(self, emb1: np.ndarray, emb2: np.ndarray,
                     idx: int, file: str) -> pd.DataFrame:
        res = self.embedding_to_span(emb1, emb2)
        if len(res) > 0:
            # add referece index to each hit
            res['i'] = idx
            res['dbfile'] = file
            # filter out redundant hits
            if self.FILTER_RESULTS:
                res = filter_result_dataframe(res)
            return res
        return []

    @staticmethod
    def validate_argument(X: np.ndarray) -> bool:
        raise NotImplementedError()


class BatchIterator:
    '''
    batch iterator for multiprocessing
    '''
    batchsize: int
    iterlen: int
    max_batchsize: int = 300
    iter: int = 0

    def __init__(self, filedict: dict, batch_size: int):
        self.num_record = len(filedict)
        self.batchsize = min(self.max_batchsize, batch_size, self.num_record)
        self.filedict = filedict
        self.iterlen = int(max(np.ceil(self.num_record/batch_size), 1))
        self.filedict_items = self.filedict.items()

    def __len__(self):
        return self.iterlen

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter >= self.iterlen:
            # for multiple uses of single iterator
            self.iter = 0
            raise StopIteration
        bstart = self.iter * self.batchsize
        bend = bstart + self.batchsize
        self.iter += 1
        # clip batch end
        bend = min(bend, self.num_record)
        # create slice object
        batchslice = slice(bstart, bend, 1)
        batchdata = itertools.islice(self.filedict_items, bstart, bend)
        return batchdata, batchslice
