'''module merging all extraction steps into user friendly functions'''
import pandas as pd
from typing import List, Tuple
import numpy as np
import torch

from .numeric import embedding_local_similarity
from .alignment import gather_all_paths
from .prepare import search_paths


class Extractor:
    '''
    main class for handling alignment extaction
    '''
    MIN_SPAN_LEN : int = 20
    WINDOW_SIZE: int = 20
    NORM: bool = 'rows'
    LIMIT_RECORDS: int = 20
    BFACTOR: float = 1
    SIGMA_FACTOR: float = 1
    GAP_OPEN: float = 0
    GAP_EXT: float = 0
    def __init__(self, *args, **kw_args):
        pass

    def nested_frame_generator(self, dataframe: pd.DataFrame, embeddings: List[torch.Tensor]) \
        -> Tuple[torch.Tensor, torch.Tensor, pd.Series, pd.Series]:
        '''
        yields (query_embedding, target_embedding, query_row, target_row)
        '''
        for query_idx, query_row in dataframe.iterrows():
            if self.LIMIT_RECORDS <= query_idx: break
            query_embedding = embeddings[query_idx]
            for target_idx, target_row in dataframe.iterrows():
                if query_idx <= target_idx: break
                target_embedding = embeddings[target_idx]
                yield (query_embedding, target_embedding, query_row, target_row)
    
    def embedding_to_span(self, X : np.ndarray, Y : np.ndarray) -> pd.DataFrame:
        '''
        convert embeddings of given X and Y tensors into dataframe
        Returns:
            results: (pd.DataFrame) alignment hits frame
        '''

        if not np.issubdtype(X.dtype, np.float32):
            X = X.astype(np.float32)
        if not np.issubdtype(Y.dtype, np.float32):
            Y = Y.astype(np.float32)
        densitymap = embedding_local_similarity(X, Y)
        paths = gather_all_paths(densitymap,
         norm=self.NORM,
        minlen=self.MIN_SPAN_LEN,
        bfactor=self.BFACTOR,
        gap_opening=self.GAP_OPEN,
        gap_extension=self.GAP_EXT)
        results = search_paths(densitymap, 
            paths=paths,
            window=self.WINDOW_SIZE,
            min_span=self.MIN_SPAN_LEN,
            sigma_factor=self.SIGMA_FACTOR,
            as_df=True)

        return results

    @staticmethod
    def validate_argument(X : np.ndarray) -> bool:
        pass
