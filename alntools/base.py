'''module merging all extraction steps into friendly functions'''



import pandas as pd
from typing import List, Tuple
import torch

from .density import embedding_similarity
from .alignment import gather_all_paths
from .prepare import search_paths




class Extractor:
    MIN_SPAN_LEN : int = 20
    WINDOW_SIZE: int = 20
    NORM: bool = True
    LIMIT_RECORDS: int = 20
    def __init__(self):
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
    
    def embedding_to_span(self, X, Y) -> pd.DataFrame:
        densitymap = embedding_similarity(X, Y)
        densitymap = densitymap.cpu().numpy()
        paths = gather_all_paths(densitymap, norm=self.NORM, minlen=self.MIN_SPAN_LEN)
        spans_locations = search_paths(densitymap, paths=paths, window=self.WINDOW_SIZE, min_span=self.MIN_SPAN_LEN)
        return pd.DataFrame(spans_locations.values())
