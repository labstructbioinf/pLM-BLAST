'''functions to prepare local alignment paths'''
from typing import Union, List, Tuple, Dict

import numpy as np
import pandas as pd

from .. numeric import move_mean, find_alignment_span


def mask_like(densitymap: np.array,
                paths: Union[List[List[int]], List[Tuple[int, int]]]) -> np.ndarray:
    '''
    create densitymap mask for visualization
    Args:
        densitymap: (np.ndarray)
        paths: (list of paths)
    Returns:
        mask: (np.ndarray) binary mask
    '''
    mask = np.zeros_like(densitymap)
    for path in paths:
        for (y, x) in path:
            assert x >= 0 and y >= 0
            mask[y, x] = 1
    return mask


def search_paths(score_matrix: np.ndarray,
                paths: Tuple[list, list],
                window: int = 10,
                min_span: int = 10,
                sigma_factor: float = 1.0,
                as_df: bool = False) -> Dict[str, Dict]:
    '''
    iterate over all paths and search for routes matching alignmnet criteria
    Args:
        substitutionmatrix: (np.ndarray) score matrix
        paths: (list) list of paths to scan
        window: (int) size of moving average window
        min_span: (int) minimal length of alignment to collect
        as_df: (bool) when True, instead of dictionary dataframe is returned
        
    Returns:
        record: (dict) alignment paths
    '''
    
    assert isinstance(score_matrix, np.ndarray)
    assert isinstance(paths, list)
    assert isinstance(window, int) and window > 0
    assert isinstance(min_span, int) and min_span > -1
    assert isinstance(sigma_factor, (int, float))

    if not np.issubsctype(score_matrix, np.float32):
        score_matrix = score_matrix.astype(np.float32)
    
    arr_sigma = score_matrix.std()
    path_threshold = sigma_factor*arr_sigma
    spans_locations = dict()
    #iterate over all paths
    for ipath, path in enumerate(paths):
        # remove one index push
        diag_ind = path - 1
        if diag_ind.size < min_span:
            continue
        # revert indices and and split them into x, y
        y, x = diag_ind[::-1, 0].ravel(), diag_ind[::-1, 1].ravel()
        pathvals = score_matrix[y, x].ravel()
        # smooth values
        if window != 1:
            line_mean = move_mean(pathvals, window)
        else:
            line_mean = pathvals
        spans = find_alignment_span(line_mean, mthreshold=path_threshold, minlen=min_span)
        # check if there is non empty alignment
        if any(spans):
            for idx, (start, stop) in enumerate(spans):
                if stop - start < min_span:
                    continue
                y1, x1 = y[start:stop], x[start:stop]
                arr_indices = np.stack([y1, x1], axis=1)
                arr_values = score_matrix[y1, x1]
                keyid = f'{ipath}_{idx}'
                spans_locations[keyid] = {
                    'pathid': ipath,
                    'spanid': idx,
                    'span_start': start,
                    'span_end' : stop,
                    'indices': arr_indices,
                    'score' : arr_values.mean(),
                    "len" : stop - start
                }
    if as_df:
        return pd.DataFrame(spans_locations.values())
    else:
        return spans_locations
 