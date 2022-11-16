'''array calculations'''

from typing import Tuple, List, Union

import numpy as np
import numba
from numba import types


@numba.jit(nopython=True, fastmath=True, cache=True)
def max_value_over_line(arr: np.ndarray, ystart: int, ystop: int, xstart: int, xstop: int):
    '''
    fix max value in row or column. When xstart == xstop - max value is calculated over other dimension
    and vice versa
    Args:
        arr: (np.ndarray)
        ystart: (int)
        ystop: (int)
        xstart: (int)
        xstop: (int)
    Returns:
        max_value: (float)
    '''
    
    if xstart == xstop:
        # iterate over array slice
        max_value = arr[ystart, xstart]
        for yidx in range(ystart+1, ystop):
            if max_value > arr[yidx, xstart]:
                max_value = arr[yidx, xstart]
    else:
        max_value = arr[ystart, xstart]
        for xidx in range(xstart+1, xstop):
            if max_value > arr[ystart, xidx]:
                max_value = arr[ystart, xidx]
    return max_value


@numba.jit('f4[:,:](f4[:,:], f4)', nogil=True, nopython=True, fastmath=True, cache=True)
def fill_matrix(a: np.ndarray, gap_penalty: float):
    '''
    fill score matrix
    Params:
        a: (np.array)
        gap_penalty (float)
    Return:
        b: (np.array)
    '''
    nrows : int = a.shape[0] + 1
    ncols : int = a.shape[1] + 1
    H : np.ndarray = np.zeros((nrows, ncols), dtype=np.float32)
    h_tmp : np.ndarray = np.zeros(4, dtype=np.float32)
    for i in range(1, nrows):
        for j in range(1, ncols):
            #gap = abs(i - j)*gap_penalty
            h_tmp[0] = H[i-1,j-1] + a[i-1,j-1]
            #h_tmp[1] = H[i-1, j] - gap_penalty
            #h_tmp[2] = H[i, j-1] - gap_penalty
            # max over first dimension - y
            h_tmp[1] = max_value_over_line(H, 1, i+1, j, j) - gap_penalty
            # max over second dimension - x
            h_tmp[2] = max_value_over_line(H, i, i, 1, j+1) - gap_penalty
            H[i,j] = max(h_tmp)
    return H


def fill_score_matrix(sub_matrix : np.ndarray, gap_penalty : float = 0.0) -> np.ndarray:
    '''
    use substitution matrix to create score matrix
    Params:
        sub_matrix: (np.array) substitution matrix in form of 2d array with shape: [num_res1, num_res2]
        gap_penalty: (float)
    Return:
        score_matrix: (np.array)
    '''
    assert gap_penalty >= 0, 'gap penalty must be positive'
    assert isinstance(gap_penalty, (float, np.float32))
    assert isinstance(sub_matrix, np.ndarray), 'substitution matrix must be numpy array'
    # func fill_matrix require np.float32 array as input
    if not np.issubsctype(sub_matrix, np.float32):
        sub_matrix = sub_matrix.astype(np.float32)
    score_matrix = fill_matrix(sub_matrix, gap_penalty=gap_penalty)
    return score_matrix


@numba.jit('f4[:](f4[:], i4)', nogil=True, nopython=True, fastmath=True, cache=True)
def move_mean(a: np.ndarray, window_width: int):
    '''
    Moving average
    first and last elements are constant
    Args:
        a: (np.ndarray np.float32)
        window_width: (int)
    Returns:
        a_ma: (np.ndarray np.float32)
    '''
    asum :float = 0.0
    count = 0
    mean0 = 0.0
    a_size = a.shape[0]
    out = np.zeros_like(a, dtype=np.float32)
    pad_start = window_width
    pad_end = a_size - window_width
    # clip boundaries
    if pad_start >= a_size:
        pad_start = a_size
    if pad_end < 0:
        pad_end = a_size
    for i in range(0, pad_start):
        asum = asum + a[i]
        count = count + 1
    mean0 = asum / count
    # fill first elements with its mean
    for i in range(pad_start):
        out[i] = mean0
    # fill middle
    for i in range(pad_start, pad_end):
        asum = asum + a[i] - a[i - window_width]
        out[i] = asum / count
    # fill last elements
    for i in range(pad_end, a_size):
        out[i] = asum / count
    return out


@numba.jit('types.Tuple((f4, i4))(f4, f4, f4)',nopython=True, cache=True)
def max_from_3(x: float, y: float, z: float) -> Tuple[float, int]:
    '''
    return value and index of biggest values
    '''
    # 2 idx should be diagonal
    if z >= y and z >= x:
        return z, 2
    if x > y and x > z:
        return x, 0
    else:
        return y, 1

@numba.jit(nopython=True, fastmath=True, cache=True)
def traceback_from_point_opt2(scoremx : np.ndarray,
                            point: Tuple[int, int],
                            gap_opening: float = 0,
                            gap_extension: float = 0,
                            stop_value: float = 1e-3) -> np.ndarray:
    '''
    find optimal route over single path
    Args:
        scoremx: (np.ndarray 2D)
        point: (tuple) y, x coordinates
        gap_penalty: (float) gap opening penalty
        gap_extension: (int) 1 or 2
        stop_value: (float) end of route criteria
    Returns:
        path (np.ndarray 2D) coordinates of path
    '''
    f_right : float = 0.0
    f_left : float = 0.0
    f_diag :int = 0
    fi_max : int = 0
    gap_penalty: float = 0
    # assume that the first move through alignment is diagonal
    fi_argmax : int = 2
    y_size : int = scoremx.shape[0]
    x_size : int = scoremx.shape[1]
    yi : int = point[0]
    xi : int = point[1]
    assert y_size > yi#, f'y size of scorematrix ({y_size}) is lower then point ({yi})'
    assert x_size > xi#, f'x size of scorematrix ({x_size}) is lower then point ({xi})'
    # set starting position
    position = 1
    # maximum size of path
    size = y_size + x_size
    path_arr = np.zeros((size, 2), dtype=np.int32)
    # do not insert starting point
    path_arr[0, 0] = yi
    path_arr[0, 1] = xi
    # iterate until border is hit
    # score matrix have one extra row and column
    while (yi > 1) and  (xi > 1):
        # find previous fi_argmax was diagnal
        if fi_argmax == 2:
            gap_penalty = gap_opening
        # otherwise gap is prolongning
        else:
            gap_penalty = gap_extension
        f_right = scoremx[yi-1, xi] - gap_penalty
        f_left = scoremx[yi, xi-1] - gap_penalty
        f_diag = scoremx[yi-1, xi-1]
        fi_max, fi_argmax = max_from_3(f_right, f_left, f_diag)
        # if maximal value if <= 0 stop loop
        if fi_max < stop_value:
            break
        # add point to path
        else:
            # diagonal move
            if fi_argmax == 2:
                yi_new = yi - 1
                xi_new = xi - 1
            # move left
            elif fi_argmax == 1:
                yi_new = yi
                xi_new = xi - 1
            # move right
            else:
                yi_new = yi - 1
                xi_new = xi
            # store index
            path_arr[position, 0] = yi_new
            path_arr[position, 1] = xi_new
            # set new indices
            yi = yi_new
            xi = xi_new
            position += 1
    # push one index up to remove zero padding effect
    # not done
    path_arr = path_arr[:position, :]
    return path_arr


@numba.jit(nopython=True, cache=True)
def find_alignment_span(means: np.ndarray,
                    minlen: int = 10,
                     mthreshold: float = 0.10) -> List[Tuple[int, int]]:
    '''
    search for points matching `mu` and `sigma` criteria
    Args:
        means: (np.ndarray)
        minlen: (int) minimal allowed length of span
        window (int) window length
    Returns:
        spans: (list of tuples)
    '''
    assert minlen > 0

    num_points : int = means.shape[0]
    alnlen : int = 0
    alnstart: int = 0
    spans : list = []
    # iterate over path scores
    for i in range(num_points): 
        alnlen = alnlen + 1
        # if False break current alignment building
        if means[i] < mthreshold:
            # minimal len criteria is filled save alignment span
            # save without current index
            if minlen < alnlen:
                alnstop = i - 1
                spans.append((alnstart, alnstop))
            # start new alignment, reinitialize params setting start point to the next iteration
            alnstart = i + 1
            alnstop = i + 1
            alnlen = 0
    # handle last iteration
    if alnlen > minlen:
        alnstop = i
        spans.append((alnstart, alnstop))
    return spans
