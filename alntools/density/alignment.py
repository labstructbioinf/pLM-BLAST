from Bio import pairwise2
import time
from functools import partial
from typing import Tuple, List

import numpy as np

#trick to use format_alignment func
con = pairwise2.namedtuple('Alignment', ['seqA', 'seqB', 'score', 'start', 'end'])

def reformat_aln(aln, sliced=None):
    '''
    refactor single alignment tuple to match `format_alignment` func
    sliced (None or list) if list returned aln is subset with residues matching
        eg. sliced = list(range(20)) aln will poses only first 20 aa's
    '''
    if sliced is None:
        seqA = [str(s) for s in aln.seqA] #just cast MyChar's to str
        seqB = [str(s) for s in aln.seqB]
    else:
        assert isinstance(sliced, list)
        seqA = [str(s) for idx, s in enumerate(aln.seqA) if idx in sliced]
        seqB = [str(s) for idx, s in enumerate(aln.seqB) if idx is sliced]
    alignment = con(seqA=seqA, 
                    seqB=seqB,
                   score=aln.score,
                   start=aln.start,
                   end=aln.end)
    
    return alignment

def fill_score_matrix(a: np.array, gap_penalty: float = 0):
    '''
    fill score matrix
    Params:
        a: (np.array)
        gap_penalty (float)
    Return:
        b: (np.array)
    '''
    nrows, ncols = a.shape
    H = np.zeros_like(a)
    for i in range(1, nrows):
        for j in range(1, ncols):
            gap_len = abs(i-j)
            h_tmp  = [
                H[i-1,j-1] + a[i,j],
                H[i:, j].max() - gap_len*gap_penalty,
                H[i,j:].max() - gap_len*gap_penalty
            ]
            H[i,j] = max(h_tmp)
    # add one row and column left and top
    H = np.pad(H, [(1,0) ,(1,0)], mode='constant', constant_values=0)
    return H


def traceback_score(scoremx,
 verbose: bool = False) -> Tuple[list, list]:
    y_size, x_size = scoremx.shape
    ysorted, xsorted = np.unravel_index(np.argsort(scoremx, axis=None),
     shape=(y_size, x_size))
    yi, xi =  ysorted[-1], xsorted[-1]
    path = [(yi, xi)]
    values = [scoremx[yi, xi]]
    fnpad = np.pad(scoremx, [(1,1) ,(1,1)], mode='constant', constant_values=0)
    index = 2
    aln_len = len(path) + 1
    while True:
        # find max
        f_right = fnpad[yi-1, xi]
        f_left = fnpad[yi, xi-1]    
        f_diag = fnpad[yi-1, xi-1]
        f_ind = [
            (yi-1, xi),
            (yi, xi-1),
            (yi-1, xi-1)
        ]
        fi = np.asarray([f_right, f_left, f_diag])
        # diagonal move
        if verbose:
            #print('vals:', fi)
            print(f'best ind/val: {fi.argmax()} {fi.max():.2f}')
            print('curr pos:', (yi, xi))
        # if maximal value if <= 0 stop loop
        if fi.max() < 0.05:
            break
        else:
            index = fi.argmax()
            yi_new, xi_new = f_ind[index]
            path.append((yi_new, xi_new))
            values.append(fi.max())
            # set new indices
            yi, xi = yi_new, xi_new
            aln_len += 1
    return path, values

def align(embedding: np.array, gap_penalty: float = 0.05):

    scores = fill_score_matrix(embedding, gap_penalty=gap_penalty)
    path, path_score = traceback_score(scores)
    return path, path_score

def traceback_from_point(scoremx, point: Tuple[int, int], verbose = False):
    y_size, x_size = scoremx.shape
    yi, xi =  point
    assert y_size > yi
    assert x_size > xi
    path = [(yi, xi)]
    values = [scoremx[yi, xi]]
    fnpad = np.pad(scoremx, [(1,1) ,(1,1)], mode='constant', constant_values=0)
    index = 2
    aln_len = len(path) + 1
    while True:
        # find max
        f_right = fnpad[yi-1, xi]
        f_left = fnpad[yi, xi-1]    
        f_diag = fnpad[yi-1, xi-1]
        f_ind = [
            (yi-1, xi),
            (yi, xi-1),
            (yi-1, xi-1)
        ]
        fi = np.asarray([f_right, f_left, f_diag])
        # diagonal move
        if verbose:
            #print('vals:', fi)
            print(f'best ind/val: {fi.argmax()} {fi.max():.2f}')
            print('curr pos:', (yi, xi))
        # if maximal value if <= 0 stop loop
        if fi.max() < 0.05:
            break
        else:
            index = fi.argmax()
            yi_new, xi_new = f_ind[index]
            path.append((yi_new, xi_new))
            values.append(fi.max())
            # set new indices
            yi, xi = yi_new, xi_new
            aln_len += 1
    # push one index up to remove zero padding effect
    path = [(y-1, x-1) for y,x in path]
    return path, values

def draw_alignment(path: list, seq1: str, seq2: str):
    '''
    draws alignment
        path: (list) result of align
        seq1: (str) residue sequence
        seq2: (str) residue sequence
    '''
    alignment = dict(up=[], relation=[], down=[])
    sc1_prev, sc2_prev = -10, -10
    for sc1, sc2 in path:
        up  = seq1[sc1]
        if sc2 == sc2_prev:
            relation = '.'
        else:
            relation = '|'
        down = seq2[sc2]
        alignment['up'].append(up)
        alignment['relation'].append(relation)
        alignment['down'].append(down)
        sc1_prev = sc1
        sc2_prev = sc2

    string = ' '.join(alignment['up']) + '\n'
    string += ' '.join(alignment['relation']) + '\n'
    string += ' '.join(alignment['down'])
    return string

def get_borderline(a: np.array) -> List[Tuple[int, int]]:
    '''
    extract all possible borderline for given 2D matrix
    '''
    height, width = a.shape
    height -= 1; width -= 1
    bottom_line = a[height, :]
    right_vertical_line = a[:, width]
    bindices = np.flatnonzero(bottom_line > 0)
    rvindices = np.flatnonzero(right_vertical_line > 0)

    indices = [(height, ind) for ind in bindices]
    indices += [(ind, width) for ind in rvindices]
    return indices

