
from typing import Tuple, List, Union

import numpy as np
import torch
import pandas as pd

from .numeric import fill_score_matrix, traceback_from_point_opt2


ACIDS_ORDER = 'ARNDCQEGHILKMFPSTWYVX'
ACID_DICT = {r: i for i,r in enumerate(ACIDS_ORDER)}
ACID_DICT['-'] = ACID_DICT['X']
ACID_DICT[' '] = ACID_DICT['X']

HTML_HEADER = """
<html>
<head>
<style>
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
th, td {
  padding: 1px;
}
</style>
</head>"""

def sequence_to_number(seq: List[str]):
    encoded = [ACID_DICT[res] for res in seq]
    return torch.LongTensor(encoded)

def list_to_html_row(data: List[str]) -> str:

    output = ""
    for letter in data:
        output += f"<td>{letter}</td>"
    return output

def draw_alignment(coords: List[Tuple[int, int]], seq1: str, seq2: str, output: Union[None, str]) -> str:
    '''
    draws alignment based on input coordinates
    Args:
        coords: (list) result of align list of tuple indices
        seq1: (str) full residue sequence 
        seq2: (str) full residue sequence
        output: (str or bool) if None output is printed
    '''
    
    assert isinstance(seq1, str) or isinstance(seq1[0], str), 'seq1 must be sting like type'
    assert isinstance(seq2, str)or isinstance(seq1[0], str), 'seq2 must be string like type'
    assert len(seq1) > 1 and len(seq2), 'seq1 or seq1 is too short'

    # check whether alignment indices exeed sequence len
    last_position = coords[-1]
    lp1, lp2 = last_position[0], last_position[1]
    if lp1 >= len(seq1):
        raise KeyError(f'mismatch between seq1 length and coords {lp1} - {len(seq1)} for seq2 {lp2} - {len(seq2)}')
    if lp2 >= len(seq2):
        raise KeyError(f'mismatch between seq1 length and coords {lp2} - {len(seq2)}')

    if output != 'html':
        newline_symbol = "\n"
    else:
        newline_symbol = "<br>"
    # container
    alignment = dict(up=[], relation=[], down=[])
    c1_prev, c2_prev = -1, -1
    
    for c1, c2 in coords:
        # check if gap occur
        up_increment   = True if c1 != c1_prev else False
        down_increment = True if c2 != c2_prev else False
        
        if up_increment:
            up = seq1[c1]
        else:
            up = '-'

        if down_increment:
            down = seq2[c2]
        else:
            down = '-'

        if up_increment and down_increment:
            relation = '|'
        else:
            relation = ' '
            
        alignment['up'].append(up)
        alignment['relation'].append(relation)
        alignment['down'].append(down)
            
        c1_prev = c1
        c2_prev = c2
    # merge into 3 line string
    if output != 'html':
        string = ''.join(alignment['up']) + '\n'
        string += ''.join(alignment['relation']) + '\n'
        string += ''.join(alignment['down'])
        if output is not None:
            return string
        else:
            print(string)
    
    # format as html table
    if output == "html":
        html_string = HTML_HEADER + '<body>\n<table>\n'
        html_string +=  "<tr>" + list_to_html_row(alignment['up']) + "</tr>\n"
        html_string += "<tr>" + list_to_html_row(alignment['relation']) + "</tr>\n"
        html_string += "<tr>" + list_to_html_row(alignment['down']) + "</tr>\n"
        html_string += "</table>\n<body>\n"
        return html_string


def get_borderline(a: np.array, cutoff: int = 10) -> np.ndarray:
    '''
    extract all possible border indices (down, right) for given 2D matrix
    for example: \n
        A A A A A X\n
        A A A A A X\n
        A A A A A X\n
        A A A A A X\n
        A A A A A X\n
        X X X X X X\n
    \n
    result will contain indices of `X` values starting from upper right to lower left
    Args:
        a: (np.ndarray)
        cutoff: (int) control how far stay from edges - the nearer the edge the shorter diagonal
    Returns:
        boderline: (np.ndarray) [len, 2]
    '''
    # width aka bottom
    height, width = a.shape
    height -= 1; width -= 1
    # clip values        

    if height < cutoff:
        hstart = 0
    else:
        hstart = cutoff

    if width < cutoff:
        bstart = 0
    else:
        bstart = cutoff
    # arange with add syntetic dimension
    # height + 1 is here for diagonal
    hindices = np.arange(hstart, height+1)[:, None]
    # add new axis
    hindices = np.repeat(hindices, 2, axis=1)
    hindices[:, 1] = width

    # same operations for bottom line
    # but in reverted order
    bindices = np.arange(bstart, width)[::-1, None]
    # add new axis
    bindices = np.repeat(bindices, 2, axis=1)
    bindices[:, 0] = height
    
    borderline = np.vstack((hindices, bindices))
    return borderline


def border_argmaxpool(array: np.ndarray,
                    cutoff: int = 10,
                    factor: int = 2) -> np.ndarray:
    '''
    get border indices of an array satysfing
    Args:
        array: (np.ndarray)
        cutoff: (int)
        factor: (int)
    Returns:
        borderindices: (np.ndarray)
    '''
    assert factor > 0
    assert cutoff >= 0
    assert isinstance(factor, int)
    assert cutoff*2 < (array.shape[0] + array.shape[1]), 'cutoff exeed array size'

    boderindices = get_borderline(array, cutoff=cutoff)
    if factor > 1:
        y, x = boderindices[:, 0], boderindices[:, 1]
        bordevals = array[y, x]
        num_values = bordevals.shape[0]    
        # make num_values divisible by `factor` 
        num_values = (num_values - (num_values % factor))
        # arange shape (num_values//factor, factor)
        # argmax over 1 axis is desired index over pool 
        arange2d = np.arange(0, num_values).reshape(-1, factor)
        arange2d_idx = np.arange(0, num_values, factor, dtype=np.int32)
        borderargmax = bordevals[arange2d].argmax(1)
        # add push factor so values  in range (0, factor) are translated
        # into (0, num_values)
        borderargmax += arange2d_idx
        return boderindices[borderargmax, :]
    else:
        return boderindices


def gather_all_paths(array: np.ndarray,
                    minlen: int = 10,
                    norm: Union[bool, str] = 'rows',
                    bfactor: int = 1,
                    gap_opening: float = 0,
                    gap_extension: float = 0,
                    with_scores: bool = False) -> List[np.ndarray]:
    '''
    calculate scoring matrix from input substitution matrix `array`
    find all Smith-Waterman-like paths from bottom and right edges of scoring matrix
    Args:
        array: (np.ndarray) raw subtitution matrix aka densitymap
        norm_rows: (bool, str) whether to normalize array per row or per array
        bfactor: (int) use argmax pooling when extracting borders, bigger values will improve performence but may lower accuracy
        with_scores: (bool) if True return score matrix
    Returns:
        paths: (list) list of all valid paths through scoring matrix
        score_matrix: (np.ndarray) scoring matrix used
    '''
    if not isinstance(array, np.ndarray):
        array = array.numpy().astype(np.float32)
    if not isinstance(norm, (str, bool)):
        raise ValueError(f'norm_rows arg should be bool/str type, but given: {norm}')
    # standarize embedding
    if isinstance(norm, bool):
        if norm:
            array -= array.mean()
            array /= (array.std() + 1e-3)
        
    elif isinstance(norm, str):
        if norm == 'rows':
            array = array - array.mean(axis=1, keepdims=True)
            array = array / array.std(axis=1, keepdims=True)
        elif norm == 'cols':
            array = array - array.mean(axis=0, keepdims=True)
            array = array / array.std(axis=0, keepdims=True)
    score_matrix = fill_score_matrix(array)
    # get all edge indices for left and bottom
    # score_matrix shape array.shape + 1
    indices = border_argmaxpool(score_matrix, cutoff=minlen, factor=bfactor)
    paths = list()
    for ind in indices:
        yi, xi = ind
        if score_matrix[yi, xi] < 1:
            continue
        path = traceback_from_point_opt2(score_matrix, ind, gap_opening=gap_opening, gap_extension=gap_extension)
        paths.append(path)
    if with_scores:
        return paths, score_matrix
    else:
        return paths


def scale_embeddings(resultsdf: pd.DataFrame, deensitymap: torch.Tensor, deep_blosum: torch.Tensor, seq1: str, seq2: str) -> pd.DataFrame:
    '''
    Args:
    '''
    assert isinstance(resultsdf, pd.DataFrame)
    assert isinstance(deep_blosum, torch.Tensor)
    assert deep_blosum.ndim == 2
    assert isinstance(seq1, str)
    assert isinstance(seq1, str)

    score_scaled = []
    prod_scaled = []
    geo_scaled = []    
    for _, row in resultsdf.iterrows():
        # extract string from alignment plot which allows to include gaps
        # draw alignemnt sequence order is reverted
        str1, _, str2 = draw_alignment(row.indices, seq2, seq1, as_string=True).split('\n')
        ycoords, xcoords = map(np.asarray, zip(*row.indices))

        remb1 = sequence_to_number(str1)
        remb2 = sequence_to_number(str2)
        
        densitymap_route = deensitymap[ycoords, xcoords]
        norm_route = deep_blosum[remb1, remb2]
        normed = densitymap_route / norm_route
        span_len = normed.shape[0]
        normedscore = normed.mean().item()
        normedprod = harmonic_mean(normed)
        normedgeo = geometric_mean(normed)
        score_scaled.append(normedscore)
        prod_scaled.append(normedprod)
        geo_scaled.append(normedgeo)

    resultsdf['score_scaled'] = score_scaled
    resultsdf['hscore_scaled'] = prod_scaled
    return resultsdf