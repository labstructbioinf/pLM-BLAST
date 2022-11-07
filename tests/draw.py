'''drawing function tests'''
import os
import sys
import time
import faulthandler 
sys.path.append('..')
import pytest

from alntools import draw_alignment

seq1_test = 'ABCDEFGHIJKLMN'
seq2_test = 'ABCDEFFGHIJKLMNOPRST'

coords_test = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (12, 13),
    (12, 14),
    (12, 15),
    (12, 16),
    (12, 17),
    (12, 18),
    (13, 19)
]
string_test =  'A--BCDEF-GIJKLM----N'
string_test += '|  | ||| ||||||    |'
string_test += 'ABCD-EFGHIJKLMNOPRST'

# case when coords dont match seq1/seq2
coords_test_invalid1 = coords_test + [(14, 19)]
coords_test_invalid2 = coords_test + [(13, 20)]


@pytest.mark.parametrize("coords", [coords_test])
@pytest.mark.parametrize("seq1", [seq1_test])
@pytest.mark.parametrize("seq2", [seq2_test])
def test_aln_draw(coords, seq1, seq2):
    
    result = draw_alignment(coords=coords, seq1=seq1, seq2=seq2)
    assert result is None, 'result should be None'
    result = draw_alignment(coords=coords, seq1=seq1, seq2=seq2, as_string=True)
    
    if string_test != string_test:
        print(string_test)
        print(result)
        raise ValueError('alignmnet is different then desired')


@pytest.mark.parametrize("coords", [coords_test_invalid1, coords_test_invalid2])
@pytest.mark.parametrize("seq1", [seq1_test, seq1_test])
@pytest.mark.parametrize("seq2", [seq2_test, seq2_test])
def test_aln_draw(coords, seq1, seq2):
    print(coords)
    try:
        result = draw_alignment(coords=coords, seq1=seq1, seq2=seq2)
    except KeyError:
        pass