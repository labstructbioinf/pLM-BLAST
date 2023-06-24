import os
import re
from typing import List
import numpy as np

path_blosum = os.path.join(os.path.dirname(__file__), 'data/blosum62.txt')

def read_blosum(path):
	array = list()
	with open(path, 'rt') as file:
		respace = re.compile('\s+')
		for line in file.readlines():
			if line.startswith('#'): continue
			elif line.startswith(' '):
				reslist = line
				reslist = reslist.strip()
				reslist = reslist[:-3]
				reslist = respace.split(reslist)
			# upper case letter
			elif line[0].isalpha() and line[0].isupper():
				aa_score = line.strip()
				aa_score = respace.split(aa_score)[1:-1]
				array.append(aa_score)
			elif not line[0].isalpha():
				continue
	arr = np.asarray(array, dtype=np.float32)
	res_to_num = {res : i for i, res in enumerate(reslist)}
	return arr, res_to_num

arr, res_to_num = read_blosum(path_blosum)
sign = -1*(arr > 0) + 1*(arr < 0)
arr_sqrt = sign*np.sqrt(np.abs(arr))
arr_sqrt /= np.abs(arr_sqrt).max()
def sequence_to_numbers(seqres: List[str]) -> List[int]:
	'''
	convert list of strings into list of numbers
	'''
	global arr, res_to_num
	assert isinstance(seqres, list), f' sequence must be list of one letter strings'
	if arr is None:
		arr, res_to_num = read_blosum(path_blosum)
	num_list = list()
	for res in seqres:
		num = res_to_num[res]
		num_list.append(num)
	return num_list

def fill_blosum_matrix(seq1, seq2):
	global arr
	seq1_num = sequence_to_numbers(seq1)
	seq2_num = sequence_to_numbers(seq2)
	seq1_mesh, seq2_mesh = np.meshgrid(seq1_num, seq2_num)
	blosum_score = arr_sqrt[seq1_mesh, seq2_mesh]
	return blosum_score
