from typing import List
import torch

def check_cohesion(sequences: List[str],
				   filedict: dict,
				   embeddings: List[torch.FloatTensor],
				   truncate: int = 1000):
	'''
	check for missmatch between sequences and their embeddings
	'''
	for (idx,file), emb in zip(filedict.items(), embeddings):
		seqlen = len(sequences[idx])
		if seqlen < truncate:
			assert seqlen == emb.shape[0], f'''
			index and embeddings files differ, for idx {idx} seqlen {seqlen} and emb {emb.shape} file: {file}'''
		else:
			pass
