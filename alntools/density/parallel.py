'''handle parallel embedding file loading'''
import os
import time
from typing import Union, List, Dict, Optional, Union
import warnings

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from torch.nn.functional import avg_pool1d



def worker_init_fn(worker_id):
	'''
	to fix dataloader speed issue, source:
	https://discuss.pytorch.org/t/dataloader-seeding-issue-for-multithreading-workloads/81044/3
	'''
	np.random.seed(np.random.get_state()[1][0] + worker_id + int(time.perf_counter() * 1000 % 1000))


class Database(torch.utils.data.Dataset):
	'''
	handle loading database composed from single files
	'''
	def __init__(self, dbpath : os.PathLike, suffix :str = '.emb', device : torch.device = torch.device('cpu')):

		self.device = device
		dirname = os.path.dirname(dbpath)
		if not (dirname == ''):
			if not os.path.isdir(dirname):
				raise FileExistsError(f'directory: {dirname} is bad')
		self.basedata = pd.read_csv(dbpath + '.csv')
		num_records = self.basedata.shape[0]
		self.embedding_files = [f'{i}{suffix}' for i in range(num_records)]
		# add preffix
		self.embedding_files = [os.path.join(dbpath, ind) for ind in self.embedding_files]
		# exclude missing proteins
		self.embedding_files = [file for file in self.embedding_files if os.path.isfile(file)]
		num_missed = int(num_records - len(self.embedding_files))
		if num_missed < num_missed:
			warnings.warn(f'{num_records - len(self.embedding_files)} missing protein embeddings')
		elif num_missed == num_records:
			raise FileNotFoundError('no embeddings found')

	def __len__(self):
		return len(self.embedding_files)

	def __getitem__(self, idx):
		embedding = torch.load(self.embedding_files[idx])
		return embedding


class DatabaseChunk(torch.utils.data.Dataset):
	'''
	handle loading database composed from single files
	'''
	def __init__(self, path: List[os.PathLike], num_records: int, flatten: bool = False):

		assert os.path.isdir(path), f"path {path} is no a valid directory"
		dirname = os.path.dirname(path)
		if not (dirname == ''):
			if not os.path.isdir(dirname):
				raise FileExistsError(f'directory: {dirname} is bad')
		self.embedding_files = [os.path.join(path, f'{f}.emb') for f in range(0, num_records)]
		# check if all file exists
		for file in self.embedding_files:
			if not os.path.isfile(file):
				raise FileExistsError(f'missing file: {file}')
		self.flatten = flatten

	def __len__(self):
		return len(self.embedding_files)

	def __getitem__(self, idx):
		embedding = torch.load(self.embedding_files[idx])
		if self.flatten:
			embedding = embedding.sum(0)
		return embedding


def load_embeddings_parallel(path: str, num_records: int, num_workers: Optional[int] = 0) -> List[torch.Tensor]:
	batch_size = 128
	# TODO optimize this choice
	dataset = DatabaseChunk(path=path, num_records=num_records)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=lambda x: x, worker_init_fn=worker_init_fn)
	embeddinglist = list()
	for batch in dataloader:
		embeddinglist.extend(batch)
	return embeddinglist


def load_embeddings_parallel_generator(path: str, num_records: int, batch_size: int = 1, num_workers: Optional[int] = 0) -> List[torch.Tensor]:
	# TODO optimize this choice
	if os.path.isfile(path):
		dataset = torch.load(path)
	elif os.path.isdir(path):
		dataset = DatabaseChunk(path=path, num_records=num_records)
	else:
		raise FileNotFoundError(f"path is not valid directory: {path}")
	
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=lambda x: x, worker_init_fn=worker_init_fn)
	for batch in dataloader:
		yield batch


def load_and_score_database(query_emb : torch.Tensor,
							dbpath: str,
							num_records: Optional[int],
							quantile : float = 0.9,
							num_workers: int = 1,
							device : torch.device = torch.device('cpu')) -> Dict[int, str]:
	'''
	perform cosine similarity screening
	Args:
		dbpath (str): file with .pt extension or directory
	Returns:
		(dict): with file id and path to embedding used
	'''
	assert 0 < quantile < 1
	if isinstance(query_emb, list):
		query_emb = query_emb[0]
	batch_size = 256
	pooling = 1
	num_workers = 0
	verbose = False
	# setup database
	if os.path.isfile(dbpath):
		dataset = torch.load(dbpath)
	elif os.path.isdir(dbpath):
		dataset = DatabaseChunk(dbpath, num_records=num_records, flatten=True)
	else:
		raise FileNotFoundError('dbpath: {dbpath} is not directory or embedding file')
	num_embeddings = len(dataset)
	dataloader = torch.utils.data.DataLoader(dataset,
								batch_size=batch_size,
								num_workers = num_workers,
								drop_last = False,
								worker_init_fn=worker_init_fn)
	num_batches = int(num_embeddings/batch_size)
	# equivalent to math.ceil
	num_batches = num_batches if num_batches*batch_size <= num_embeddings else num_batches + 1
	if query_emb.shape[0] != 1:
		query_emb = query_emb.sum(0, keepdim=True).T
	if query_emb.ndim == 1:
		query_emb = query_emb.view(-1, 1)
	if pooling > 1:
		query_emb = avg_pool1d(query_emb.T, pooling).T
	scorestack = []
	dataset_files = dataset.embedding_files
	with tqdm(total = num_batches) as pbar:
		for i, batch in enumerate(dataloader):
			# TODO process all queries at once
			# this should give huge performence boost
			score = batch_cosine_similarity(query_emb, batch, poolfactor=pooling)
			scorestack.append(score)
			pbar.update(1)
	
	scorestack = torch.cat(scorestack)
	if scorestack.shape[0] != num_embeddings:
		raise ValueError(f'cosine sim screening result different number of embeddings than db {scorestack.shape[0]} - {num_embeddings}')
	# quantile filter
	quantile_threshold = torch.quantile(scorestack, quantile)
	scoremask = (scorestack >= quantile_threshold)
	# convert maask to indices
	scoreidx = torch.nonzero(scoremask, as_tuple=False).tolist()
	filedict = { i : file for i, (file, condition) in enumerate(zip(dataset_files, scoremask)) if condition }

	if verbose:
		print(f'{len(scoreidx)}/{len(dataset_files)}')
		print(len(dataset))
		print(scoremask.shape)
	assert scorestack.shape[0] == len(dataset_files)
	return filedict



def batch_cosine_similarity(x : torch.Tensor, B : torch.Tensor, poolfactor: int) -> torch.Tensor:
	'''
	first dimension should be embedding dimenson, expects x: [embdim, 1] and B: [embdim, batch_size]
	'''
	assert x.ndim == 2
	assert B.ndim == 2
	if poolfactor > 1:
		B = avg_pool1d(B.T, poolfactor).T
	# embedding dimension match
	if B.shape[0] != x.shape[0]:
		B = B.T
	score = torch.nn.functional.cosine_similarity(x, B, dim=0)
	if score.ndim > 1:
		score = score.ravel()
	return score


def load_full_embeddings(filelist : List[os.PathLike],
							poolfactor:  Union[int, None] = None) -> List[torch.Tensor]:
	
	'''
	read per residue embeddings
	Args:
		filelist: (str) list of files each file should be separate protein embedding
	Returns:
		stack: (list[torch.Tensor])
	'''
	stack = []
	'''
	dataset = DatabaseChunk(dbpath=filelist)
	dataloader = torch.utils.data.DataLoader(dataset,
								batch_size=batch_size,
								collate_fn = lambda x: x,
								num_workers = num_workers,
								drop_last = False)
	stack = []
	for batch in tqdm(dataloader):
		stack.extend(batch)
	'''
	missing_files : int = 0

	for itr, file in enumerate(filelist):
		if not os.path.isfile(file):
				missing_files += 1
				continue
		if poolfactor is not None:
			embedding = torch.load(file).float()
			embedding = avg_pool1d(embedding.unsqueeze(0), poolfactor)
			stack.append(embedding.squeeze(0))
		else:
			stack.append(torch.load(file).numpy())

	assert missing_files==0, f'embedding missing files: {missing_files}'
	return stack
