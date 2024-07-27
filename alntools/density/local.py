'''functions for local density extracting'''
from typing import List, Union, Tuple, Dict
from typing import Generator
import math
import itertools

import torch as th
import torch.nn.functional as F


def batch_slice_iterator_script(listlen: int, batchsize: int) -> List[Tuple[int, int]]:

	assert isinstance(listlen, int)
	assert isinstance(batchsize, int)
	assert listlen > 0
	assert batchsize > 0
	# clip if needed
	batchsize = listlen if listlen < batchsize else batchsize
	batches: List[Tuple[int, int]] = list()
	num_batches: int = math.ceil(listlen / batchsize)
	for b in range(num_batches):
		bstart = b*batchsize
		bstop = min((b + 1)*batchsize, listlen)
		batches.append((bstart, bstop))
	return batches


def chunk_cosine_similarity(query : Union[th.Tensor, List[th.Tensor]],
							targets : List[th.Tensor],
							quantile: float, dataset_files : List[str],
							stride: int = 3, kernel_size: int = 30) -> List[Dict[int, str]]:

	# soft type check
	assert isinstance(targets, Dict)
	if isinstance(query, th.Tensor):
		assert query.ndim == 2
	elif isinstance(query, list):
		assert len(query) > 0
		assert isinstance(query[0], th.Tensor)
	assert isinstance(quantile, float)
	assert 0 <= quantile <= 1
	assert 0 <= quantile <= 1
	assert isinstance(dataset_files, list)
	assert isinstance(kernel_size, int)
	
	
	results = list()
	seqlen_targets = list(itertools.chain(*[v[1] for v in targets.values()]))
	num_targets = len(seqlen_targets)
	if isinstance(query, th.Tensor):
		query = [query]
	scorestack = th.zeros(num_targets)
	assert num_targets == len(dataset_files), f'{num_targets} != {len(dataset_files)}'
	# scorestack: [num_targets, num_queries]
	scorestack = chunk_score_batchdb(query, targets=targets, stride=stride, kernel_size=kernel_size)
	quantile_threshold = th.quantile(scorestack, quantile, dim=0)
	scoremask = (scorestack >= quantile_threshold)
	# convert mask to indices
	for qnb in range(scorestack.shape[1]):
		q_sm = scoremask[:, qnb].view(-1).tolist()
		q_st = scorestack[:, qnb].view(-1).tolist()
		filedict: Dict[int, str] = {
			i : dict(file=file, score=score) 
				for i, (file, condition, score) in enumerate(zip(dataset_files, q_sm, q_st))
				if condition
			}
		results.append(filedict)
	del scorestack
	return results


@th.jit.script
def norm_chunk(target, kernel_size: int, embdim: int, stride: int):
	'''
	Returns:
		torch.Tensor: [?]
	'''
	'''
	Returns:
		torch.Tensor: [?]
	'''
	# hard type params
	unfold_kernel: List[int] = [kernel_size, embdim]
	stride_kernel: List[int] = [stride, 1]

	target = target.unsqueeze(0).unsqueeze(0)
	target_norm = th.nn.functional.unfold(target, kernel_size=unfold_kernel, stride=stride_kernel)
	target_norm = target_norm.squeeze()
	target_norm = target_norm.pow(2).sum(0).sqrt()
	return target_norm



@th.jit.script
def unfold_targets(targets: List[th.Tensor], kernel_size: int, stride: int, embdim: int):
	'''
	[kernel_size*embdim, num_folds]
	each fold is euclidian normalized
	'''
	tgt_flat: List[th.Tensor] = list()
	tgt_folds: List[int] = list()
	num_folds: int = 0
	unfold_kernel: List[int] = [kernel_size, embdim]
	stride_kernel: List[int] = [stride, 1]
	for tgt in targets:
		# target: [1, kernel_size*embdim, num_folds]
		target = th.nn.functional.unfold(tgt.unsqueeze(0).unsqueeze(0), kernel_size=unfold_kernel, stride=stride_kernel)
		#print(target.shape)
		num_tgt_folds = target.shape[2]
		tgt_folds.append(num_tgt_folds)
		tgt_flat.append(target)
		num_folds += num_tgt_folds
	#tgt_flat_t = th.empty((1, kernel_size*embdim, num_folds))
	tgt_flat_t = th.cat(tgt_flat, dim=2)
	tgt_flat_t_norm = tgt_flat_t
	tgt_flat_t_norm = tgt_flat_t.pow(2).sum(1, keepdim=True).sqrt()
	tgt_flat_t /= tgt_flat_t_norm
	return tgt_flat_t.swapdims(0, 2).squeeze(-1), tgt_folds


def unfold_large_db(targets: List[th.tensor], kernel_size: int, stride: int, embdim: int):
	'''
	convert embedding list into batches of unfolded tensors
	Args:
		targets: (list[Tensors]) database embeddings
	Returns:
		Dict[int, Tuple[Tensor, List[int]]] each key is a batch id and values
		  are unfolded embeddings with their sizes
	'''
	assert isinstance(targets, list)
	if targets[0].shape[1] != embdim:
		raise ValueError(f'miss-shape between target and embdim {targets[0].shape} and {embdim}')
	num_targets = len(targets)
	
	unfold_size: List[int] = list()
	batches = dict()
	for i, (bstart, bstop) in enumerate(batch_slice_iterator_script(num_targets, batchsize=12800)):
		# shape: [target_folds, query_folds]()
		batch_targets, unfold_size = unfold_targets(targets[bstart:bstop],
											   kernel_size=kernel_size,
												stride=stride, embdim=embdim)
		batches[i] = (batch_targets, unfold_size)
	return batches


@th.jit.script
def chunk_score_batch(queries: List[th.Tensor], targets: List[th.Tensor], stride: int, kernel_size: int):
	'''
	perform chunk cosine similarity screening using fold/unfold and matmul instead of convolutions

	Args:
		queries: (List[torch.FloatTensor])
		targets: (List[torch.FloatTensor]) embeddings to compare
		stride: (int)
		kernel_size: (int)
	Returns:
		(torch.FloatTensor): [num_targets, num_queries]
	'''
	num_targets: int = len(targets)
	num_queries: int = len(queries)
	embdim: int = queries[0].shape[-1]
	kernels: List[th.Tensor] = list()
	kernel_splits: List[int] = list()
	unfolds: List[int] = list()
	result_stack: List[th.Tensor] = list()
	kernels, kernel_splits = unfold_targets(queries, kernel_size=kernel_size, stride=stride, embdim=embdim)
	kernels = kernels.swapdims(0, 1)

	for bstart, bstop in batch_slice_iterator_script(num_targets, batchsize=12800):
		# shape: [target_folds, query_folds]()
		batch_targets, unfold_size = unfold_targets(targets[bstart:bstop],
											   kernel_size=kernel_size,
												stride=stride, embdim=embdim)
		#print(batch_targets.shape, kernels.shape)
		results = th.matmul(batch_targets, kernels)
		result_stack.append(results)
		unfolds.extend(unfold_size)
	result_stack = th.cat(result_stack, dim=0)
	scorestack_t = th.empty((num_targets, num_queries), dtype=th.float32)
	# split over queries
	for qid, qsplit in enumerate(result_stack.split(kernel_splits, dim=1)):
		qsplit, _ = qsplit.max(1)
		# split over targets
		for tid, tqslit in enumerate(qsplit.split(unfolds, dim=0)):
			scorestack_t[tid, qid] = tqslit.max()
	assert (~th.isnan(scorestack_t)).all(), f"nans found {th.isnan(scorestack_t).sum()}%"
	del result_stack
	return scorestack_t


@th.jit.script
def chunk_score_batchdb(queries: List[th.Tensor], targets: Dict[int, Tuple[th.Tensor, List[int]]], stride: int, kernel_size: int):
	'''
	perform chunk cosine similarity screening using fold/unfold and matmul instead of convolutions
	Args:
		queries: (List[torch.FloatTensor])
		targets: (Dict[int, Tuple[th.Tensor, List[int]]]) database as dictionary of batches
		stride: (int)
		kernel_size: (int)
	Returns:
		(torch.FloatTensor): [num_targets, num_queries]
	'''
	num_targets: int = 0
	num_queries: int = len(queries)
	embdim: int = queries[0].shape[-1]
	kernels: List[th.Tensor] = list()
	kernel_splits: List[int] = list()
	unfolds: List[int] = list()
	result_stack: List[th.Tensor] = list()
	kernels, kernel_splits = unfold_targets(queries, kernel_size=kernel_size, stride=stride, embdim=embdim)
	kernels = kernels.swapdims(0, 1)
	for batch_targets, unfold_size in targets.values():
		#print(batch_targets.shape, kernels.shape)
		results = th.matmul(batch_targets, kernels)
		result_stack.append(results)
		unfolds.extend(unfold_size)
		num_targets += len(unfold_size)
	result_stack = th.cat(result_stack, dim=0)
	scorestack_t = th.empty((num_targets, num_queries), dtype=th.float32)
	# split over queries
	for qid, qsplit in enumerate(result_stack.split(kernel_splits, dim=1)):
		qsplit, _ = qsplit.max(1)
		# split over targets
		for tid, tqslit in enumerate(qsplit.split(unfolds, dim=0)):
			scorestack_t[tid, qid] = tqslit.max()
	assert (~th.isnan(scorestack_t)).all(), f"nans found {th.isnan(scorestack_t).sum()}%"
	del result_stack
	return scorestack_t

@th.jit.script
def calculate_pool_embs(embs: List[th.Tensor]) -> List[th.Tensor]:
    """
    convert embeddings to torch.float32 and [seqlen, 64]
    """
    if len(embs) == 0:
        raise ValueError('target database is empty')
    return [F.avg_pool1d(emb.float().unsqueeze(0), 16).squeeze() for emb in embs]
