'''functions for local density extracting'''
from typing import List, Union, Tuple, Dict
from typing import Generator
import math

import torch as th
import torch.nn.functional as F


def batch_slice_iterator(listlen: int, batchsize: int) -> Generator[slice, None, None]:

	assert isinstance(listlen, int)
	assert isinstance(batchsize, int)
	assert listlen > 0
	assert batchsize > 0
	# clip if needed
	batchsize = listlen if listlen < batchsize else batchsize
	num_batches: int = math.ceil(listlen / batchsize)
	for b in range(num_batches):
		bstart = b*batchsize
		bstop = min((b + 1)*batchsize, listlen)
		sl = slice(bstart, bstop)
		yield sl


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


def last_pad(a: th.Tensor, dim: int, pad: Tuple[int, int]) -> th.Tensor:
	'''
	pads tensor up/down or left right
	'''
	assert pad[0] > 0
	assert pad[1] > 0
	num_dim = a.ndim
	#assert a.ndim == 4, f'only 4D tensors are supplied'
	if num_dim == 3:
		a = a.unsqueeze(0)
	elif num_dim == 2:
		a = a.unsqueeze(0).unsqueeze(0)
	assert dim == -1 or dim == -2
	assert len(pad) == 2, f'padding works only for 2 element pad tuple'
	if dim == -1:
		left_factor, right_factor = pad[0], pad[1]
		left_pad = a[:, :, :, 0].repeat_interleave(repeats=left_factor, dim=-2)
		right_pad = a[:, :, :, -1].repeat_interleave(repeats=right_factor, dim=-2)
		left_pad = left_pad.unsqueeze(0).swapdims(-1, -2)
		right_pad = right_pad.unsqueeze(0).swapdims(-1, -2)
		a = th.cat((left_pad, a, right_pad), dim=dim)
	elif dim == -2:
		up_factor, down_factor = pad[0], pad[1]
		up_pad = a[:, :, 0, :].repeat_interleave(repeats=up_factor, dim=-2)
		down_pad = a[:, :, -1, :].repeat_interleave(repeats=down_factor, dim=-2)
		up_pad = up_pad.unsqueeze(0)
		down_pad = down_pad.unsqueeze(0)
		a = th.cat((up_pad, a, down_pad), dim=dim)
		
	if num_dim == 3:
		a = a.squeeze(0)
	elif num_dim == 2:
		a = a.squeeze(0).squeeze(0)
	return a


def sequence_to_filters(protein: th.Tensor,
						kernel_size : int,
						stride: int = 1,
						norm :bool = True,
						with_padding : bool = True):
	r'''
	Convert embedding to tensor of 2d filters with shape of:
	[num_filters, 1, kernel_size, emb_size]
	from input `protein` with shape: 
	[seq_len, emb_size]
	Args:
		kernel_size: int
		stride: (int)
		norm (bool) whether to apply normalization to filters
	Returns:
		filters: (torch.FloatTensor) [num_filters, 1, kernel_size, emb_size]
	'''
	r: int = 0
	padding: int = 0
	num_filters: int = 0
	# store device for possible CUDA use
	device = protein.device
	seqlen, emb_size = protein.shape[0], protein.shape[1]
	if protein.ndim != 2:
		raise ArithmeticError(f'protein arg must be 2 dim, given: {protein.shape}')
	# kernel must be <= sequence lenght
	if kernel_size >= seqlen:
		kernel_size = int(seqlen)
	# calculate padding values
	if with_padding:
		r = (seqlen - kernel_size) % stride
		if r > 0:
			padding = stride - r
		if padding % 2 == 0:
			paddingh = padding//2 - 1
			paddingw = padding//2
		else:
			padding = (padding - 1)//2 
			paddingh = paddingw = padding
			paddingh, paddingw = int(paddingh), int(paddingw)
		protein = last_pad(protein, dim=-2, pad=(paddingh, paddingw))
	# protein = F.pad(protein, (0, 0, paddingh, paddingw), 'constant', 0.0)
	filter_list: List[th.Tensor] = list()
	# rare case when embedding is very small
	if 0 < seqlen-kernel_size < stride:
		stride = seqlen - kernel_size
	for start in range(0, seqlen-kernel_size + 1, stride):
			# single filter of size (1, kerenl_size, num_emb_feats)
			# last sample boundary when padding is false
			if start + kernel_size > seqlen:
				start = seqlen - kernel_size
			filt = protein.narrow(0, start, kernel_size)
			filter_list.append(filt)
	# merge filters and create addditional dimension for it
	# filters: [num_filters, 1, kernel_size, emb_size]
	filters = th.stack(filter_list)
	filters = filters.view(-1, 1, kernel_size, emb_size)
	# normalize filters
	if norm:
		num_filters = filters.shape[0]
		norm_val = filters.view(num_filters, -1).pow(2).sum(1, keepdim=True)
		norm_val[norm_val == 0] = 1e-5
		norm_val = norm_val.sqrt().view(num_filters, 1, 1, 1)
		filters /= norm_val
	return filters.to(device)


def calc_density(protein: th.Tensor, filters: th.Tensor,
				 stride : int = 1, with_padding: bool = True) -> th.Tensor:
	'''
	convolve `protein` with set of `filters`
	* warning output is unsqueezed
	params:
		protein: (seqlen, emb_size)
		filters: (num_filters, 1, kernel_size, emb_size)
	'''
	assert filters.ndim == 4, 'invalid filters shape required (num_filters, 1, kernel_size, emb_size)'
	# add dimensions up to 4
	if protein.ndim == 2:
		protein = protein.unsqueeze(0).unsqueeze(0)
	elif protein.ndim == 3:
		protein = protein.unsqueeze(0)
	else:
		raise ValueError(f'protein incorrect number of dimensions: {protein.ndim} expected 2 or 3')
	kernel_size = filters.shape[2]
	if kernel_size % 2 == 0:
		paddingh = kernel_size//2 - 1
		paddingw = kernel_size//2
	else:
		padding = (kernel_size - 1)//2 
		paddingh = paddingw = padding
	# add padding to protein to maintain size
	if with_padding:
		protein = last_pad(protein, dim=-2, pad=(paddingh, paddingw))
	density = F.conv2d(protein, filters, stride = stride)
	#output density
	return density


def norm_image(arr: th.Tensor, kernel_size : int):
	'''
	apply L2 norm over image for conv normalization
	'''
	if arr.ndim == 3:
		arr = arr.unsqueeze(0)
	elif arr.ndim == 2:
		arr = arr.unsqueeze(0).unsqueeze(0)
	elif arr.ndim > 3 or arr.ndim < 2:
		ValueError(f'arr dim is ={arr.ndim} required 2-4') 
	if kernel_size % 2 == 0:
		padding = kernel_size//2
	else:
		padding = (kernel_size - 1)//2 
	H, W = arr.shape[-2], arr.shape[-1]
	# normalization kernel: (kernel_size, emb_dim = W)
	arr_sliced = F.unfold(arr, (kernel_size, W))
	arr_sliced_norm = arr_sliced.pow(2).sum(1, keepdim=True).sqrt()
	arr_sliced /= arr_sliced_norm
	arr_normalized = F.fold(arr_sliced, (H, W), (kernel_size, W))
	return arr_normalized


@th.jit.script
def get_symmetric_density(X, Y, kernel_size : int):
	'''
	caclulates symmetric density for two proteins
	output is in form of (num_residues_x, num_residues_y)
	return torch.FloatTensor
	'''
	# image 
	Y_as_image = norm_image(Y, kernel_size)
	X_as_image = norm_image(X, kernel_size)
	# filters are normalized when slicing 
	X_as_filters = sequence_to_filters(X, kernel_size)
	Y_as_filters = sequence_to_filters(Y, kernel_size)
	XY_density = calc_density(X_as_image, Y_as_filters)
	YX_density = calc_density(Y_as_image, X_as_filters)
	return (XY_density + YX_density.T)/(2*kernel_size)


@th.jit.script
def get_fast_density(X, Y, kernel_size : int):

	if X.shape[0] < Y.shape[0]:
		as_image = norm_image(Y, kernel_size)
		as_filters = sequence_to_filters(X, kernel_size)
	else:
		as_image = norm_image(X, kernel_size)
		as_filters = sequence_to_filters(Y, kernel_size)
	density = calc_density(as_image, as_filters)
	return density


def get_multires_density(X: th.Tensor,
						 Y: th.Tensor,
						 kernels: Union[List[int], int] = 1,
						 raw: bool=False):
	'''
	compute X, Y similarity by convolving them
	result shape is (Y.shape[0], X.shape[0]) in other words
	(num. of Y residues, num of X residues)
	remarks
		* only odd kernel will result strict matching between input X,Y indices and resulting density
		* the lower kernel size is the better resolution in resulting density
		* bigger kernels will speed up computations
	params:
		X, Y - (torch.Tensor) protein embeddings as 2D tensors
		kernels (list or int) define the resolution of embedding map, the lower the better
		raw (bool) if more then one kernel is supplied then resulting density will store all of them separately
	'''
	if not isinstance(kernels, (list, tuple)):
		kernels = [kernels]
	if X.shape[0] == 1 or Y.shape[0] == 1:
		raise ValueError(f'''
		X, Y embedding shape must follow the pattern (num_res, emb_dim)
		where num_res > 1 and emb_dim > 1
		given X: {X.shape} and Y {Y.shape}
		''')
	if X.shape[1] != Y.shape[1]:
		raise ValueError(f'''
		X, Y embedding shape must follow the pattern (num_res, emb_dim)
		where num_res > 1 and emb_dim > 1
		given X: {X.shape} and Y {Y.shape}
		''')
	for ks in kernels:
		if ks >= X.shape[0] or ks >= Y.shape[0]:
			raise ValueError(f'''kernel exceeds length of sequences ks {ks}
			X seq len {X.shape[0]} and Y seq len {Y.shape[0]}
			''')
	result = list()
	for ks in kernels:
		result.append(
			get_symmetric_density(X, Y, ks).unsqueeze(0)
		)
	result = th.cat(result, 0)
	if not raw:
		result = result.mean(0)
	return result


def chunk_cosine_similarity(query : Union[th.Tensor, List[th.Tensor]],
							targets : List[th.Tensor],
							quantile: float, dataset_files : List[str],
							stride: int = 3, kernel_size: int = 30) -> List[Dict[int, str]]:
	# soft type check
	assert isinstance(targets, list)
	if isinstance(query, th.Tensor):
		assert query.ndim == 2
	elif isinstance(query, list):
		assert len(query) > 0
		assert isinstance(query[0], th.Tensor)
	assert isinstance(quantile, float)
	assert 0 <= quantile <= 1
	assert isinstance(dataset_files, list)
	assert isinstance(kernel_size, int)
	
	# type checks
	kernel_size = int(kernel_size)
	num_targets = len(targets)
	if isinstance(query, th.Tensor):
		query = [query]
	scorestack = th.zeros(num_targets)
	seqlens: List[int] = [emb.shape[0] for emb in targets] + [q.shape[0] for q in query]
	# change kernel size if the shortest sequence in targets is smaller then kernel size
	min_seqlen = min(seqlens)
	if kernel_size > min_seqlen:
		kernel_size = min_seqlen
	assert len(targets) == len(dataset_files), f'{len(targets)} != {len(dataset_files)}'
	# scorestack: [num_targets, num_queries]
	scorestack = chunk_score_batch(query, targets=targets, stride=stride, kernel_size=kernel_size)
	quantile_threshold = th.quantile(scorestack, quantile, dim=0)
	scoremask = (scorestack >= quantile_threshold)
	# convert mask to indices
	results = list()
	for qnb in range(scorestack.shape[1]):
		q_scoremask = scoremask[:, qnb].view(-1)
		filedict: Dict[int, str] = {
			i : file for i, (file, condition) in enumerate(zip(dataset_files, q_scoremask)) if condition
			}
		results.append(filedict)
	return results


@th.jit.script
def norm_chunk(target, kernel_size: int, embdim: int, stride: int):
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



@th.jit.script
def single_process(query_kernels: th.Tensor, target: th.Tensor, stride: int):
	'''
	Args:
		query_kernels: (torch.Tensor) []
	Returns:
		torch.Tensor: [query_kernels, target_kernels]
	'''
	embdim : int = int(target.shape[-1])
	kernel_size: int = int(query_kernels.shape[-2])
	density = calc_density(target, query_kernels, stride=stride, with_padding=False).squeeze()
	density /= norm_chunk(target=target, kernel_size=kernel_size, embdim=embdim, stride=stride)
	if density.ndim == 1:
		density = density.unsqueeze(0)
	return density


@th.jit.script
def chunk_score(query, targets: List[th.Tensor], stride: int, kernel_size: int):
	'''
	perform chunk cosine similarity screening
	Args:
		query: (torch.FloatTensor)
		targets: (list of torch.FloatTensor) embeddings to compare
		stride: (int)
		kernel_size: (int)
	Returns:
		scorestack: (torch.FloatTensor) [num_targets, 1]
	'''
	num_targets: int = len(targets)
	scorestack: List[float] = list()
	query_kernels = sequence_to_filters(query, kernel_size=kernel_size,
										norm=True, with_padding=False)
	
	#for i, target in enumerate(targets, 0):
	#		density = single_process(query_kernels, target, stride=stride)
	#	scorestack[i] = density.max()
	scorestack = [single_process(query_kernels, t, stride=stride).max().item() for t in targets]
	scorestack_t = th.zeros(num_targets, dtype=th.float32)
	for i in th.arange(num_targets):
		scorestack_t[i] = scorestack[i]
	if scorestack_t.shape[0] != num_targets:
		raise ValueError(f'''cosine sim screening result different
		  number of embeddings than db {scorestack_t.shape[0]} - {num_targets}''')
	return scorestack_t.unsqueeze(1)


def unfold_large_db(targets, kernel_size: int, stride: int, embdim: int):

	num_targets = len(targets)
	targets_unfold: List[th.Tensor] = list()
	unfold_size: List[int] = list()
	for bstart, bstop in batch_slice_iterator_script(num_targets, batchsize=12800):
		# shape: [target_folds, query_folds]()
		batch_targets, unfold_size = unfold_targets(targets[bstart:bstop],
											   kernel_size=kernel_size,
												stride=stride, embdim=embdim)
		targets_unfold.append(targets_unfold)
		unfold_size.extend(unfold_size)
	targets_unfold = th.cat(targets_unfold, dim=0)
	return targets_unfold, unfold_size


@th.jit.script
def chunk_score_batch(queries: List[th.Tensor], targets: List[th.Tensor], stride: int, kernel_size: int):
	'''
	perform chunk cosine similarity screening

	Args:
		queries (List[torch.FloatTensor]):
		targets (List[torch.FloatTensor]): embeddings to compare
		stride (int):
		kernel_size (int):
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
	return scorestack_t


@th.jit.script
def calculate_pool_embs(embs: List[th.Tensor]) -> List[th.Tensor]:
    """
    convert embeddings to torch.float32 and [seqlen, 64]
    """
    if len(embs) == 0:
        raise ValueError('target database is empty')
    return [F.avg_pool1d(emb.float().unsqueeze(0), 16).squeeze() for emb in embs]