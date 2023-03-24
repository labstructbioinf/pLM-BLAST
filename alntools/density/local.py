'''functions for local density extracting'''
from typing import List, Union, Tuple

from tqdm import tqdm
import torch as th
import torch.nn.functional as F


def last_pad(a: th.Tensor, dim: int, pad: Tuple[int, int]):
    '''
    pads tensor up/down or left right
    '''
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
                        norm :bool = True,
                        with_padding : bool = True):
    r'''
    create 2d filters with shape of:
    (num_filters, 1, kernel_size, emb_size)
    from input `protein` with shape: 
    (seq_len, emb_size)
    params:
        kernel_size List[int]
        norm (bool) whether to apply normalization to filters
    returns:
        
    '''
    filter_stride = 1
    device = protein.device
    if kernel_size % 2 == 0:
        paddingh = kernel_size//2 - 1
        paddingw = kernel_size//2
    else:
        padding = (kernel_size - 1)//2 
        paddingh = paddingw = padding
    paddingh, paddingw = int(paddingh), int(paddingw)
    if protein.ndim == 2:
        pass
    elif protein.ndim < 3:
        raise ArithmeticError(f'protein arg must be at least 2 dim, bug given: {protein.shape}')
    seq_len, emb_size = protein.shape[-2], protein.shape[-1]
    if with_padding:
        protein = last_pad(protein, dim=-2, pad=(paddingh, paddingw))
    # protein = F.pad(protein, (0, 0, paddingh, paddingw), 'constant', 0.0)
    filter_list = []
    for start in range(0, seq_len, filter_stride):
            # single filter of size (1, kerenl_size, num_emb_feats)
            # last sample boundary when padding is false
            if start + kernel_size > seq_len:
                start = seq_len - kernel_size
            filt = protein.narrow(0, start, kernel_size).unsqueeze(0)
            filter_list.append(filt)
    filters = th.cat(filter_list, 0)
    filters = filters.view(seq_len, 1, kernel_size, emb_size)
    # normalize filters
    if norm:
        norm_val = filters.view(seq_len, -1).pow(2).sum(1, keepdim=True)
        norm_val[norm_val == 0] = 1e-5
        norm_val = norm_val.sqrt().view(seq_len, 1, 1, 1)
        filters /= norm_val
    return filters.to(device)


def calc_density(protein: th.Tensor, filters: th.Tensor,
                 stride : int = 1, with_padding: bool = True) -> th.Tensor:
    '''
    convolve `protein` with set of `filters`
    params:
        filters (num_filters, 1, kernel_size, emb_size)
    '''
    assert filters.ndim == 4, 'invalid filters shape required (num_filters, 1, kernel_size, emb_size)'
    if protein.ndim == 2:
        protein = protein.unsqueeze(0)
    elif protein.ndim == 3:
        protein = protein.unsqueeze(0).unsqueeze(0)
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
    #protein = F.pad(protein, (0, 0, paddingh, paddingw), mode='constant', value=0.0)
    density = F.conv2d(protein, filters, stride = stride)
    #output density
    density2d = density.squeeze()
    return density2d


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


def chunk_cosine_similarity(query : th.Tensor,
                            targets : List[th.Tensor],
                            quantile, dataset_files : List[str],
                            stride = 3, kernel_size = 30) -> List[dict]:
    
    assert isinstance(targets, list)
    num_targets = len(targets)
    scorestack = th.zeros(num_targets)
    assert len(targets) == len(dataset_files), f'{len(targets)} != {len(dataset_files)}'
    scorestack = chunk_score(query, targets, stride = 3, kernel_size=20)
    quantile_threshold = th.quantile(scorestack, quantile)
    scoremask = (scorestack >= quantile_threshold)
    # convert mask to indices
    filedict = {
        i : file for i, (file, condition) in enumerate(zip(dataset_files, scoremask)) if condition
        }
    return filedict

def chunk_score(query, targets, stride: int , kernel_size: int) -> th.FloatTensor:
    '''
    perform chunk cosine similarity screening
    '''
    num_targets = len(targets)
    scorestack = th.zeros(num_targets)
    embedding_dim_size = query.shape[1]
    query_kernels = sequence_to_filters(query, kernel_size=kernel_size,
                                        norm=True, with_padding=False)
    with tqdm(total=num_targets, desc='scoring embeddings') as pbar:
        for i, target in enumerate(targets, 0):
            density = calc_density(target, query_kernels,
                                   stride=stride, with_padding=False)
            target = target.unsqueeze(0).unsqueeze(0)
            target_norm = th.nn.functional.unfold(target,
                                                  (kernel_size, embedding_dim_size),
                                                  stride=(stride, 1))
            target_norm = target_norm.squeeze()
            target_norm = target_norm.pow(2).sum(0).sqrt()
            density = density/target_norm
            scorestack[i] = density.max()
            if i % 10 == 0:
                pbar.update(10)
    if scorestack.shape[0] != num_targets:
        raise ValueError(f'''cosine sim screening result different
          number of embeddings than db {scorestack.shape[0]} - {num_targets}'''
                         )
    return scorestack
