import sys
import os
from typing import List
from pathlib import Path
from functools import partial
from collections import namedtuple

from tqdm import tqdm 
import pandas as pd
import torch as th
from torch.nn.functional import max_pool2d, avg_pool2d
import numpy as np

sys.path.append('../..')
from .local import get_multires_density
from .search_signal import smooth_image, smooth_signal, peak_width
#import graph_toolbox as gt
#from network_utils.dataloader import graph_from_pdb, setup_dataset_and_loader
from .region import refactor_2d_mask_to_diagonals, check_colinearity, span_to_matrix, diff_tensor
from .gauss import gaussian_mean, gaussian_from_shape, diagonal_from_shape

protein_letters = 'ACDEFGHIKLMNPQRSTVWY' + 'X'
residue_to_num_dict = {res : num for num, res in enumerate(protein_letters)}
num_to_residue_dict = {v : k for k, v in residue_to_num_dict.items()}
similarity_template = namedtuple('similarity_template', ['seq1_resid', 'seq2_resid', 'density', 'seq1_density', 'seq2_density', 'field', 'max_lens'])
similarity_1d = namedtuple('similarity_1d', ['resid', 'density1d'])
    
    
def score_all(embs):
    gkernel = gaussian_from_shape(20, 20)
    num_samples = len(embs)
    score_arr = th.zeros((num_samples, num_samples))
    kernel_iter = (11, 13)
    with th.no_grad():
        for i, x in tqdm(enumerate(embs)):
            for j, y in enumerate(embs):
                if j > i:
                    break
                score = get_multires_density(x, y, kernel_iter=kernel_iter)
                score_avg = gaussian_mean(score, kernel=gkernel).max()
                score_arr[i, j] = score_avg
                score_arr[j, i] = score_avg
    return score_arr

def one_vs_all_cuda(ref, queries):
    
    assert isinstance(ref, (th.FloatTensor, th.cuda.FloatTensor))
    assert isinstance(queries, list)
    
    device = th.device('cuda:0')
    ref = ref if ref.device == device else ref.to(device)
    gkernel=gaussian_from_shape(20, 20)
    num_samples = len(queries)
    arr_list = list()
    kernel_iter = (11, 13)
    
    with th.no_grad(): 
        for j, que in tqdm(enumerate(queries)):
            que = que if que.device == device else que.to(device)
            score = get_multires_density(ref, que, kernel_iter=kernel_iter)
            arr_list.append(score.cpu())
    return arr_list

def score_xy(embs1 : List[th.Tensor],
             embs2=List[th.Tensor],
             device: th.device = th.device('cuda:0')) -> th.Tensor:
    '''
    params:
        embs1 list of embeddings
        embs2 list of embeddings
        device torch device
    return:
        score arr
    
    '''
    num_samples1, num_samples2 = len(embs1), len(embs2)
    score_arr = th.zeros((num_samples1, num_samples2),
                         dtype=th.float, device=device)
    kernel_iter = (11, 13)
    filt = diagonal_from_shape(13, 13, device=device)
    with th.no_grad():
        for i, x in tqdm(enumerate(embs1)):
            for j, y in enumerate(embs2):
                if j == i:
                    break
                arr = get_multires_density(x, y)
                score_mx = gaussian_mean(arr, filt).max()
                score_arr[i, j] = score_mx
    return score_arr


def func(tp):
    x, y = tp
    return get_multires_density(x.cuda(), y.cuda(), kernel_iter=(11, 13))


def score_all_cuda_mp(embs):
    device = th.device('cuda:0')
    num_samples = len(embs)
    score_arr = th.zeros((num_samples, num_samples),
                         dtype=th.float, device=device)
    kernel_iter = (11, 13)
    filt = diagonal_from_shape(13, 13).cuda()
    num_streams = 10
    streams = [th.cuda.Stream() for _ in range(num_samples)]
    th.cuda.synchronize()
    for i, x in tqdm(enumerate(embs)):
        x = x if x.device == device else x.to(device)
        with th.cuda.stream(streams[i]):
            for j, y in enumerate(embs):
                if j > i:
                    break
                y = y if y.device == device else y.to(device)
                arr = get_multires_density(x, y, kernel_iter=kernel_iter)
                score_mx = gaussian_mean(arr, filt).max()
                score_arr[i, j] = score_mx
                score_arr[j, i] = score_mx
    return score_arr


class SearchSequences:
    kernel_sizes = range(11, 33, 4)
    def __init__(self):
        pass
    
    def calculate(self, reference, query_list):
        '''
        reference (1, num_residues)
        query list of tensors (1, num_residues)
        '''
        quantile_stack = list()
        seq_lens = list()
        density_ref = get_multires_density(reference, reference, kernel_iter=self.kernel_sizes)
        kernel_20 = gaussian_from_shape(20, 20)
        kernel_30 = gaussian_from_shape(30, 30)
        kernel_40 = gaussian_from_shape(40, 40)
        mean_20 = gaussian_mean(density_ref, kernel_20).max()
        mean_30 = gaussian_mean(density_ref, kernel_30).max()
        mean_40 = gaussian_mean(density_ref, kernel_40).max()
        means_ref = th.tensor([mean_20, mean_30, mean_40]).unsqueeze(0)
        
        with th.no_grad():
            for query_sample in query_list:
                density = get_multires_density(reference, query_sample, kernel_iter=self.kernel_sizes)
                shape_ratio = density.shape[0]/density.shape[1]
                x_filt_shape_20 = int(20*shape_ratio)
                x_filt_shape_30 = int(30*shape_ratio)
                x_filt_shape_40 = int(40*shape_ratio)
                kernel_20 = gaussian_from_shape(x_filt_shape_20, 20)
                kernel_30 = gaussian_from_shape(x_filt_shape_30, 30)
                kernel_40 = gaussian_from_shape(x_filt_shape_40, 40)
                mean_20 = gaussian_mean(density, kernel_20).max()
                mean_30 = gaussian_mean(density, kernel_30).max()
                mean_40 = gaussian_mean(density, kernel_40).max()

                means = th.tensor([mean_20, mean_30, mean_40]).unsqueeze(0)
                quantile_stack.append(means)
                seq_lens.append(query_sample.shape[0])
            
            quantile_stack = th.cat(quantile_stack, dim=0)
            quantile_stack_factor = (quantile_stack/means_ref).numpy()
            qframe = pd.DataFrame(quantile_stack_factor, columns=[f'sim_{q}' for q in [20, 30, 40]])
            qframe['len'] = seq_lens
        return qframe
    
    def calculate_all_vs_all(self, reference, query_list):
        '''
        reference (1, num_residues)
        query list of tensors (1, num_residues)
        '''    
        with th.no_grad():
            quantile_stack = list()
            seq_lens = list()
            for reference in query_list:
                density_ref = get_multires_density(reference, reference, kernel_iter=self.kernel_sizes)
                kernel_20 = gaussian_from_shape(20, 20)
                kernel_30 = gaussian_from_shape(30, 30)
                kernel_40 = gaussian_from_shape(40, 40)
                mean_20 = gaussian_mean(density_ref, kernel_20).max()
                mean_30 = gaussian_mean(density_ref, kernel_30).max()
                mean_40 = gaussian_mean(density_ref, kernel_40).max()
                means_ref = th.tensor([mean_20, mean_30, mean_40]).unsqueeze(0)
                for query_sample in query_list:
                    density = get_multires_density(reference, query_sample, kernel_iter=self.kernel_sizes)
                    shape_ratio = density.shape[0]/density.shape[1]
                    x_filt_shape_20 = int(20*shape_ratio)
                    x_filt_shape_30 = int(30*shape_ratio)
                    x_filt_shape_40 = int(40*shape_ratio)
                    kernel_20 = gaussian_from_shape(x_filt_shape_20, 20)
                    kernel_30 = gaussian_from_shape(x_filt_shape_30, 30)
                    kernel_40 = gaussian_from_shape(x_filt_shape_40, 40)
                    mean_20 = gaussian_mean(density, kernel_20).max()
                    mean_30 = gaussian_mean(density, kernel_30).max()
                    mean_40 = gaussian_mean(density, kernel_40).max()
                    means = th.tensor([mean_20, mean_30, mean_40]).unsqueeze(0)
                    quantile_stack.append(means)
                    seq_lens.append(query_sample.shape[0])
            quantile_stack = th.cat(quantile_stack, dim=0)
            quantile_stack_factor = (quantile_stack/means_ref).numpy()
            qframe = pd.DataFrame(quantile_stack_factor, columns=[f'sim_{q}' for q in [20, 30, 40]])
            qframe['len'] = seq_lens
        return qframe
    
        
class LocalRegionAlign:
    '''
    class for extracting similarity from 2d binary masks
    '''
    def __init__(self, arr, arr_density, area_size=40, verbose=False):
        
        #arr = LocalRegionAlign.preprocess(arr)
        shifts_arr = th.arange(-arr.shape[0], arr.shape[1], 1)
        shifts_dict = refactor_2d_mask_to_diagonals(arr)
        shifts_used = th.Tensor(list(shifts_dict.keys()))
        positions_raw = list(shifts_dict.values())
        max_lens_list = [(r[:, 1] - r[:, 0]).max() for r in shifts_dict.values()]
        max_lens = th.tensor(max_lens_list)
        max_lens_smoothed = smooth_signal(max_lens, 5).long()
        indices_diagonal =  gt.tensor_ops.offdiagonal_indices(arr)
        shifts_best = th.nonzero(max_lens > area_size).flatten()
        '''
        shifts_sparse = shifts_best.clone()
        shifts_sparse = th.nonzero(diff_tensor(shifts_sparse) > 3).flatten() + th.tensor([1]).long()
        shifts_sparse = th.cat((th.tensor([0]), shifts_sparse))
        shifts_best = shifts_best[shifts_sparse]
        '''
        regions = list()
        for shift_idx in shifts_best:

            span_start, span_stop = peak_width(max_lens_smoothed, shift_idx)
            span_as_diag_index =  th.arange(span_start-shift_idx, span_stop-shift_idx)
            if verbose:
                print('idx: ', shift_idx, 'reg size:', max_lens_smoothed[shift_idx], 'span:', span_start, span_stop)
            max_span = positions_raw[shift_idx]
            if max_span.shape[0] > 1: #find max span
                diff = max_span[:, 1] - max_span[:, 0]
                reference = max_span[diff == diff.max(), :]
            else:
                reference = max_span
                
            diagonal_id = shifts_used[shift_idx]
            diagonal_idx = th.nonzero(shifts_arr == diagonal_id).squeeze().item()
            reference_dynamic = reference.clone()
            matched = list()
            
            for shift in (diagonal_idx + span_as_diag_index).long():
                
                diagonal_val_tmp = indices_diagonal[shift]
                shift_val = shifts_arr[shift].item()
                if shift_val not in shifts_dict:
                    continue
                samples = shifts_dict[shift_val]
                matches = check_colinearity(samples, reference)
                if matches is not None:
                    reference_dynamic = th.cat((reference_dynamic, matches), 0)
                    for match in matches:
                        indices = span_to_matrix(match, diagonal_val_tmp)
                        matched.append(indices)
                else:
                    continue
            region_area_ids = th.cat(matched)
            field = region_area_ids.shape[0]
            x, y = region_area_ids.split(1, 1)
            x, y = x.squeeze(), y.squeeze()
            x_sorted, indices = th.sort(x)
            y_sorted = y[indices]
            region_density = arr_density[x_sorted, y_sorted]
            x1, y1 = self.partial_density(x_sorted, region_density)
            x2, y2 = self.partial_density(y_sorted, region_density)
            seq1_density = similarity_1d(x1, y1)
            seq2_density = similarity_1d(x2, y2)
            regions.append(similarity_template(x_sorted,
                                               y_sorted,
                                               region_area_ids,
                                              seq1_density,
                                              seq2_density,
                                              field=field,
                                              max_lens=max_lens))
        self.regions = regions
    
    @classmethod
    def from_raw_density(cls, density, condition, area_size=40, verbose=False):
        '''
        init with preprocessing (downsampling)
        '''
        density_mask = density > condition
        density_mask = cls._preprocess_mask(density_mask)
        
        density = cls._preprocess_density(density)
        
        return cls(density_mask, density, area_size=area_size, verbose=verbose)
    
    @classmethod
    def _preprocess_mask(self, arr):
        '''
        logical downsampling as OR
        '''
        if isinstance(arr, th.BoolTensor):
            arr = arr.float()
        if arr.ndim == 2:
            arr = arr.unsqueeze(0)
        img = max_pool2d(arr, (2,2)).squeeze()
        return img.bool()
    
    @classmethod
    def _preprocess_density(self, arr):
        '''
        preprocess 2d density
        '''
        if arr.ndim == 2:
            arr = arr.unsqueeze(0)
        image = avg_pool2d(arr, (2,2)).squeeze()
        return image
        
    def widest_area(self):
        
        sizes = dict()
        for i, region in enumerate(self.regions):
            sizes[i] = region.field
        size_max = max(sizes.values())
        size_argmax = [i for k,v in sizes.items() if v == size_max]
        
        return self.regions[size_argmax[0]]
                
                
    def partial_density(self, indices, density):
        '''
        calculate 1d density distribution (unnormed) from axis mask 
        indices and 2d density
        '''
        amplitude = th.bincount(indices, density)
        #print(amplitude)
        bins, counts = th.unique(indices, return_counts=True)
        #remove zeros from bincount output
        amplitude = amplitude[th.nonzero(amplitude)].squeeze()
        density_1d = amplitude.float()/counts.float()
        
        return bins, density_1d