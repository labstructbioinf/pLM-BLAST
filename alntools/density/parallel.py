import sys
import os
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
from .gauss import gaussian_mean, gaussian_from_shape




def score_all_mp(embs):

    gkernel=gaussian_from_shape(20, 20)
    num_samples = len(embs)
    score_arr = th.zeros((num_samples, num_samples))
    
    kernel_iter = (11, 15, 19, 23, 27)
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
