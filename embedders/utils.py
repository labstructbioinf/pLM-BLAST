
from typing import List

import torch
from torch.nn.functional import avg_pool1d

def pool_batch(emb_batch: List[torch.Tensor], factor: int = 2) -> List[torch.Tensor]:
    
    if not isinstance(factor, int):
        raise TypeError("pool factor must be integer, got: {type(factor)}")
    emb_pooled = list()
    for emb in emb_batch:
        emb_pooled.append(avg_pool1d(emb, kernel_size=factor))
    return emb_pooled