
from typing import List, Generator
import math

import numpy as np


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


def slice_iterator_with_seqlen(seqlens: List[int], resperbatch: int = 256*30):
    """
    fixed size batch
    """
    num_seq = len(seqlens)
    cumsum = np.cumsum([0]+seqlens)

    locations = [0]
    idx = 1
    for i, cums in enumerate(cumsum):

        split_condition = resperbatch * idx
        if cums > split_condition:
            locations.append(i-1)
            idx += 1

    if locations[-1] != num_seq:
         locations.append(num_seq)
    batchstart = locations[:-1]
    batchend = locations[1:]
    for bs, be in zip(batchstart, batchend):
        sl = slice(bs, be)
        yield sl