import os
import gc
from tqdm import tqdm
import torch

from torch.nn.functional import avg_pool1d

indir = "/home/nfs/kkaminski/PLMBLST/ecod70db_20220902"

dirlist = os.listdir(indir)
print('records: ', len(dirlist))

# add preffix
filelist = [os.path.join(indir, file) for file in dirlist]
filelist = [file  for file in filelist]
# filter files
filelist = filelist[16000:]
filelist = [file for file in filelist if file.endswith('.emb.sum')]
num_files = len(filelist)
with tqdm(total = num_files) as pbar:
	for i, file in enumerate(filelist):
		emb = torch.load(file)
		if emb.dtype == torch.half:
			pass
		else:
			torch.save(emb.half(), file)
		if i % 10 == 0:
			pbar.update(3)
		del emb
		gc.collect()
