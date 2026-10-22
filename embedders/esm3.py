import os
import re
import gc
import tempfile
from contextlib import nullcontext

import torch
from tqdm import tqdm

from .base import select_device, save_as_separate_files
from .dataset import HDF5Handle


EMBEDDER = 'esm3_sm_open_v1'
regex_aa = re.compile(r"[UZOB]")


def embed_esm3_batch(model, seqlist, device):
	"""sequence-only ESM3 embeddings, without BOS/EOS or padding"""
	tokenizer = model.tokenizers.sequence
	ids = tokenizer(seqlist, add_special_tokens=True, padding=True,
					return_tensors='pt')['input_ids'].to(device)
	for i, seq in enumerate(seqlist):
		if int((ids[i] != tokenizer.pad_token_id).sum()) != len(seq) + 2:
			raise ValueError('ESM3 expects one token per residue and two special tokens')
	# Keep padded positions out of attention to real residues.
	sequence_id = (ids == tokenizer.pad_token_id).long()
	batch_size, length = ids.shape
	# Explicit batch dimensions are needed for the absent annotation tracks.
	function_tokens = torch.zeros((batch_size, length, 8), dtype=torch.long, device=device)
	residue_tokens = torch.zeros((batch_size, length, 16), dtype=torch.long, device=device)
	coords = torch.full((batch_size, length, 3, 3), float('nan'), device=device)
	precision = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()
	with torch.no_grad(), precision:
		result = model(sequence_tokens=ids, sequence_id=sequence_id,
					   function_tokens=function_tokens,
					   residue_annotation_tokens=residue_tokens, structure_coords=coords)
		embeddings = result.embeddings.float().cpu()
	return [embeddings[i, 1:len(seq) + 1].clone() for i, seq in enumerate(seqlist)]


def main_esm3(df, args, iterator, rank_id=0):
	"""calculate embeddings with the optional ESM3 SDK"""
	if args.use_fastt5:
		raise ValueError('--use_fastt5 is only supported for ProtT5')
	try:
		from esm.models.esm3 import ESM3
	except ImportError as e:
		raise ImportError('ESM3 requires the optional esm SDK in a compatible environment; '
						  'see README.md. fair-esm alone does not provide ESM3.') from e
	device = select_device(args)
	# Loading on CPU avoids the SDK default bfloat16 on older CUDA devices.
	model = ESM3.from_pretrained(EMBEDDER, device=torch.device('cpu'))
	dtype = torch.float16 if device.type == 'cuda' else torch.float32
	model = model.to(device=device, dtype=dtype).eval()
	print(f'model: {EMBEDDER} loaded on {device} (sequence only)')
	if args.asdir and not os.path.isdir(args.output):
		os.mkdir(args.output)
	seqlist_all = df['sequence'].tolist()
	batch_files = []
	with tempfile.TemporaryDirectory() as tmpdirname:
		for batch_id_filename, batchslice in tqdm(iterator, total=len(iterator)):
			args.last_batch = batch_id_filename
			seqlist = [regex_aa.sub('X', seq.upper()) for seq in seqlist_all[batchslice]]
			batch_index = list(range(batchslice.start, batchslice.stop))
			embeddings = embed_esm3_batch(model, seqlist, device)
			if args.asdir:
				save_as_separate_files(embeddings, batch_index=batch_index, directory=args.output)
			elif args.h5py:
				if args.nproc == 1:
					HDF5Handle(args.output).write_batch(embeddings, batch_index)
				else:
					HDF5Handle(args.output).write_batch_mp(embeddings, batch_index)
			else:
				batch_file = os.path.join(tmpdirname, f'emb_{batch_id_filename}')
				torch.save(embeddings, batch_file)
				batch_files.append(batch_file)
			del embeddings
			gc.collect()
		if not args.asdir and not args.h5py:
			stack = []
			for fname in batch_files:
				stack.extend(torch.load(fname))
			torch.save(stack, args.output)
