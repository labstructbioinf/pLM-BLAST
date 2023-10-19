import os
import re
from typing import List
import tempfile
import shutil

from tqdm import tqdm
import pandas as pd
import torch

from .base import save_as_separate_files
from .base import select_device
from .schema import BatchIterator
from .dataset import HDF5Handle

regex_aa = re.compile(r"[UZOB]")
EMBEDDER = 'esm2_t33_650M_UR50D'


def fsdb_wrappered_setup(embedder_name: str) -> torch.nn.Module:
	# based on https://github.com/guruace/esm2-esm-1v/blob/main/examples/esm2_infer_fairscale_fsdp_cpu_offloading.py
	# initialize the model with FSDP wrapper

	from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
	from fairscale.nn.wrap import enable_wrap, wrap

	fsdp_params = dict(
	mixed_precision=True,
	flatten_parameters=True,
	state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
	cpu_offload=True,  # enable cpu offloading
	)
	with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
		model, alphabet = torch.hub.load("facebookresearch/esm:main", embedder_name)
		batch_converter = alphabet.get_batch_converter()
		model.eval()
		# Wrap each layer in FSDP separately
		for name, child in model.named_children():
			if name == "layers":
				for layer_name, layer in child.named_children():
					wrapped_layer = wrap(layer)
					setattr(child, layer_name, wrapped_layer)
	model = wrap(model)

	return model, batch_converter


def main_esm(df: pd.DataFrame, args, iterator: BatchIterator, rank_id: int = 0):
	
	device = select_device(args)
	embedder_name = args.embedder if args.embedder != "esm" else EMBEDDER
	print('loading model: ', embedder_name)
	# giant model case
	if embedder_name == 'esm2_t48_15B_UR50D':
		model, batch_converter = fsdb_wrappered_setup(embedder_name)
	else:
		model, alphabet = torch.hub.load("facebookresearch/esm:main", embedder_name)
		batch_converter = alphabet.get_batch_converter()
		model.eval()  # disables dropout for deterministic resultsS
		model = model.to(device)
	batch_files = []
	seqlist_all = df['sequence'].tolist()
	lenlist_all = df['seqlens'].tolist()
	with tempfile.TemporaryDirectory() as tmpdirname:
		for batch_id_filename, batchslice in tqdm(iterator, total=len(iterator)):
			args.last_batch = batch_id_filename
			seqlist = seqlist_all[batchslice]
			lenlist = lenlist_all[batchslice]
			batch_index = list(range(batchslice.start, batchslice.stop))
			data = [
			(f'seq_{i}', seq)
			for i, seq in enumerate(seqlist)]
			# encode as batch
			batch_labels, batch_strs, batch_tokens = batch_converter(data)
			batch_tokens = batch_tokens.to(device=device, non_blocking=True)
			with torch.no_grad():
				repr_layers = 11
				results = model(batch_tokens, repr_layers=[repr_layers], return_contacts=False)
				#print(results['logits'].shape)
				token_representations = results["representations"][repr_layers]
				# expected size
				# [batch_size, seqlen, embdim]
				if token_representations.ndim > 3:
					token_representations = token_representations.squeeze()
				if args.gpu:
					token_representations = token_representations.to(device='cpu')
			# remove sequence padding
			num_batch_embeddings = token_representations.shape[0]
			assert num_batch_embeddings == len(seqlist)
			embeddings_filt = []
			# remove batch padding
			for i in range(num_batch_embeddings):
				seq_len = lenlist[i]
				# batch shape [args.batch_size, max_seqlen, 1280]
				emb = token_representations[i, 1 : seq_len + 1, :]
				embeddings_filt.append(emb.clone())
			if args.asdir:
				save_as_separate_files(embeddings_filt, batch_index=batch_index, directory=tmpdirname)
			elif args.h5py:
				if args.nproc == 1:
					HDF5Handle(args.output).write_batch(embeddings_filt, batch_index)
				else:
					HDF5Handle(args.output).write_batch_mp(embeddings_filt, batch_index)		
			else:
				batch_id_filename = os.path.join(tmpdirname, f"emb_{batch_id_filename}")
				torch.save(embeddings_filt, batch_id_filename)
				batch_files.append(batch_id_filename)
		# creating output
		if args.asdir:
			print(f'copying embeddings to: {args.output}')
			if not os.path.isdir(args.output):
				os.mkdir(args.output)
			for file in os.listdir(tmpdirname):
				shutil.copy(src=os.path.join(tmpdirname, file), dst=os.path.join(args.output, file))
		# merge batch_data if `asdir` is false
		else:
			stack = []
			for fname in batch_files:
				stack.extend(torch.load(fname))
			torch.save(stack, args.output)
