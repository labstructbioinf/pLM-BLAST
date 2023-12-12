import os
import re
import time
import argparse
import warnings
import tempfile
import gc

import pandas as pd
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5EncoderModel

from .dataset import HDF5Handle
from .base import save_as_separate_files, select_device
from .schema import BatchIterator
regex_aa = re.compile(r"[UZOB]")
# default embedder
DEFAULT_EMBEDDER_PT5: str = 'Rostlab/prot_t5_xl_half_uniref50-enc'
DEFAULT_EMBEDDER_PROST: str = 'Rostlab/ProstT5'
DEFAULT_DTYPE = torch.float32
DEFAULT_WAIT_TIME: float = 0.05

def main_prottrans(df: pd.DataFrame,
				    args: argparse.Namespace,
					  iterator: BatchIterator,
					  rank_id: int = 1):
	'''
	calulates embeddings for any embedding model fittable to transformer T5EncoderModel
	'''
	device = select_device(args)
	# select appropriate embedding model
	if args.embedder == 'pt':
		embedder_name = DEFAULT_EMBEDDER_PT5
	elif args.embedder == 'prost':
		embedder_name = DEFAULT_EMBEDDER_PROST
	tokenizer = T5Tokenizer.from_pretrained(embedder_name, do_lower_case=False)
	if args.use_fastt5:
		# implementation based on https://github.com/Ki6an/fastT5/issues/70
		from fastT5 import generate_onnx_representation
		from fastT5 import get_onnx_runtime_sessions
		from fastT5 import quantize, OnnxT5
		model_path = generate_onnx_representation(embedder_name)
		model_path_quant = quantize(model_path)
		model_sessions = get_onnx_runtime_sessions(model_path_quant, default=False)
		model = OnnxT5(model_path_quant, model_sessions)
	else:
		torch_dtype = torch.float16 if args.gpu else DEFAULT_DTYPE
		model = T5EncoderModel.from_pretrained(embedder_name, torch_dtype=torch_dtype)
		model.to(device)
		model.eval()
	print(f'model: {embedder_name} loaded on {device}')
	gc.collect()
	if df.seqlens.max() > 1000:
		warnings.warn('''dataset poses sequences longer then 1000 aa, this may lead to memory overload and long running time''')
	batch_files = []
	if args.asdir and not os.path.isdir(args.output):
		os.mkdir(args.output)
	seqlist_all = df['sequence'].tolist()
	lenlist_all = df['seqlens'].tolist()
	with tempfile.TemporaryDirectory() as tmpdirname:
		for batch_id_filename, batchslice in tqdm(iterator, total=len(iterator)):
			args.last_batch = batch_id_filename
			seqlist = seqlist_all[batchslice]
			lenlist = lenlist_all[batchslice]
			# add empty character between all residues
			# his is mandatory for pt5 embedders
			seqlist = [' '.join(list(seq)) for seq in seqlist]
			batch_index = list(range(batchslice.start, batchslice.stop))
			ids = tokenizer.batch_encode_plus(seqlist, add_special_tokens=True, padding="longest")
			input_ids = torch.tensor(ids['input_ids']).to(device, non_blocking=True)
			attention_mask = torch.tensor(ids['attention_mask']).to(device, non_blocking=True)
			with torch.no_grad():
				embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
				embeddings = embeddings.last_hidden_state.cpu()
			# remove sequence padding
			num_batch_embeddings = len(embeddings)
			assert num_batch_embeddings == len(seqlist)
			embeddings_filt = list()
			for i in range(num_batch_embeddings):
				seq_len = lenlist[i]
				emb = embeddings[i]
				if emb.shape[0] < seq_len:
					raise KeyError(f'sequence is longer then embedding {emb.shape} and {seq_len} ')
				# clone unpadded tensor to aviod memory issues	   
				embeddings_filt.append(emb[:seq_len].clone())
			# store each batch depending on save mode
			if args.asdir:
				save_as_separate_files(embeddings_filt, batch_index=batch_index, directory=args.output)
			elif args.h5py:
				if args.nproc == 1:
					HDF5Handle(args.output).write_batch(embeddings_filt, batch_index)
				else:
					HDF5Handle(args.output).write_batch_mp(embeddings_filt, batch_index)
			else:
				batch_id_filename = os.path.join(tmpdirname, f"emb_{batch_id_filename}")
				torch.save(embeddings_filt, batch_id_filename)
				batch_files.append(batch_id_filename)
			del embeddings
			del embeddings_filt
			gc.collect()
		# merge batch_data if `asdir` is false
		if not args.asdir and not args.h5py:
			stack = []
			for fname in batch_files:
				stack.extend(torch.load(fname))
			torch.save(stack, args.output)
