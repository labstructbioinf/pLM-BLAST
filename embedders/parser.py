import os
import argparse
from typing import List, Union, Iterable, Tuple

import numpy as np
from Bio import SeqIO
import pandas as pd
import torch

def create_parser() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description =
		"""
		Embedding script create embeddings from sequences via desired embedder
		by default `seq` column in used as embedder input. Records are stored
		as list maintaining dataframe order. 
		In python load via: 
		>>> import torch
		>>> torch.load(..)
		or 
		>>> import pickle
		>>> with open(.., 'rb') as f:
		>>>	embs = pickle.load(f)
		
		example use:
			python embeddings.py data.csv data.pt -cname seqfull
		""",
		formatter_class=argparse.RawDescriptionHelpFormatter
		)
	parser.add_argument('input', help='csv/pickle (.csv or .p) with `seq` column',
						type=str)
	parser.add_argument('output', help=\
		'''resulting list of embeddings file or directory if `asdir` is True''',
						type=str)
	parser.add_argument('-embedder', '-e', help=\
		"""
		type of embedder available `pt` for prot_t5_xl_half_uniref50-enc, `esm`
		for esm2_t33_650M_UR50D or prost for ProtT5-XL-U50
		""",
						dest='embedder', type=str, default='pt')
	parser.add_argument('-cname', '-col', help='custom sequence column name',
						dest='cname', type=str, default='')
	parser.add_argument('-r', '-head', help='number of rows from begining to use',
						dest='head', type=int, default=0)
	parser.add_argument('-tail', help='number of rows from end to use',
						dest='tail', type=int, default=0)
	parser.add_argument('--cuda', '--gpu', help='if specified cuda device is used default False',
						dest='gpu', default=False, action='store_true')
	parser.add_argument('-batch_size', '-b', '-bs', help=\
		'''batch size for loader longer sequences may require lower batch size set 0 to adaptive batch mode''',
						dest='batch_size', type=int, default=32)
	parser.add_argument('--asdir', '-ad', '-dir', help=\
		"""
		whether save output as directory where each embedding is a separate file,
		named as df index which is mandatory for big dataframes
		""",
		action='store_true', default=False)
	parser.add_argument('--truncate', '-t', default=1000, help=\
		"""
		cut sequences longer then parameter, helps to prevent OOM errors
		""",
		type=int, dest='truncate')
	args = parser.parse_args()
	return args


def validate_args(args: argparse.Namespace, verbose: bool = False) -> pd.DataFrame:
	'''
	handle argparse arguments
	'''
	# gather input file
	if args.input.endswith('csv'):
		df = pd.read_csv(args.input)
	elif args.input.endswith('.p') or args.input.endswith('.pkl'):
		df = pd.read_pickle(args.input)
	elif args.input.endswith('.fas') or args.input.endswith('.fasta'):
		# convert fasta file to df
		data = SeqIO.parse(args.input, 'fasta')
		# unpack
		data = [[record.description, record.seq] for record in data]
		df = pd.DataFrame(data, columns=['desc', 'seq'])
		df.set_index('desc', inplace=True)
	else:
		raise FileNotFoundError(f'invalid input infile extension {args.input}')
	# reset index for embeddings output file names
	df.index = list(range(df.shape[0]))

	if df.shape[0] == 0:
		raise AssertionError('input dataframe is empty: ', args.input)
	out_basedir = os.path.dirname(args.output)
	if out_basedir == '':
		pass
	else:
		if not args.asdir and not os.path.isdir(out_basedir):
			raise FileNotFoundError(f'output directory is invalid: {out_basedir}')
		elif args.asdir and not os.path.isdir(args.output):
			os.mkdir(args.output)

	if (args.embedder == 'pt'):
		pass
	elif args.embedder.startswith('esm'):
		pass
	elif args.embedder.startswith('prost'):
		pass
	else:
		raise ValueError("invalid embedder option", args.embedder)

	if args.cname != '':
		if args.cname not in df.columns:
			raise KeyError(f'no column: {args.cname} available in file: {args.input}, columns: {df.columns}')
		else:
			print(f'using column: {args.cname}')
			if 'seq' in df.columns and args.cname != 'seq':
				df.drop(columns=['seq'], inplace=True)
			df.rename(columns={args.cname: 'seq'}, inplace=True)

	if args.gpu:
		if not torch.cuda.is_available():
			raise ValueError('gpu is not available, but device is set to gpu and what now?')

	if args.truncate < 1:
		raise ValueError('truncate must be greater then zero')
	
	if args.head > 0:
		df = df.head(args.head)
	elif args.head < 0:
		raise ValueError('head value is negative')
	if args.tail > 0:
		df = df.tail(args.tail)
	elif args.tail < 0:
		raise ValueError('tail value is negative')
	
	
	df.reset_index(inplace=True)
	print('embedder: ', args.embedder)
	print('input frame: ', args.input)
	print('sequence column: ', args.cname)
	print('device: ', 'gpu' if args.gpu else 'cpu')
	print('sequence cut threshold: ', args.truncate)
	print('save mode: ', 'directory' if args.asdir else 'file')
	print()
	return df

	
def prepare_dataframe(df: pd.DataFrame, batch_size: int, truncate: int) -> Tuple[pd.DataFrame, List[slice]]:
		'''preprocess frame'''
		# prepare dataframe
		df.reset_index(inplace=True)
		num_records = df.shape[0]
		# cut sequences
		#df['seq'] = df['seq'].apply(lambda x : x if len(x) < 600 else x[:600])
		# stats
		df['seqlens'] = df['seq'].apply(len)
		df['seq'] = df.apply(lambda row: row['seq'][:truncate] if row['seqlens'] > truncate else row['seq'], axis=1)
		df['seqlens'] = df['seq'].apply(len)
		batch_iterator = make_iterator(df['seqlens'].tolist(), batch_size)
		num_batches = len(batch_iterator)
		print('num seq:', num_records)
		print('num batches:', num_batches)
		print(f'sequence len range {df.seqlens.min()} - {int(df.seqlens.mean())} - {df.seqlens.max()}')
		
		return df, batch_iterator


def make_iterator(seqlens: List[int], batch_size: int) -> List[slice]:
	'''
	create batch iterator over sequence lists via slices
	'''
	iterator: List[slice]
	seqnum = len(seqlens)
	# fixed batch size
	if batch_size != 0:
		startbatch = list(range(0, seqnum, batch_size))
		if startbatch[-1] != seqnum:
			startbatch += [seqnum]
		iterator = [slice(start, stop) for start, stop in zip(startbatch[:-1], startbatch[1:])]
	else:
		iterator = calculate_adaptive_batchsize(seqlen_list = seqlens)
	if len(iterator) == 0:
		raise ValueError('sequence batch iterator is empty')
	return iterator


def save_as_separate_files(embeddings: List[torch.Tensor],
						   batch_index: List[Union[str, int]],
						   directory: os.PathLike) -> List[os.PathLike]:

	assert len(embeddings) == len(batch_index)
	assert len(embeddings) > 0 and isinstance(embeddings[0], torch.Tensor)

	batch_index = [str(idx) for idx in batch_index]
	filelist = []
	for batch_i, emb_i in zip(batch_index, embeddings):
		path_i = os.path.join(directory, batch_i) + '.emb'
		torch.save(emb_i.half(), path_i)
		filelist.append(path_i)

	return filelist


def calculate_adaptive_batchsize(seqlen_list, resperbatch: int = 4000) -> Iterable:
	'''
	create slice iterator over sequence list
	Returns:
		endbatch_index: (Iterable[slice]) iterator over start stop batch indices
	'''
	assert len(seqlen_list) > 1
	len_cumsum = np.cumsum(seqlen_list)
	# add zero at the begining
	endbatch_index = []
	batchend = resperbatch
	for i, csum in enumerate(len_cumsum):
		if csum > batchend:
			endbatch_index.append(i - 1)
			# increment batch size
			batchend += resperbatch
	# add last index and 0
	startbatch_index =  [0] + endbatch_index
	endbatch_index = endbatch_index + [len(seqlen_list)]
	batch_iterator = [slice(start, stop) for start, stop in \
					   zip(startbatch_index, endbatch_index)]
	return batch_iterator
