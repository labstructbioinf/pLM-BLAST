import os
import sys
import argparse
from typing import List, Union, Iterable, Tuple

import numpy as np
from Bio import SeqIO
import pandas as pd
import torch

from .schema import BatchIterator
from .checkpoint import checkpoint_from_json

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
			python embeddings.py start data.csv data.pt -cname seqfull
			# for fasta input
			python embeddings.py start data.fasta data.pt
			# for file per embedding output
			python embeddings.py start data.fasta data --asdir
			# resume interrupted calculations
			python embeddings.py resume data
		""",
		formatter_class=argparse.RawDescriptionHelpFormatter
		)
	parsers = parser.add_subparsers(title='options', required=True, dest='subparser_name')
	start_group = parsers.add_parser(name='start', help='starting new calculations')
	resume_group = parsers.add_parser(name='resume', help=\
	"""
	continue calculations from checkpoint, checkpoint is automatically created and stored as
	emb_checkpoint.json in output directory this is only available when using asdir flag
	""")
	resume_group.add_argument('checkpoint', type=str, help=\
						   'previous calculation directory or checkpoint file')
	start_group.add_argument('input', help='csv/pickle (.csv or .p) with `seq` column',
						type=str)
	start_group.add_argument('output', help=\
		'''resulting list of embeddings file or directory if `asdir` is True''',
						type=str)
	start_group.add_argument('-embedder', '-e', help=\
		"""
		name of the embedder by default `pt` - prot_t5_xl_half_uniref50-enc, `esm`
		for esm2_t33_650M_UR50D or prost for ProtT5-XL-U50 you can olso specify full
		embedder name and it should be downloaded automaticaly
		""",
						dest='embedder', type=str, default='pt')
	start_group.add_argument('-cname', '-col', help='custom sequence column name',
						dest='cname', type=str, default='')
	start_group.add_argument('-r', '-head', help='number of rows from begining to use',
						dest='head', type=int, default=0)
	start_group.add_argument('-tail', help='number of rows from end to use',
						dest='tail', type=int, default=0)
	start_group.add_argument('--cuda', '--gpu', help='if specified cuda device is used default False',
						dest='gpu', default=False, action='store_true')
	start_group.add_argument('-batch_size', '-b', '-bs', help=\
		'''batch size for loader longer sequences may require lower batch size set 0 to adaptive batch mode''',
						dest='batch_size', type=int, default=32)
	store_group = start_group.add_mutually_exclusive_group()
	store_group.add_argument('--asdir', help=\
		"""
		whether save output as directory where each embedding is a separate file,
		named as df index which is mandatory for large number of sequences
		""",
		action='store_true', default=False)
	store_group.add_argument('--h5py', help=\
		"""
		output embeddings will be stored as hdf5 file with .h5
		""",
		action='store_true', default=False)
	start_group.add_argument('-truncate', '-t', default=1000, help=\
		"""
		cut sequences longer then parameter, similar to sequence[:truncate], helps to prevent OOM errors
		""",
		type=int, dest='truncate')
	start_group.add_argument('--use_fastt5', action='store_true', help=\
		"""
		experimental feature - uses https://github.com/Ki6an/fastT5 for inference speed
		""")
	start_group.add_argument('-res_per_batch', default=6000, type=int, help=\
		"""
		set the maximal number of residues in each batch, only used when batch_size is set to 0
		""")
	start_group.add_argument('--last_batch', help=argparse.SUPPRESS, type=int, default=0)
	start_group.add_argument('-nproc', '-np', help='number of process to spawn', default=1,
						  type=int)
	args = parser.parse_args()
	if args.subparser_name == 'resume':
		args = checkpoint_from_json(args.checkpoint)
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
			raise ValueError('''
					gpu is not visible by pytorch, but device is set to gpu make sure 
					that you torch package is built with gpu support''')
		if args.nproc > 1:
			if torch.cuda.device_count() < args.nproc:
				raise argparse.ArgumentError('''
								 not enough cuda visible devices requested %d available %d
								 ''' % (args.nproc, torch.cuda.device_count()))
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
	if args.res_per_batch <= 0:
		raise ValueError('res per batch must be > 0')
	
	df.reset_index(inplace=True)
	print('embedder: ', args.embedder)
	print('input file: ', args.input)
	print('sequence column: ', args.cname)
	print('device: ', 'gpu' if args.gpu else 'cpu')
	print('sequence cut threshold: ', args.truncate)
	print('save mode: ', 'directory' if args.asdir else 'file')
	print()
	return df

	
def prepare_dataframe(df: pd.DataFrame, args: argparse.Namespace, rank_id: int = 1) -> Tuple[pd.DataFrame, BatchIterator]:
		'''
		preprocess frame, if last_batch argument is supplied then iterator will start
			from [last_batch:]
		'''
		num_records: int = 0
		# prepare dataframe
		df.reset_index(inplace=True)
		num_records = df.shape[0]
		# cut sequences
		df['seqlens'] = df['seq'].apply(len)
		df['seq'] = df.apply(lambda row: \
					   row['seq'][:args.truncate] if row['seqlens'] > args.truncate else row['seq'], axis=1)
		df['seqlens'] = df['seq'].apply(len)
		batch_list = make_iterator(df['seqlens'].tolist(), args.batch_size, args.res_per_batch)
		batch_iterator = BatchIterator(batch_list=batch_list, start_batch=args.last_batch)
		if args.nproc > 1:
			batch_iterator.set_local_rank(rank=rank_id, num_rank=args.nproc)
		print('total num seq: ', num_records)
		print('num batches %d/%d' % (batch_iterator.current_batch, batch_iterator.total_batches))
		print('num batches skipped:', args.last_batch)
		print('sequence len dist: %d - %d - %d' % (df.seqlens.min(), int(df.seqlens.mean()),df.seqlens.max()))
		return df, batch_iterator


def make_iterator(seqlens: List[int], batch_size: int, res_per_batch: int) -> List[slice]:
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
		iterator = calculate_adaptive_batchsize_div4(seqlen_list=seqlens, resperbatch=res_per_batch)
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
	num_seq = len(seqlen_list)
	len_cumsum = np.cumsum(seqlen_list)
	# add zero at the begining
	endbatch_index = []
	batchend = resperbatch
	for i, csum in enumerate(len_cumsum):
		num_seq = i - batchend
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


def calculate_adaptive_batchsize_div4(seqlen_list, resperbatch: int = 4000) -> List[slice]:
	'''
	create slice iterator over sequence list with conditions
	* each batch have >= resperbatch residues
	* each batch size is dividable by 4
	Returns:
		endbatch_index: (List[slice]) iterator over start stop batch indices
	'''
	assert isinstance(seqlen_list, list)
	assert isinstance(resperbatch, int)
	assert len(seqlen_list) > 1
	assert all([n_res <= resperbatch for n_res in seqlen_list]), \
		'requested resperbatch is lower then number of residues in single sequence'
	num_seq_total = len(seqlen_list)
	step: int = 4
	# add zero at the begining
	endbatch_index = list()
	batchstart: int = 0
	batchend: int = 0
	while batchend < num_seq_total:
		batchend = batchend + step
		num_res = sum(seqlen_list[batchstart:batchend])
		num_seq = batchend - batchstart
		#print(batchend, num_res, num_seq)
		if num_res > resperbatch:
			# case when batch_size = step exeeds resperbatch
			if num_seq == step:
				batchend -= 2
				num_res = sum(seqlen_list[batchstart:batchend])
				if num_res > resperbatch:
					batchend -= 1
				endbatch_index.append(batchend)
				batchstart = batchend
			# more then 4 elements in batch
			else:
				batchend -= step
				endbatch_index.append(batchend)
				batchstart = batchend
	# add last index and 0
	startbatch_index =  [0] + endbatch_index
	endbatch_index = endbatch_index + [num_seq_total]
	batch_iterator = [slice(start, stop) for start, stop in \
					   zip(startbatch_index, endbatch_index)]
	return batch_iterator

