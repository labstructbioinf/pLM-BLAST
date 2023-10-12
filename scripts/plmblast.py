import sys
import os
import gc
import math
import time
import argparse
import concurrent
import itertools
import datetime
from typing import List


import pandas as pd
from numba import set_num_threads
import torch
from torch.nn.functional import avg_pool1d
from tqdm import tqdm
from Bio.Align import substitution_matrices

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from alntools.parser import get_parser
from embedders.base import read_input_file
import alntools.density as ds
import alntools as aln

set_num_threads(1)
blosum62 = substitution_matrices.load("BLOSUM62")
### FUNCTIONS and helper data

# ANSI escape sequences for colors
colors = {
	'black': '\033[30m',
	'red': '\033[31m',
	'green': '\033[32m',
	'yellow': '\033[33m',
	'blue': '\033[34m',
	'magenta': '\033[35m',
	'cyan': '\033[36m',
	'white': '\033[37m',
	'reset': '\033[0m'  # Reset to default color
	}


def check_cohesion(sequences: List[str], filedict: dict, embeddings: list[torch.FloatTensor], truncate: int = 1000):
	'''
	check for missmatch between sequences and their embeddings
	'''
	for (idx,file), emb in zip(filedict.items(), embeddings):
		seqlen = len(sequences[idx])
		if seqlen < truncate:
			assert seqlen == emb.shape[0], f'''
			index and embeddings files differ, for idx {idx} seqlen {seqlen} and emb {emb.shape} file: {file}'''
		else:
			pass


def calc_con(s1, s2):
	aa_list = list('ARNDCQEGHILKMFPSTWYVBZX*')
	res=[]
	for c1, c2 in zip(list(s1), list(s2)):
		if c1=='-' or c2=='-': 
			res+=' '
			continue
		bscore = blosum62[aa_list.index(c1)][aa_list.index(c2)]
		if bscore >= 6 or c1==c2:
			res+='|'
		elif bscore >= 0:
			res+='+'
		else:
			res+='.'
	return ''.join(res)
	

def calc_similarity(s1, s2):
	def aa_to_group(aa):
		for pos, g in enumerate(['GAVLI', 'FYW', 'CM', 'ST', 'KRH', 'DENQ', 'P', '-', 'X']):
			g = list(g)
			if aa in g: return pos
		assert False
	res = [aa_to_group(c1)==aa_to_group(c2) for c1, c2 in zip(list(s1), list(s2))]
	return sum(res)/len(res)
	

def calc_ident(s1, s2):
	res = [c1==c2 for c1, c2 in zip(list(s1), list(s2))]
	return sum(res)/len(res)
	

def tensor_transform(x: torch.Tensor):
	return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def filtering_db(args: argparse.Namespace) -> dict:
	'''
	apply pre-screening
	'''
	if args.COS_PER_CUT < 100:
		query_filedict = dict()
		if args.use_chunks:
			if args.verbose:
				print('Loading database for chunk cosine similarity screening...')
			dbfile = os.path.join(args.db, 'emb.64')
			if not os.path.isfile(dbfile):
				raise FileNotFoundError('missing pooled embedding file')
			embedding_list = torch.load(dbfile)
			filelist = [os.path.join(args.db, f'{f}.emb') for f in range(0, db_df.shape[0])]  # db_df is a database index
			# TODO make avg_pool1d parallel
			query_emb_chunkcs = [avg_pool1d(emb.unsqueeze(0), 16).squeeze() for emb in query_embs]
			for i, emb in enumerate(query_emb_chunkcs):
				filedict = ds.local.chunk_cosine_similarity(
					query=emb,
					targets=embedding_list,
					quantile=args.COS_PER_CUT/100,
					dataset_files=filelist,
					stride=10)
				query_filedict[i] = filedict
		else:
			if args.verbose:
				print('Using regular cosine similarity screening...')
			for i, emb in enumerate(query_embs):
				filedict = ds.load_and_score_database(emb,
														dbpath=args.db,
														quantile=args.COS_PER_CUT/100,
														num_workers=args.MAX_WORKERS)
				filedict = {k: v.replace('.emb.sum', f'.emb{emb_type}') for k, v in filedict.items()}
				query_filedict[i] = filedict
	else:
		filelist = [os.path.join(args.db, f'{f}.emb') for f in range(0, db_df.shape[0])]  # db_df is a database index
		filedict = {k: v for k, v in zip(range(len(filelist)), filelist)}
		query_filedict = {0: filedict}
	return query_filedict


def prepare_output(args: argparse.Namespace, resdf: pd.DataFrame, query_id: str, query_seq: str) -> pd.DataFrame:
	if args.raw:
		resdf.to_pickle(args.output)
	elif len(resdf) == 0:
		print('No hits found!')
	else:
		resdf = resdf[resdf.score >= args.ALN_CUT]
		if len(resdf) == 0:
			print(f'No matches found! Try reducing the alignment_cutoff parameter. The current cutoff is {args.ALN_CUT}')
		else:
			# print('Preparing output...')
			resdf = resdf.drop(columns=['span_start', 'span_end', 'pathid', 'spanid', 'len'])            
			resdf['qid'] = query_id
			# TODO simplify above expressions
			resdf['sid'] = resdf['i'].apply(lambda i: db_df.iloc[i]['id'])
			resdf['sdesc'] = resdf['i'].apply(lambda i: db_df.iloc[i]['description'].replace(';', ' '))
			resdf['tlen'] = resdf['i'].apply(lambda i: len(db_df.iloc[i]['sequence']))
			resdf['qlen'] = len(query_seq)
			resdf['qstart'] = resdf['indices'].apply(lambda i: i[0][1])
			resdf['qend'] = resdf['indices'].apply(lambda i: i[-1][1])
			resdf['tstart'] = resdf['indices'].apply(lambda i: i[0][0])
			resdf['tend'] = resdf['indices'].apply(lambda i: i[-1][0])
			resdf['match_len'] = resdf['qend'] - resdf['qstart'] + 1

			assert all(resdf['qstart'].apply(lambda i: i <= len(query_seq) - 1))
			assert all(resdf['qend'].apply(lambda i: i <= len(query_seq) - 1))

			resdf.sort_values(by='score', ascending=False, inplace=True)
			resdf.reset_index(inplace=True)
			resdf.index = range(1, len(resdf) + 1)

			for idx, row in resdf.iterrows():
				tmp_aln = aln.alignment.draw_alignment(row.indices,
														db_df.iloc[row.i].sequence,
														query_seq,
														output='str')
				tmp_aln = tmp_aln.split('\n')
				resdf.at[idx, 'qseq'] = tmp_aln[2]
				resdf.at[idx, 'tseq'] = tmp_aln[0]
				resdf.at[idx, 'con'] = calc_con(tmp_aln[2], tmp_aln[0])
				resdf.at[idx, 'ident'] = round(calc_ident(tmp_aln[2], tmp_aln[0]), 2)
				resdf.at[idx, 'similarity'] = round(calc_similarity(tmp_aln[2], tmp_aln[0]), 2)

			resdf.drop(columns=['index', 'indices', 'i'], inplace=True)
			resdf.index.name = 'index'
			resdf = resdf[['qid', 'score', 'ident', 'similarity', 'sid', 'sdesc', 'qstart', 'qend', 'qseq', 'con',
							'tseq', 'tstart', 'tend', 'tlen', 'qlen', 'match_len']]
			resdf = resdf.head(args.MAX_TARGETS)
			if args.mqmf and not args.mqsf:
				resdf.to_csv(f"{args.output}/{query_id}.hits.csv", sep=';')
			elif args.mqsf:
				return resdf
			else:
				resdf.to_csv(args.output, sep=';')
				return False


if __name__ == "__main__":

	time_start = datetime.datetime.now()

	args = get_parser()
	module = aln.base.Extractor()
	module.FILTER_RESULTS = True
	module.WINDOW_SIZE = args.WINDOW_SIZE
	module.GAP_EXT = args.GAP_EXT
	module.SIGMA_FACTOR = args.SIGMA_FACTOR	
	EMB_POOL = 1

	if args.global_aln == 'True':
		module.BFACTOR = 'global'
		if args.verbose:
			print('Global alignment will be used')
	else:
		module.BFACTOR = 1
		if args.verbose:
			print('Local alignment will be used')

	# Load database 
	db_index = aln.filehandle.find_file_extention(args.db)
	
	if args.verbose:
		print(f"Loading database {colors['yellow']}{args.db}{colors['reset']}")

	db_df = read_input_file(db_index)
	db_df.set_index(db_df.columns[0], inplace=True)
	# Read query 
	if args.verbose:
		print(f"Loading query {colors['yellow']}{args.query}{colors['reset']}")
	query_index = aln.filehandle.find_file_extention(args.query)
	query_embs = args.query + '.pt'

	query_df = read_input_file(query_index)
	query_ids = query_df['id'].tolist()
	query_seqs = query_df['sequence'].tolist()
	
	query_embs = torch.load(query_embs)
	query_embs_pool = [
		tensor_transform(
			avg_pool1d(tensor_transform(emb).unsqueeze(0), EMB_POOL)
			).squeeze() for emb in query_embs
		]
	query_embs_pool = [emb.numpy() for emb in query_embs_pool]

	if query_df.shape[0] != len(query_embs):
		raise ValueError(f'The length of the embedding file and the sequence df are different: {query_df.shape[0]} != {len(query_embs)}')
	for q_number, (qs, qe) in enumerate(zip(query_seqs, query_embs)):
		if len(qs) != len(qe):
			raise ValueError(f'''
					The length of the embedding and the query sequence are different:
					 query index {q_number} {len(qs)} != {len(qe)}''')

	##########################################################################
	# 								filtering								 #
	##########################################################################
	query_filedict = filtering_db(args)
	num_indices_per_query = [len(vals) for vals in query_filedict.values()]
	batch_size = 20*args.MAX_WORKERS
	batch_size = min(300, batch_size)
	num_batches_per_query = [max(math.floor(nind/batch_size), 1) for nind in num_indices_per_query]

	if len(query_filedict) == 0:
		print(f'{colors["red"]}No matches after pre-filtering. Consider lowering the -cosine_percentile_cutoff{colors["reset"]}')
		sys.exit(0)

	##########################################################################
	# 								plm-blast								 #
	##########################################################################
	for query_index, (query_id, query_seq) in enumerate(zip(query_ids, query_seqs)):

		iter_id = 0
		records_stack = list()
		query_emb = query_embs_pool[query_index]

		batches = num_batches_per_query[0]
		filedict = list(query_filedict.values())[query_index]
		filelist = list(filedict.values())
		embedding_list = ds.load_full_embeddings(filelist=filelist)
		num_indices = len(embedding_list)

		for batch_start in tqdm(range(0, batches), desc='Comparison of embeddings', leave=False):
			bstart = batch_start*batch_size
			bend = bstart + batch_size
			# batch indices should not exeed num_indices
			bend = min(bend, num_indices)
			batchslice = slice(bstart, bend, 1)
			filedictslice = itertools.islice(filedict.items(), bstart, bend)
			# submit a batch of jobs
			# concurrent poolexecutor may spawn to many processes which will lead 
			# to OS error batching should fix this issue
			job_stack = {}
			with concurrent.futures.ProcessPoolExecutor(max_workers = args.MAX_WORKERS) as executor:

				for (idx, file), emb in zip(filedictslice, embedding_list[batchslice]):
					job = executor.submit(module.full_compare, query_emb, emb, idx, file)
					job_stack[job] = iter_id
					iter_id += 1
				
				time.sleep(0.1)
				for job in concurrent.futures.as_completed(job_stack):
					try:
						res = job.result()
						if len(res) > 0:
							records_stack.append(res)
					except Exception as e:
						raise AssertionError('job not done', e)	
			gc.collect()

		if records_stack: 
			resdf = pd.concat(records_stack)
		else:
			print(f'for {query_id} is 0 hits')
			continue

		if resdf.score.max() > 1:
			print(records_stack[0].score.max())
			print(f'{colors["red"]}Error: score is greater then one{colors["reset"]}', resdf.score.min(), resdf.score.max())
			sys.exit(0)

		if not args.mqsf:
			prepare_output(args, resdf, query_id, query_seq)
		elif args.mqsf:
			if "multi_query_db" in locals():
				resdf = prepare_output(args, resdf, query_id, query_seq)
				multi_query_db = pd.concat([multi_query_db, resdf], axis=0, ignore_index=True)
			else:
				multi_query_db = prepare_output(args, resdf, query_id, query_seq)

	if args.mqsf and "multi_query_db" in locals():
		if not multi_query_db.empty:
			multi_query_db.to_csv(args.output, sep=';')

	time_end = datetime.datetime.now()

	print(f'{colors["green"]}Done!{colors["reset"]} Time {time_end-time_start}')
