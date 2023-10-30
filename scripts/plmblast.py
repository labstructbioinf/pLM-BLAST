import sys
import os
import gc
import time
import argparse
import concurrent
import datetime
from typing import List, Dict

import torch
import mkl
import numba
import pandas as pd
from torch.nn.functional import avg_pool1d
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from alntools.parser import get_parser
from embedders.base import read_input_file
import alntools.density as ds
import alntools as aln


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


def check_cohesion(sequences: List[str],
				   filedict: dict,
				   embeddings: List[torch.FloatTensor],
				   truncate: int = 1000):
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


def filtering_db(args: argparse.Namespace, query_embs: List[torch.Tensor]) -> Dict[int, List[str]]:
	'''
	apply pre-screening
	Returns:
		(dict) each key is query_id, and values are embeddings above threshold
	'''
	assert len(query_embs) > 0
	assert isinstance(query_embs, list)
	# set torch num CPU limit
	torch.set_num_threads(args.MAX_WORKERS)
	num_queries = len(query_embs)
	if args.COS_PER_CUT < 100:
		query_filedict = dict()
		if args.use_chunks:
			if args.verbose:
				print('Loading database for chunk cosine similarity screening...')
			dbfile = os.path.join(args.db, 'emb.64')
			if not os.path.isfile(dbfile):
				raise FileNotFoundError('missing pooled embedding file emb.64')
			embedding_list: List[torch.Tensor] = torch.load(dbfile)
			dbsize = db_df.shape[0]
			filelist = [os.path.join(args.db, f'{f}.emb') for f in range(0, dbsize)]
			# TODO make avg_pool1d parallel
			# conver to float 
			query_emb_chunkcs = [emb.float() for emb in query_embs]
			# pool
			query_emb_chunkcs = [avg_pool1d(emb.unsqueeze(0), 16).squeeze() for emb in query_embs]
			# loop over all query embeddings
			for i, emb in tqdm(enumerate(query_emb_chunkcs), total=num_queries, desc='screening seqences'):
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
		filelist = [os.path.join(args.db, f'{f}.emb') for f in range(0, dbsize)]  # db_df is a database index
		filedict = {k: v for k, v in zip(range(len(filelist)), filelist)}
		query_filedict = {0: filedict}
	return query_filedict


if __name__ == "__main__":

	time_start = datetime.datetime.now()

	args = get_parser()
	module = aln.base.Extractor()
	module.FILTER_RESULTS = True
	module.WINDOW_SIZE = args.WINDOW_SIZE
	module.GAP_EXT = args.GAP_EXT
	module.SIGMA_FACTOR = args.SIGMA_FACTOR
	module.BFACTOR = 'global' if args.global_aln else 1
	if args.verbose:
		print('%s alignment mode' % 'global' if args.global_aln else 'local')
	
	# Load database 
	db_index = aln.filehandle.find_file_extention(args.db)
	if args.verbose:
		print(f"Loading database {colors['yellow']}{args.db}{colors['reset']}")
	db_df = read_input_file(db_index)
	db_df.set_index(db_df.columns[0], inplace=True)

	# Load query
	if args.verbose:
		print(f"Loading query {colors['yellow']}{args.query}{colors['reset']}")
	# read sequence file
	query_index = aln.filehandle.find_file_extention(args.query)
	query_df = read_input_file(query_index)
	if os.path.isfile(args.query + '.pt'):
		query_embs: List[torch.Tensor] = torch.load(args.query + '.pt')
	elif os.path.isdir(args.query):
		query_embs_files = filelist = [os.path.join(args.query, f'{f}.emb') for f in range(0, query_df.shape[0])]
		query_embs: List[torch.Tensor] = ds.load_full_embeddings(query_embs_files, poolfactor=1)
	else:
		raise FileNotFoundError(f'''
						  query embedding file or directory not found in given location: {args.query}
						  please make sure that query is appropriate directory with embeddings or file
						  with .pt extension
						  ''')
	# add id column
	if 'id' not in query_df.columns:
		query_df['id'] = list(range(0, query_df.shape[0]))
	else:
		query_df['id_tmp'] = query_df['id'].copy()
		query_df['id'] = list(range(0, query_df.shape[0]))
	query_ids = query_df['id'].tolist()
	query_seqs = query_df['sequence'].tolist()
	query_embs_pool = [emb.float().numpy() for emb in query_embs]

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
	batch_size = 20*args.MAX_WORKERS
	# TODO optimize this choice
	batch_size = min(300, batch_size)
	query_filedict = filtering_db(args, query_embs)
	batch_loader = aln.filehandle.BatchLoader(query_ids=query_ids,
										    query_seqs=query_seqs,
											filedict=query_filedict,
											batch_size=batch_size,
											mode="emb")
	if len(query_filedict) == 0:
		print(f'{colors["red"]}No matches after pre-filtering. Consider lowering the -cosine_percentile_cutoff{colors["reset"]}')
		sys.exit(0)
	##########################################################################
	# 								plm-blast								 #
	##########################################################################
	result_stack: List[pd.DataFrame] = list()
	# limit threads for concurrent
	mkl.set_num_threads(1)
	numba.set_num_threads(1)
	for query_index, embedding_index, embedding_list in tqdm(batch_loader, desc='searching for alignments'):
		iter_id = 0
		query_emb = query_embs_pool[query_index]
		job_stack = {}
		with concurrent.futures.ProcessPoolExecutor(max_workers = args.MAX_WORKERS) as executor:
			for (idx, emb) in zip(embedding_index, embedding_list):
				job = executor.submit(module.full_compare, query_emb, emb, idx)
				job_stack[job] = iter_id
				iter_id += 1
			time.sleep(0.1)
			for job in concurrent.futures.as_completed(job_stack):
				try:
					res = job.result()
					if len(res) > 0:
						# add identifiers
						res['id'] = query_index
						result_stack.append(res)
				except Exception as e:
					raise AssertionError('job not done', e)	
		gc.collect()

	if len(result_stack) > 0: 
		result_df = pd.concat(result_stack)
	else:
		print(f'No valid hits given pLM-BLAST parameters!')
		sys.exit(0)

	# Invalid plmblast score encountered
	if result_df.score.max() > 1.01:
		print(f'{colors["red"]}Error: score is greater then one{colors["reset"]}', result_df.score.min(), result_df.score.max())
		sys.exit(0)
		
	# run postprocessing
	results = list()
	for qid, rows in result_df.groupby('id'):
		query_result = aln.postprocess.prepare_output(args, rows, qid, query_seqs[qid], db_df)
		query_result['id'] = qid
		results.append(query_result)
	results = pd.concat(results)

	# save results in desired mode
	if args.separate:
		for qid, row in results.groupby('id'):
			row.to_csv(os.path.join(args.output, f"{qid}.csv"), sep=';')
	else:
		output_name = args.output if args.output.endswith('.csv') else args.output + '.csv'
		results.to_csv(output_name, sep=';')

	time_end = datetime.datetime.now()
	print(f'{colors["green"]}Done!{colors["reset"]} Time {time_end-time_start}')
