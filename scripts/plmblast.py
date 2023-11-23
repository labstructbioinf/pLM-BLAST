import sys
import os
import gc
import time
import concurrent
import datetime
from typing import List

import torch
import mkl
import numba
import pandas as pd
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


if __name__ == "__main__":

	time_start = datetime.datetime.now()
	args = get_parser()
	module = aln.base.Extractor(min_spanlen=args.WINDOW_SIZE,
							 window_size=args.WINDOW_SIZE,
							 sigma_factor=args.SIGMA_FACTOR,
							 filter_results=True,
							 bfactor='global' if args.global_aln else args.bfactor)
	module.GAP_EXT = args.GAP_EXT
	if args.verbose:
		print('%s alignment mode' % 'global' if args.global_aln else 'local')
	
	# Load database  index file
	db_index = aln.filehandle.find_file_extention(args.db)
	if args.verbose:
		print(f"Loading database {colors['yellow']}{args.db}{colors['reset']}")
	dbdf = read_input_file(db_index, plmblastid='dbid')
	# Load query
	if args.verbose:
		print(f"Loading query {colors['yellow']}{args.query}{colors['reset']}")
	# read sequence file
	query_index = aln.filehandle.find_file_extention(args.query)
	query_df = read_input_file(query_index, plmblastid='queryid')
	if os.path.isfile(args.query + '.pt'):
		query_embs: List[torch.Tensor] = torch.load(args.query + '.pt')
	elif os.path.isdir(args.query):
		query_embs = ds.parallel.load_embeddings_parallel(args.query, num_records=query_df.shape[0])
	else:
		raise FileNotFoundError(f'''
						  query embedding file or directory not found in given location: {args.query}
						  please make sure that query is appropriate directory with embeddings or file
						  with .pt extension
						  ''')
	# add id column if not present already
	# id is user based id column (typically string) to identify query
	# queryid is integer to identify search results
	query_ids = query_df['queryid'].tolist()
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
	# TODO optimize this choice
	jobs_per_process = 20
	batch_size = jobs_per_process*args.MAX_WORKERS
	query_filedict = aln.prepare.apply_database_screening(args,
													    query_embs=query_embs,
														 dbsize=dbdf.shape[0])
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
	# TODO wrapp this into context manager
	# limit threads for concurrent
	mkl.set_num_threads(1)
	numba.set_num_threads(1)
	for query_index, embedding_index, embedding_list in tqdm(batch_loader, desc='searching for alignments'):
		iter_id = 0
		query_emb = query_embs_pool[query_index]
		job_stack = dict()
		# submit jobs
		with concurrent.futures.ProcessPoolExecutor(max_workers = args.MAX_WORKERS) as executor:
			for (idx, emb) in zip(embedding_index, embedding_list):
				job = executor.submit(module.full_compare, query_emb, emb, idx)
				job_stack[job] = iter_id
				iter_id += 1
			time.sleep(0.1)
			# collect jobs
			for job in concurrent.futures.as_completed(job_stack):
				try:
					res = job.result()
					if res is not None:
						# add identifiers
						res['queryid'] = query_index
						result_stack.append(res)
				except Exception as e:
					raise AssertionError('job error:', e)	
		gc.collect()

	if len(result_stack) > 0: 
		result_df = pd.concat(result_stack, ignore_index=True)
		result_df.reset_index(inplace=True)
	else:
		print(f'No valid hits given pLM-BLAST parameters!')
		sys.exit(0)

	# Invalid plmblast score encountered
	if result_df.score.max() > 1.01:
		print(f'{colors["red"]}Error: score is greater then one{colors["reset"]}', result_df.score.min(), result_df.score.max())
		sys.exit(0)
		
	# run postprocessing
	results = list()
	result_df = result_df.merge(query_df[['queryid', 'id', 'sequence']], on='queryid', how='left')
	for qid, rows in result_df.groupby('queryid'):
		query_result = aln.postprocess.prepare_output(rows, dbdf, alignment_cutoff=args.alignment_cutoff)
		results.append(query_result)
	results = pd.concat(results, axis=0)
	# save results in desired mode
	if args.separate:
		for qid, row in results.groupby('qid'):
			row.to_csv(os.path.join(args.output, f"{qid}.csv"), sep=';')
	else:
		output_name = args.output if args.output.endswith('.csv') else args.output + '.csv'
		results.to_csv(output_name, sep=';')

	time_end = datetime.datetime.now()
	print('total hits found: ', results.shape[0])
	print(f'{colors["green"]}Done!{colors["reset"]} Time {time_end-time_start}')
