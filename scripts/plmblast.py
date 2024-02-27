import sys
import os
import gc
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
from typing import List, Dict

import torch
import mkl
import numba
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from alntools.parser import get_parser
from alntools.prepare.screening import apply_database_screening
from alntools.postprocess.format import add_duplicates
from alntools.filehandle import DataObject
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
	module = aln.Extractor( \
					enh=args.enh,
					norm=False, # legacy arg always false
					bfactor='global' if args.global_aln else args.bfactor,
					sigma_factor=args.sigma_factor,
					gap_penalty=args.gap_penalty)
	# other params
	module.FILTER_RESULTS = True
	module.MIN_SPAN_LEN = args.min_spanlen
	module.WINDOW_SIZE = args.window_size
	print("num cores: ", args.workers)

	#module.show_config()
	# Load database index file
	dbdata = DataObject.from_dir(args.db, objtype="database")
	# Load query index file
	querydata = DataObject.from_dir(args.query, objtype="query")
	# add id column if not present already
	##########################################################################
	# 								filtering								 #
	##########################################################################
	batch_size = 30*args.workers
	query_filedict = apply_database_screening(args,
													   	querydata=querydata,
														dbdata=dbdata)
	# initialize embedding iterator
	batch_loader = aln.filehandle.BatchLoader(querydata=querydata,
										    dbdata=dbdata,
											filedict=query_filedict,
											batch_size=batch_size)
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
	for query_index, embedding_index, query_emb, embedding_list in tqdm(batch_loader, desc='searching for alignments'):
		iter_id = 0
		job_stack = list()
		with ProcessPoolExecutor(max_workers = args.workers) as executor:
			for (idx, emb) in zip(embedding_index, embedding_list):
				job = executor.submit(module.full_compare, query_emb, emb, query_index, idx)
				job_stack.append(job)
			time.sleep(0.1)
			for job in as_completed(job_stack):
				try:
					res = job.result()
					if res is not None:
						result_stack.append(res)
				except Exception as e:
					raise AssertionError('job not done', e)	
		gc.collect()

	if len(result_stack) > 0: 
		result_df = pd.concat(result_stack)
		print(f'hit candidates, {result_df.shape[0]}')
	else:
		print(f'No valid hits given pLM-BLAST parameters!')
		sys.exit(0)

	# Invalid plmblast score encountered
	# only valid when signal ehancement is off
	if result_df.score.max() > 1.01 and not args.enh:
		print(f'{colors["red"]}Error: score is greater then one{colors["reset"]}', result_df.score.min(), result_df.score.max())
		sys.exit(0)
	print('merging results')
	# run postprocessing
	results: List[pd.DataFrame] = list()
	result_df = result_df.merge(
		querydata.indexdata[['run_index', 'id', 'sequence']].copy(),
		 left_on="queryid", right_on="run_index", how='left')
	for qid, rows in result_df.groupby('queryid'):
		query_result = aln.postprocess.prepare_output(rows, dbdata.indexdata, alignment_cutoff=args.alignment_cutoff)
		results.append(query_result)
	results = pd.concat(results, axis=0)
	if len(results) == 0:
		print(f'No valid hits given pLM-BLAST parameters after requested alignment cutoff {args.alignment_cutoff}!')
		sys.exit(1)
	if args.reduce_duplicates:
		results = results.reset_index(drop=True)
		results = add_duplicates(results)
	results.sort_values(by=['qid', 'score'], ascending=False, inplace=True)
	# create output directory if needed
	if os.path.dirname(args.output) != "":
		os.makedirs(os.path.dirname(args.output), exist_ok=True)
	# save results in desired mode
	if args.separate:
		for qid, row in results.groupby('qid'):
			row.to_csv(os.path.join(args.output, f"{qid}.csv"), sep=';', index=False)
	else:
		output_name = args.output if args.output.endswith('.csv') else args.output + '.csv'
		results.to_csv(output_name, sep=';', index=False)

	time_end = datetime.datetime.now()
	print('total hits found: ', results.shape[0])
	print(f'{colors["green"]}Done!{colors["reset"]} Time {time_end-time_start}')
	# stats
