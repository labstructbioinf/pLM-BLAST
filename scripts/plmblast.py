import sys
import os
import gc
import datetime
from typing import List
import multiprocessing
from functools import partial

import mkl
import numba
import pandas as pd
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from alntools.parser import get_parser
from alntools.filehandle import DataObject
from alntools.base import Extractor
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
	module = Extractor(min_spanlen=args.min_spanlen,
							 window_size=args.WINDOW_SIZE,
							 sigma_factor=args.SIGMA_FACTOR,
							 filter_results=True,
							 bfactor='global' if args.global_aln else args.bfactor,
							 enhance_signal=args.enh)
	module.GAP_EXT = args.GAP_EXT
	module.NORM = False
	module.show_config()
	# Load database index file
	dbdata = DataObject.from_dir(args.db, objtype="database")
	# Load query index file
	querydata = DataObject.from_dir(args.query, objtype="query")
	# add id column if not present already
	##########################################################################
	# 								filtering								 #
	##########################################################################
	batch_size = aln.cfg.jobs_per_process*args.workers
	query_filedict = aln.prepare.apply_database_screening(args,
													   	querydata=querydata,
														dbdata=dbdata)
	# initialize embedding iterator
	batch_loader = aln.filehandle.BatchLoader(querydata=querydata,
										    dbdata=dbdata,
											filedict=query_filedict,
											batch_size=batch_size)
	##########################################################################
	# 								plm-blast								 #
	##########################################################################
	# TODO wrapp this into context manager
	# limit threads for concurrent
	os.environ["MKL_DYNAMIC"] = str(False)
	os.environ["OMP_NUM_THREADS"] = str(args.workers)
	mkl.set_num_threads(1)
	numba.set_num_threads(1)
	torch.set_num_threads(1)
	print(torch.get_num_threads()
	print(multiprocessing.cpu_count()))
	with multiprocessing.Manager() as manager:
		result_stack: List[pd.DataFrame] = manager.list()
		compare_fn = partial(module.full_compare_args, result_stack=result_stack)
		for query_index, embedding_index, query_emb, embedding_list in tqdm(batch_loader, desc='searching for alignments'):
			# submit jobs
			with multiprocessing.Pool(processes=args.workers) as pool:
				iterable = ((query_emb, emb, db_index, query_index) for db_index, emb in zip(embedding_index, embedding_list))
				job_stack = pool.map_async(compare_fn, iterable, chunksize=aln.cfg.mp_chunksize)
				# collect jobs
				job_stack.wait()
			gc.collect()
		results_stack = list(result_stack)
		if len(result_stack) > 0: 
			result_df = pd.concat(result_stack, ignore_index=True)
			result_df.reset_index(inplace=True)
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
	results = list()
	result_df = result_df.merge(
		querydata.indexdata[['run_index', 'id', 'sequence']],
		 left_on="queryid", right_on="run_index", how='left')
	for qid, rows in result_df.groupby('queryid'):
		query_result = aln.postprocess.prepare_output(rows, dbdata.indexdata, alignment_cutoff=args.alignment_cutoff)
		results.append(query_result)
	results = pd.concat(results, axis=0)
	if len(results) == 0:
		print(f'No valid hits given pLM-BLAST parameters after requested alignment cutoff {args.alignment_cutoff}!')
		sys.exit(0)
	
	results.sort_values(by=['qid', 'score'], ascending=False, inplace=True)
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
