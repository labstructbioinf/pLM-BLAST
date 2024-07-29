import sys
import os
import gc
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import datetime
from typing import List, Dict

import mkl
import numba
import pandas as pd
from tqdm import tqdm
import torch

mkl.set_num_threads(1)
numba.set_num_threads(1)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import alntools.settings as cfg
from alntools.parser import get_parser
from alntools.prepare.screening import apply_database_screening
from alntools.postprocess.format import add_duplicates
from alntools.filehandle import DataObject
import alntools as aln


if __name__ == "__main__":

	start_time = datetime.datetime.now()

	args = get_parser()
	module = aln.Extractor( \
					enh=args.enh,
					norm=False, # legacy arg always false
					bfactor='global' if args.global_aln else args.bfactor,
					sigma_factor=args.sigma_factor,
					gap_penalty=args.gap_penalty,
					min_spanlen=args.min_spanlen,
					window_size=args.window_size)
	# other params
	module.FILTER_RESULTS = True
	print("num cores: ", args.workers)

	#module.show_config()
	# Load database index file
	dbdata = DataObject.from_dir(args.db, objtype="database")
	# Load query index file
	querydata = DataObject.from_dir(args.query, objtype="query")
	# add id column if not present already
	##########################################################################
	# 						filtering (reduce search space)					 #
	##########################################################################
	batch_size = cfg.jobs_per_process*args.workers
	query_filedict = apply_database_screening(args, querydata=querydata, dbdata=dbdata)
	if args.only_scan:
		# round float values in json 
		# https://stackoverflow.com/questions/54370322/how-to-limit-the-number-of-float-digits-jsonencoder-produces
		class RoundingFloat(float):
			__repr__ = staticmethod(lambda x: format(x, '.3f'))
		json.encoder.float = RoundingFloat
		with open(args.output, "wt") as fp:
			json.dump(query_filedict, fp)
		sys.exit(0)
	else:
		# simplify dictionary
		query_filedict = {
        	queryid: { targetid: value['file']
            	for targetid, value in outer_dict.items()
        	}
        	for queryid, outer_dict in query_filedict.items()
    	}
	# initialize embedding iterator
	batch_loader = aln.filehandle.BatchLoader(querydata=querydata,
										      dbdata=dbdata,
											  filedict=query_filedict,
											  batch_size=batch_size)
	if len(query_filedict) == 0:
		print(f'{cfg.colors["red"]}No matches after pre-filtering. Consider lowering the -cosine_percentile_cutoff{cfg.colors["reset"]}')
		sys.exit(1)
	##########################################################################
	# 						main   plm-blast loop							 #
	##########################################################################
	# limit threads for concurrent
	mkl.set_num_threads(1)
	numba.set_num_threads(1)
	tmpresults: List[str] = list()
	with tempfile.TemporaryDirectory() as tmpdir, tqdm(total=len(batch_loader), desc='searching for alignments') as pbar:
		for itr, (query_index, embedding_index, query_emb, embedding_list) in enumerate(batch_loader):
			job_stack = list()
			result_stack: List[pd.DataFrame] = list()
			with ProcessPoolExecutor(max_workers = args.workers) as executor:
				for (idx, emb) in zip(embedding_index, embedding_list):
					job = executor.submit(module.full_compare, query_emb, emb, query_index, idx)
					job_stack.append(job)
				time.sleep(0.05)
				for job in as_completed(job_stack):
					try:
						res = job.result()
						if res is not None:
							result_stack.append(res)
					except Exception as e:
						raise AssertionError('job failed', e)	
				if len(result_stack) > 0:
					tmpfile = os.path.join(tmpdir, f"{itr}.p")
					pd.concat(result_stack).to_pickle(tmpfile)
					tmpresults.append(tmpfile)
			gc.collect()
			pbar.update(1)
		if len(tmpresults) > 0: 
			result_df = pd.concat([pd.read_pickle(f) for f in tmpresults])
			print(f'hit candidates, {result_df.shape[0]}')
		else:
			print(f'No valid alignemnts for given pLM-BLAST parameters! Consider changing input parameters and validate your inputs')
			sys.exit(0)
	# Invalid plmblast score encountered
	# only valid when signal ehancement is off
	if result_df.score.max() > 1.01 and not args.enh:
		print(f'{cfg.colors["red"]}Error: score is greater then one{cfg.colors["reset"]}', result_df.score.min(), result_df.score.max())
		sys.exit(1)
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
		sys.exit(0)
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

	script_time = datetime.datetime.now() - start_time
	print('total hits found: ', results.shape[0])
	print(f'{cfg.colors["green"]}Done!{cfg.colors["reset"]}\nTime {script_time}')
	# stats
