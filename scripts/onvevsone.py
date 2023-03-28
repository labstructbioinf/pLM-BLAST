import sys
import time
import os
import argparse
import concurrent

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import alntools.density as ds
import alntools as aln
from alntools.postprocess import filter_result_dataframe


def get_parser():
	parser = argparse.ArgumentParser(description =  
		"""
		Searches a database of embeddings with a query embedding
		""",
		formatter_class=argparse.RawDescriptionHelpFormatter
		)
	parser.add_argument('db', help='Database embeddings and index (`csv` and `pt_emb.p` extensions will be added automatically)',
						type=str)
	parser.add_argument('emb', help='embeddings', type=str)																				
	parser.add_argument('output', help='Output csv file',
						type=str)
												
	parser.add_argument('-alignment_cutoff', help='Alignment score cut-off (default: %(default)s)',
						type=float, default=0.4, dest='ALN_CUT')		
						
	parser.add_argument('-sigma_factor', help='The Sigma factor defines the greediness of the local alignment search procedure. Values <1 may result in longer alignments (default: %(default)s)',
						type=float, default=1, dest='SIGMA_FACTOR')		    			    
						
	parser.add_argument('-win', help='Window length (default: %(default)s)',
						type=int, default=1, choices=range(26), metavar="[1-25]", dest='WINDOW_SIZE')				    

	parser.add_argument('-span', help='Minimal alignment length (default: %(default)s)',
						type=int, default=15, dest='MIN_SPAN_LEN')
						
	parser.add_argument('-max_targets', help='Maximal number of targets that will be reported in output (default: %(default)s)',
						type=int, default=500, dest='MAX_TARGETS')
					
	parser.add_argument('-workers', help='Number of CPU workers (default: %(default)s)',
						type=int, default=1, dest='MAX_WORKERS')			    
							
	parser.add_argument('-gap_open', help='Gap opening penalty (default: %(default)s)',
						type=float, default=0, dest='GAP_OPEN')				    
						
	parser.add_argument('-gap_ext', help='Gap extension penalty (default: %(default)s)',
						type=float, default=0, dest='GAP_EXT')

	parser.add_argument('-emb_pool', help='embedding type (default: %(default)s) ',
						type=int, default=1, dest='EMB_POOL', choices=[1, 2, 4]) 

	parser.add_argument('--no-filter', dest='filter', help='filter results',
		    	action='store_false', default=True)   

	args = parser.parse_args()
	# validate provided parameters
	assert args.MIN_SPAN_LEN >= args.WINDOW_SIZE, 'Span has to be >= window!'
	assert args.MAX_TARGETS > 0
	assert args.MAX_WORKERS > 0, 'At least one CPU core is needed!'
	return args


if __name__=='__main__':
	args = get_parser()
	module = aln.base.Extractor()
	module.SIGMA_FACTOR = args.SIGMA_FACTOR
	module.WINDOW_SIZE = args.WINDOW_SIZE
	module.GAP_OPEN = args.GAP_OPEN
	module.GAP_EXT = args.GAP_EXT
	module.BFACTOR = 1
	module.FILTER_RESULTS = args.filter
	data = pd.read_csv(args.db)
	embedding_list = np.load(args.emb)
	step = 50
	indices1 = data.idx1.tolist()
	indices2 = data.idx2.tolist()
	filedict = {idx1 : idx2 for idx1, idx2 in zip(indices1, indices2)}
	embedding_list1 = [embedding_list[i] for i in indices1]
	embedding_list2 = [embedding_list[i] for i in indices2]
	#assert dbsize == len(filelist)
	record_stack : list = []
	batchiter = aln.base.BatchIterator(filedict, 20*args.MAX_WORKERS)
	print(len(batchiter))
	print('filter', module.FILTER_RESULTS)
	print('workers', args.MAX_WORKERS)
	print('batch size', batchiter.batchsize)
	print('num batches per query', len(batchiter))
	t0 = time.perf_counter()
	with tqdm(desc='scoring', total = data.shape[0]) as pbar:
		iterid = 0
		for idx1, idx2, emb1, emb2 in zip(indices1, indices2, embedding_list1, embedding_list2):

			for batchfiles, batchslice in batchiter:
				job_stack = {}
				with concurrent.futures.ProcessPoolExecutor(max_workers = args.MAX_WORKERS) as executor:
					# submit jobs
					for (idx1, idx2), emb1, emb2 in zip(batchfiles, embedding_list1[batchslice], embedding_list2[batchslice]):
							iter_id = f'{idx1}-{idx2}'
							job = executor.submit(module.full_compare, emb1, emb2, iter_id, '', args.ALN_CUT)
							job_stack[job] = iter_id
							time.sleep(0.01)
					# gather jobs
					for job in concurrent.futures.as_completed(job_stack, timeout = 10):
						try:
							res = job.result()
							if len(res) > 0:
								record_stack.append(res)
						except Exception as e:
							raise AssertionError('job not done', e)
						if iterid % step == 0:
							pbar.update(step)
						iterid += 1

	jobtime = int(time.perf_counter() - t0)
	record_stack = pd.concat(record_stack, axis=0)
	record_stack.to_pickle(args.output)
	print(f'Job done in {jobtime} s')
	print('results: ', record_stack.shape)