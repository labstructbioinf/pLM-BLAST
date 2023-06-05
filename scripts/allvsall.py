import sys
import time
import os
import argparse
import pandas as pd
import concurrent
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
	head = 100
	data = pd.read_csv(args.db + '.csv').head(head)
	dbsize = data.shape[0]
	num_combinations = dbsize * (dbsize - 1) // 2
	filelist = [os.path.join(args.db, f'{fileid}.emb') for fileid in range(0, dbsize)]
	filelist = filelist[:head]
	filedict = {i : file for i, file in enumerate(filelist)}
	#assert dbsize == len(filelist)
	embedding_list = ds.load_full_embeddings(filelist=filelist, poolfactor=4)
	embedding_list = [emb.numpy() for emb in embedding_list]
	record_stack : list = []
	step = 50

	batchiter = aln.base.BatchIterator(filedict, 20*args.MAX_WORKERS)
	print(len(batchiter))
	print('filter', module.FILTER_RESULTS)
	print('workers', args.MAX_WORKERS)
	print('samples', len(embedding_list), len(filelist))
	print('batch size', batchiter.batchsize)
	print('num batches per query', len(batchiter))
	t0 = time.perf_counter()
	with tqdm(desc='scoring', total = num_combinations) as pbar:
		iterid = 0
		for idx, (file, emb) in enumerate(zip(filelist, embedding_list)):
			for batchfiles, batchslice in batchiter:
				job_stack = {}
				with concurrent.futures.ProcessPoolExecutor(max_workers = args.MAX_WORKERS) as executor:
					# submit jobs
					for (idx1, file1), emb1 in zip(batchfiles, embedding_list[batchslice]):
							iter_id = f'{idx}-{idx1}'
							job = executor.submit(module.full_compare, emb, emb1, iter_id, file1, args.ALN_CUT)
							job_stack[job] = iter_id
							time.sleep(0.05)
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