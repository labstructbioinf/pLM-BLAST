import sys
import os
import gc
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import datetime
from typing import List, Dict

import torch
import torch.multiprocessing as mp
import mkl
import numba
import pandas as pd
from tqdm import tqdm

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
					window_size=args.window_size,
					gpu_support=args.gpu)
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
	# initialize embedding iterator
	batch_loader = aln.filehandle.BatchLoader(querydata=querydata,
										      dbdata=dbdata,
											  filedict=query_filedict,
											  batch_size=batch_size,
											  gpu_support=args.gpu)
	if len(query_filedict) == 0:
		print(f'{cfg.colors["red"]}No matches after pre-filtering. Consider lowering the -cosine_percentile_cutoff{cfg.colors["reset"]}')
		sys.exit(1)
	##########################################################################
	# 						main   plm-blast loop							 #
	##########################################################################
	result_stack: List[pd.DataFrame] = list()
	# limit threads for concurrent
	mkl.set_num_threads(1)
	numba.set_num_threads(1)
	tmpresults: List[str] = list()
	my_device = "cuda"

	def process_task(query_index, idx, density):
		"""
		Processes a single task by comparing the density matrix using a GPU.

		Parameters:
		- query_index: Index of the query
		- idx: Index of the embedding
		- density: Density matrix to be compared

		Returns:
		- Result of the comparison
		"""
		try:
			return module.full_compare_gpu(query_index, idx, density)
		except Exception as e:
			raise AssertionError('Job not done', e)


	def compute_density(qe, te, enh=False):
		"""
		Computes the density matrix between two sets of embeddings.

		Parameters:
		- qe: tensor of shape (num_residues, embedding_dim)
		- te: tensor of shape (num_residues, embedding_dim)
		- enhance: boolean flag to indicate if density enhancement should be applied

		Returns:
		- density: numpy array of the computed density matrix
		"""
		if qe.shape[1] != te.shape[1]:
			raise ValueError(f"Shape mismatch: qe.shape[1] ({qe.shape[1]}) != te.shape[1] ({te.shape[1]})")
		
		assert qe.ndim == 2 and te.ndim == 2, 'Input tensors must have 2 dimensions [num residues, embedding dim]'
		assert qe.shape[1] == te.shape[1], f'Embedding size is different for qe, Y - {qe.shape[1]} and {te.shape[1]}'

		# Normalize the embeddings
		emb1_normed = qe / torch.linalg.norm(qe, dim=1, keepdim=True)
		emb2_normed = te / torch.linalg.norm(te, dim=1, keepdim=True)

		if emb1_normed.shape[1] != emb2_normed.shape[1]:
			raise ValueError(f"Shape mismatch: emb1_normed.shape[1] ({emb1_normed.shape[1]}) != emb2_normed.shape[1] ({emb2_normed.shape[1]})")

		density = torch.matmul(emb1_normed, emb2_normed.T).T

		if enh:
			density_mean_0 = torch.mean(density, dim=0, keepdim=True)
			density_std_0 = torch.std(density, dim=0, keepdim=True)
			density_left = (density - density_mean_0) / density_std_0

			density_mean_1 = torch.mean(density, dim=1, keepdim=True)
			density_std_1 = torch.std(density, dim=1, keepdim=True)
			density_right = (density - density_mean_1) / density_std_1

			density = (density_left + density_right) / 2

		return density.float().cpu().numpy()


	with tempfile.TemporaryDirectory() as tmpdir:
		# Iterate through the batch loader to process query and embedding pairs
		for itr, (query_index, embedding_index, query_emb, embedding_list) in enumerate(tqdm(batch_loader, desc='Searching for alignments')):
			job_stack = list()
			result_stack: List[pd.DataFrame] = list()

			if args.gpu:

				# Load the query embedding tensor
				qe = torch.load(query_emb, map_location=my_device)
				
				# Use a multiprocessing pool to handle concurrent tasks
				with mp.Pool(processes=args.workers) as pool:
					for emb, idx in zip(embedding_list, embedding_index):
						# Load the target embedding tensor
						te = torch.load(emb, map_location=my_device)
						
						# Compute the density matrix
						density = compute_density(qe, te, args.enh)

						# Apply the task to the pool
						job = pool.apply_async(process_task, (query_index, idx, density))
						job_stack.append(job)

					time.sleep(0.05)
					
					# Collect results from the completed jobs
					for job in job_stack:
						try:
							res = job.get()
							if res is not None:
								result_stack.append(res)
						except Exception as e:
							raise AssertionError('Job not done', e)
					# Save results to a temporary file if there are any
					if len(result_stack) > 0:
						tmpfile = os.path.join(tmpdir, f"{itr}.p")
						pd.concat(result_stack).to_pickle(tmpfile)
						tmpresults.append(tmpfile)

				# Collect garbage to free memory
				gc.collect()

			else:
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
							raise AssertionError('job not done', e)	
					if len(result_stack) > 0:
						tmpfile = os.path.join(tmpdir, f"{itr}.p")
						pd.concat(result_stack).to_pickle(tmpfile)
						tmpresults.append(tmpfile)
				gc.collect()
				

		# Combine results from all temporary files if any
		if len(tmpresults) > 0:
			result_df = pd.concat([pd.read_pickle(f) for f in tmpresults])
			print(f'Hit candidates, {result_df.shape[0]}')
		else:
			print(f'No valid alignments for given pLM-BLAST parameters! Consider changing input parameters and validate your inputs')
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

	script_time = datetime.datetime.now() - start_time
	print('total hits found: ', results.shape[0])
	print(f'{cfg.colors["green"]}Done!{cfg.colors["reset"]}\nTime {script_time}')
	# stats
