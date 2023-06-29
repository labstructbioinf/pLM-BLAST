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
import torch
from torch.nn.functional import avg_pool1d
from tqdm import tqdm
from Bio.Align import substitution_matrices
blosum62 = substitution_matrices.load("BLOSUM62")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import alntools.density as ds
import alntools as aln

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

def range_limited_float_type(arg, MIN, MAX):
	""" Type function for argparse - a float within some predefined bounds """
	try:
		f = float(arg)
	except ValueError:
		raise argparse.ArgumentTypeError("Must be a floating point number")
	if f <= MIN or f >= MAX :
		raise argparse.ArgumentTypeError("Argument must be <= " + str(MAX) + " and >= " + str(MIN))
	return f


def get_parser():
	parser = argparse.ArgumentParser(description =  
		"""
		Searches a database of embeddings with a query embedding
		""",
		formatter_class=argparse.RawDescriptionHelpFormatter
		)

	range01 = lambda f:range_limited_float_type(f, 0, 1)
	range0100 = lambda f:range_limited_float_type(f, 0, 100)

	# input and output

	parser.add_argument('db', help='database embeddings and index',
						type=str)	

	parser.add_argument('query', help='query embedding and index',
						type=str)	

	parser.add_argument('output', help='output file (csv by default or pickle if --raw option is used)',
						type=str)	

	parser.add_argument('--raw', help='skip postprocessing steps and return pickled pandas dataframe with all alignments', 
			 			action='store_true', default=False)

	# cosine similarity scan

	parser.add_argument('-cosine_percentile_cutoff', help='percentile cutoff for cosine similarity (default: %(default)s). The lower the value, the more sequences will be returned by the pre-screening procedure and aligned with the more accurate but slower pLM-BLAST',
						type=range0100, default=95, dest='COS_PER_CUT')	

	parser.add_argument('-use_chunks', help='use fast chunk cosine similarity screening instead of regular cosine similarity screening. (default: %(default)s)',
			 action='store_true', default=True)

	# plmblast

	parser.add_argument('-alignment_cutoff', help='pLM-BLAST alignment score cut-off (default: %(default)s)',
						type=range01, default=0.3, dest='ALN_CUT')						

	parser.add_argument('-win', help='Window length (default: %(default)s)',
						type=int, default=10, choices=range(50), metavar="[1-50]", dest='WINDOW_SIZE')	

	parser.add_argument('-span', help='Minimal alignment length (default: %(default)s). Must be greater than or equal to the window length',
						type=int, default=25, choices=range(50), metavar="[1-50]", dest='MIN_SPAN_LEN')			

	parser.add_argument('-max_targets', help='Maximum number of targets to include in output (default: %(default)s)',
						type=int, default=1500, dest='MAX_TARGETS')	

	parser.add_argument('--global_aln', help='use global pLM-BLAST alignment. Use only if you expect the query to be a single-domain sequence (default: %(default)s)',
			 			action='store_true', default=False)

	parser.add_argument('-gap_ext', help='Gap extension penalty (default: %(default)s)',
						type=float, default=0, dest='GAP_EXT')

	# misc

	parser.add_argument('--verbose', help='Be verbose (default: %(default)s)', action='store_true', default=True)
	
	parser.add_argument('-workers', help='Number of CPU workers (default: %(default)s)',
						type=int, default=10, dest='MAX_WORKERS')	

	parser.add_argument('-sigma_factor', help='The Sigma factor defines the greediness of the local alignment search procedure (default: %(default)s)',
						type=float, default=2, dest='SIGMA_FACTOR')	

	#parser.add_argument('-bfactor', help='bfactor (default: %(default)s)',
	#					 type=int, default=3, choices=range(1,4), metavar="[1-3]", dest='BF')
	
	#parser.add_argument('-emb_pool', help='embedding type (default: %(default)s) ',
	#					type=int, default=1, dest='EMB_POOL', choices=[1, 2, 4])

	args = parser.parse_args()
	
	# validate provided parameters
	assert args.MAX_TARGETS > 0
	assert args.MAX_WORKERS > 0
	
	assert args.MIN_SPAN_LEN >= args.WINDOW_SIZE, 'The minimum alignment length must be equal to or greater than the window length'
	

	return args

def check_cohesion(frame, filedict, embeddings, truncate=600):
	sequences = frame.sequence.tolist()
	for (idx,file), emb in zip(filedict.items(), embeddings):
		seqlen = len(sequences[idx])
		if seqlen < 600:
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
		for pos, g in enumerate(['GAVLI', 'FYW', 'CM', 'ST', 'KRH', 'DENQ', 'P', '-']):
			g = list(g)
			if aa in g: return pos
		assert False
	res = [aa_to_group(c1)==aa_to_group(c2) for c1, c2 in zip(list(s1), list(s2))]
	return sum(res)/len(res)
	
def calc_ident(s1, s2):
	res = [c1==c2 for c1, c2 in zip(list(s1), list(s2))]
	return sum(res)/len(res)
	

##########################################################################
# 						MAIN											 #
##########################################################################

time_start = datetime.datetime.now()

args = get_parser()
module = aln.base.Extractor()
module.FILTER_RESULTS = True
module.WINDOW_SIZE = args.WINDOW_SIZE
module.GAP_EXT = args.GAP_EXT

EMB_POOL = 1

if args.global_aln:
	module.BFACTOR = 'global'
	if args.verbose:
		print('Global alignment will be used')
else:
	module.BFACTOR = 1
	if args.verbose:
		print('Local alignment will be used')

module.SIGMA_FACTOR = args.SIGMA_FACTOR

# Load database 
db_index = args.db + '.csv'
if args.verbose:
	print(f"Loading database {colors['yellow']}{args.db}{colors['reset']}")
if not os.path.isfile(db_index):
	raise FileNotFoundError(f'Invalid database index file name {db_index}')

db_df = pd.read_csv(db_index)
db_df.set_index(db_df.columns[0], inplace=True)

# Read query 
if args.verbose:
	print(f"Loading query {colors['yellow']}{args.query}{colors['reset']}")
		
query_index = args.query + '.csv'
query_embs = args.query + '.pt_emb.p'
query_df = pd.read_csv(query_index)
query_embs = torch.load(query_embs)

if query_df.shape[0] != len(query_embs):
	raise ValueError(f'The length of the embedding file and the sequence df are different: {query_df.shape[0]} != {len(query_emb)}')

query_seqs = query_df['sequence'].tolist()
query_seqs: List[str]= [str(seq) for seq in query_seqs]

assert len(query_seqs) == 1, "Multi-query input not implemented" 

query_seq = query_seqs[0]

##########################################################################
# 						filtering										 #
##########################################################################

def tensor_transform(x):
	return x.permute(*torch.arange(x.ndim - 1, -1, -1))

query_filedict = dict()
if args.use_chunks:
	
	if args.verbose:
		print('Loading database for chunk cosine similarity screening...')

	dbfile = os.path.join(args.db, 'emb.64')
	embedding_list = torch.load(dbfile)
	
	filelist = [os.path.join(args.db, f'{f}.emb') for f in range(0, db_df.shape[0])] # db_df is database index
	
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
											dbpath = args.db,
											quantile = args.COS_PER_CUT/100,
											num_workers = args.MAX_WORKERS)
		filedict = { k : v.replace('.emb.sum', f'.emb{emb_type}') for k, v in filedict.items()}
		query_filedict[i] = filedict


filelist = list(filedict.values())

# check_cohesion(db_df, filedict, embedding_list)

if len(filedict) == 0:
	print(f'{colors["red"]}No matches after pre-filtering. Consider lowering the -cosine_percentile_cutoff{colors["reset"]}')
	sys.exit(0)
	
##########################################################################
# 						plm-blast										 #
##########################################################################
	
# Multi-CPU search
if args.verbose:
	print(f'Running pLM-BLAST for {colors["yellow"]}{len(filedict)}{colors["reset"]} hits...')
	
query_embs_pool = [
	tensor_transform(
		avg_pool1d(tensor_transform(emb).unsqueeze(0), EMB_POOL)
		).squeeze() for emb in query_embs
	]
query_embs_pool = [emb.numpy() for emb in query_embs_pool]

iter_id = 0
records_stack = []
num_indices_per_query = [len(vals) for vals in query_filedict.values()]
batch_size = 20*args.MAX_WORKERS
batch_size = min(300, batch_size)
num_batches_per_query = [max(math.floor(nind/batch_size), 1) for nind in num_indices_per_query]
num_batches = sum(num_batches_per_query)
	
#with tqdm(total=num_batches, desc='Comparison of embeddings') as progress_bar:
	#for filedict, query_emb, batches in zip(query_filedict.values(), query_embs_pool, num_batches_per_query):
		
filedict = list(query_filedict.values())[0]
query_emb = query_embs_pool[0]
batches = num_batches_per_query[0]

embedding_list = embedding_list = ds.load_full_embeddings(filelist=filelist)
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

resdf = pd.concat(records_stack)

if resdf.score.max() > 1:
	print(records_stack[0].score.max())
	print(f'{colors["red"]}Error: score is greater then one{colors["reset"]}', resdf.score.min(), resdf.score.max())
	sys.exit(0)

time_end = datetime.datetime.now()

print(f'{colors["green"]}Done!{colors["reset"]} Time {time_end-time_start}')

# Prepare output
if args.raw:
	resdf.to_pickle(args.output)
elif len(resdf) == 0:
	print('No hits found!')
else:
	resdf = resdf[resdf.score>=args.ALN_CUT]
	if len(resdf) == 0:
		print(f'No matches found! Try reducing the alignment_cutoff parameter. The current cutoff is {args.ALN_CUT}')
	else:
		print('Preparing output...')
		resdf.drop(columns=['span_start', 'span_end', 'pathid', 'spanid', 'len'], inplace=True)
		resdf['sid'] = resdf['i'].apply(lambda i:db_df.iloc[i]['id'])
		resdf['sdesc'] = resdf['i'].apply(lambda i:db_df.iloc[i]['description'])
		resdf['tlen'] = resdf['i'].apply(lambda i:len(db_df.iloc[i]['sequence']))
		resdf['qlen'] = len(query_seq)
		resdf['qstart'] = resdf['indices'].apply(lambda i:i[0][1])
		resdf['qend'] = resdf['indices'].apply(lambda i:i[-1][1])
		resdf['tstart'] = resdf['indices'].apply(lambda i:i[0][0])
		resdf['tend'] = resdf['indices'].apply(lambda i:i[-1][0])

		assert all(resdf['qstart'].apply(lambda i: i <= len(query_seq)-1))
		assert all(resdf['qend'].apply(lambda i: i <= len(query_seq)-1))

		resdf.sort_values(by='score', ascending=False, inplace=True)
		resdf.reset_index(inplace=True)

		# alignment, conservation, etc.
		for idx, row in resdf.iterrows():
			tmp_aln = aln.alignment.draw_alignment(row.indices, 
										   db_df.iloc[row.i].sequence,
										   query_seq,
										   output='str')
			tmp_aln=tmp_aln.split('\n')
			resdf.at[idx, 'qseq'] = tmp_aln[2]
			resdf.at[idx, 'tseq'] = tmp_aln[0]
			resdf.at[idx, 'con'] = calc_con(tmp_aln[2], tmp_aln[0])
			resdf.at[idx, 'ident'] = calc_ident(tmp_aln[2], tmp_aln[0])
			resdf.at[idx, 'similarity'] = calc_similarity(tmp_aln[2], tmp_aln[0])
		# reset index
		resdf.drop(columns=['index', 'indices', 'i'], inplace=True)
		resdf.index.name = 'index'
		# order columns
		resdf = resdf[['score','ident','similarity','sid', 'sdesc','qstart','qend','qseq','con','tseq', 'tstart', 'tend', 'tlen', 'qlen']]
		# clip df
		resdf = resdf.head(args.MAX_TARGETS)
		# save
		resdf.to_csv(args.output)
