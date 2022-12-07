import sys
import os
import gc
import math
import time
import argparse
import concurrent
import itertools

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from Bio.Align import substitution_matrices
blosum62 = substitution_matrices.load("BLOSUM62")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import alntools.density as ds
import alntools as aln

import datetime

### FUNCTIONS



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

	parser.add_argument('db', help='Database embeddings and index (`csv` and `pt_emb.p` extensions will be added automatically)',
						type=str)
											
	parser.add_argument('query', help='Query embedding and index (`csv` and `pt_emb.p` extensions will be added automatically)',
						type=str)
											
	parser.add_argument('output', help='Output csv file',
						type=str)
										
	group = parser.add_mutually_exclusive_group()
						
	group.add_argument('-cosine_cutoff', help='Cosine similarity cut-off (0..1)',
						type=range01, default=None, dest='COS_SIM_CUT')	
						
	group.add_argument('-cosine_percentile_cutoff', help='Cosine similarity percentile cut-off (0-100)',
						type=range0100, default=None, dest='COS_PER_CUT')						 
								
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
						
	#parser.add_argument('-bfactor', help='bfactor (default: %(default)s)',
	#					 type=int, default=3, choices=range(1,4), metavar="[1-3]", dest='BF')		
						
	parser.add_argument('-workers', help='Number of CPU workers (default: %(default)s)',
						type=int, default=1, dest='MAX_WORKERS')			    
							
	parser.add_argument('-gap_open', help='Gap opening penalty (default: %(default)s)',
						type=float, default=0, dest='GAP_OPEN')				    
						
	parser.add_argument('-gap_ext', help='Gap extension penalty (default: %(default)s)',
						type=float, default=0, dest='GAP_EXT')

	parser.add_argument('-emb_pool', help='embedding type (default: %(default)s) ',
						type=int, default=1, dest='EMB_POOL', choices=[1, 2, 4])			    

	args = parser.parse_args()
	# validate provided parameters
	assert args.MIN_SPAN_LEN >= args.WINDOW_SIZE, 'Span has to be >= window!'
	assert args.MAX_TARGETS > 0
	assert args.MAX_WORKERS > 0, 'At least one CPU core is needed!'
	assert args.COS_SIM_CUT != None or args.COS_PER_CUT != None, 'Please define COS_PER_CUT _or_ COS_SIM_CUT!'

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

def full_compare(emb1, emb2, idx, file):
	global module 
	res = module.embedding_to_span(emb1, emb2)
	if len(res) > 0:
		if res.score.max() >= args.ALN_CUT:
			# add referece index to each hit
			res['i'] = idx
			res['dbfile']  = file
			# filter out redundant hits
			res = aln.postprocess.filter_result_dataframe(res)
			if res.score.max() > 1:
				raise KeyError('score err', res.score.max())
			return res
	return []
    
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
    
time_start = datetime.datetime.now()

args = get_parser()
module = aln.base.Extractor()
module.SIGMA_FACTOR = args.SIGMA_FACTOR
module.WINDOW_SIZE = args.WINDOW_SIZE
module.GAP_OPEN = args.GAP_OPEN
module.GAP_EXT = args.GAP_EXT
module.BFACTOR = 1

truncate = 600
### MAIN
# read database 
db_index = args.db + '.csv'
print(f'Using database: {args.db}')
print('Loading database...')
if not os.path.isfile(db_index):
	raise FileNotFoundError(f'invalid database frame file, {db_index}')
db_df = pd.read_csv(db_index)
db_df.set_index(db_df.columns[0], inplace=True)

# read query 
query_index = args.query + '.csv'
query_emb = args.query + '.pt_emb.p'

query_df = pd.read_csv(query_index)
query_emb = torch.load(query_emb)[0]
# select embeddings pool factor
if args.EMB_POOL == 1:
	emb_type = ''
elif args.EMB_POOL == 2:
	emb_type =  '.512'
elif args.EMB_POOL == 4:
	emb_type = '.256'
else:
	raise ValueError(f'invalid EMB_POOL value: {args.EMB_POOL}')
query_emb_pool = torch.nn.functional.avg_pool1d(query_emb.T, args.EMB_POOL).T
query_seq = str(query_df.iloc[0].sequence)
print('cosine similarity screening ...')
filedict = ds.load_and_score_database(query_emb,
									dbpath = args.db,
									quantile = args.COS_PER_CUT/100,
									num_workers = args.MAX_WORKERS)
print(f'{len(filedict)} hits after pre-filtering')
filedict = { k : v.replace('.emb.sum', f'.emb{emb_type}') for k, v in filedict.items()}
filelist = list(filedict.values())


print(f'loading per residue embeddings with pool: {emb_type}')
embedding_list = ds.load_full_embeddings(filelist=filelist, num_workers=1)
check_cohesion(db_df, filedict, embedding_list)


if len(filedict) == 0:
	print('No hits after pre-filtering. Consider lowering `cosine_cutoff`')
	sys.exit(0)
         
iter_id = 0
records_stack = []
num_indices = len(filedict)
batch_size = 20*args.MAX_WORKERS
batch_size = min(300, batch_size)
num_batch = max(math.floor(num_indices/batch_size), 1)
# Multi-CPU search
print('running plm blast')
with tqdm(total=num_batch) as progress_bar:
	for batch_start in range(0, num_batch):
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
				job = executor.submit(full_compare, query_emb_pool, emb, idx, file)
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
			progress_bar.update(1)
		gc.collect()

resdf = pd.concat(records_stack)
if resdf.score.max() > 1:
	print(records_stack[0].score.max())
	print('score is greater then one', resdf.score.min(), resdf.score.max())
	sys.exit(0)

time_end = datetime.datetime.now()

print(f'Time {time_end-time_start}')
# Prepare output
if len(records_stack) == 0:
	print('No hits found!')
else:
	resdf = pd.concat(records_stack)
	resdf.to_csv('tmpres.csv')
	resdf = resdf[resdf.score>=args.ALN_CUT]
	if len(resdf) == 0:
		print(f'No hits found! Try decreasing the alignment_cutoff parameter. Current cut-off is {args.ALN_CUT}')	
	else:
		print('Preparing output...')
		resdf.drop(columns=['span_start', 'span_end', 'pathid', 'spanid', 'len', 'y1', 'x1'], inplace=True)
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
