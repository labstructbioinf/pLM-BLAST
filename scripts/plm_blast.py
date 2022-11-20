import sys, os, gc, math
import argparse
import concurrent

import pandas as pd
import numpy as np

import torch
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

from Bio.Align import substitution_matrices
blosum62 = substitution_matrices.load("BLOSUM62")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import alntools.density as ds
import alntools as aln

import datetime

parser = argparse.ArgumentParser(description =  
	"""
	Searches a database of embeddings with a query embedding
	""",
	formatter_class=argparse.RawDescriptionHelpFormatter
	)


def range_limited_float_type(arg, MIN, MAX):
    """ Type function for argparse - a float within some predefined bounds """
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f <= MIN or f >= MAX :
        raise argparse.ArgumentTypeError("Argument must be <= " + str(MAX) + " and >= " + str(MIN))
    return f

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
					 type=int, default=1, choices=range(26), metavar="[1-25]", dest='WIN')				    

parser.add_argument('-span', help='Minimal alignment length (default: %(default)s)',
					 type=int, default=15, dest='SPAN')
					 
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

args = parser.parse_args()

assert args.SPAN >= args.WIN, 'Span has to be >= window!'
assert args.MAX_TARGETS > 0
assert args.MAX_WORKERS > 0, 'At least one CPU core is needed!'

assert args.COS_SIM_CUT!=None or args.COS_PER_CUT!=None, 'Please define COS_PER_CUT _or_ COS_SIM_CUT!'

### FUNCTIONS

def check(df, embs):
    for seq, emb in zip(df.sequence.tolist(), embs):
        assert len(seq) == len(emb), 'index and embeddings files differ'

def compare(emb1, emb2, window=1, min_span=15, bfactor=3, gap_opening=0, gap_extension=0, sigma_factor=1):
    
    densitymap = ds.embedding_similarity(emb1, emb2)
    arr = densitymap.cpu().numpy()
    
    paths = aln.alignment.gather_all_paths(densitymap, bfactor=bfactor, 
                                           gap_opening=gap_opening, 
                                           gap_extension=gap_extension)
    
    spans_locations = aln.prepare.search_paths(arr,
                                                paths=paths,
                                                window=window,
                                                sigma_factor=sigma_factor,
                                                min_span=min_span)
    results = pd.DataFrame(spans_locations.values())
    
    return results
     
def full_compare(emb1, emb2, i):   
	res = compare(emb1, emb2, window=args.WIN, min_span=args.SPAN, 
					   bfactor=3, 
					   gap_opening=args.GAP_OPEN,
					   gap_extension=args.GAP_EXT,
					   sigma_factor=args.SIGMA_FACTOR)

	if len(res)>0:
		if res.score.max() >= args.ALN_CUT:
			# add referece index to each hit
			res['i'] = i
		
			# filter out redundant hits
			res = aln.postprocess.filter_result_dataframe(res)
		
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
    return np.mean(res)         
    
def calc_ident(s1, s2):
    res = [c1==c2 for c1, c2 in zip(list(s1), list(s2))]
    return np.mean(res)     
    
### MAIN

# read database 
db_index = args.db+'.csv'
db_emb = args.db+'.pt_emb.p'
print('Loading database...')
db_df = pd.read_csv(db_index)
db_df.set_index(db_df.columns[0], inplace=True)
db_embs = torch.load(db_emb)
check(db_df, db_embs)
print(f'{len(db_embs)} sequences in the database')


# read query 
query_index = args.query+'.csv'
query_emb = args.query+'.pt_emb.p'
print('Loading query...')
query_df = pd.read_csv(query_index)
query_embs = torch.load(query_emb)
assert len(query_embs)==1
query_emb = query_embs[0]
query_seq = query_df.iloc[0].sequence
print(f'query sequence length is {len(query_seq)}')
check(query_df, query_embs)

# Cosine similarity pre-screening
print('Calculating cosine similarity...')
all_embs = np.array([i.numpy().mean(0) for i in db_embs])
cos_sim = cosine_similarity(query_emb.numpy().mean(0).reshape(1, -1), all_embs)[0]

if args.COS_PER_CUT:
	defined_COS_CUT = np.percentile(cos_sim, float(args.COS_PER_CUT))
else:
	assert args.COS_SIM_CUT
	defined_COS_CUT = args.COS_SIM_CUT
print(f'using {np.round(defined_COS_CUT, 2)} cosine similarity cut-off')

cos_count = np.sum(cos_sim >= defined_COS_CUT)
print(f'{cos_count} hits after pre-filtering')

if cos_count==0:
	print('No hits after pre-filtering. Consider lowering `cosine_cutoff`')
	sys.exit(0)

            
# Multi-CPU search
time_start = datetime.datetime.now()

iter_id = 0
records_stack = []
indices = np.where(cos_sim >= defined_COS_CUT)[0]

# batching 
#num_indices = indices.size
#batch_size = 200 # 10*args.MAX_WORKERS
#num_batch = max(math.floor(num_indices/batch_size), 1)
#print(f'{num_batch} chunks:')

#for chunk in np.array_split(indices, num_batch):

with concurrent.futures.ThreadPoolExecutor(max_workers=args.MAX_WORKERS) as executor:

	job_stack = {}
					
	for i in indices: # use chunk for batching
		job = executor.submit(full_compare, query_emb, db_embs[i], i)
		job_stack[job] = iter_id

	with tqdm(total=len(indices)) as progress_bar:
		for job in concurrent.futures.as_completed(job_stack):
			res = job.result()
			if len(res) > 0:
				records_stack.append(res)
			progress_bar.update(1)

	gc.collect()
		
time_end = datetime.datetime.now()

print(f'Time {time_end-time_start}')

# Prepare output
if len(records_stack)==0:
	print('No hits found!')
else:
	res_df = pd.concat(records_stack)
	res_df = res_df[res_df.score>=args.ALN_CUT]

	if len(res_df)==0:
		print(f'No hits found! Try decreasing the alignment_cutoff parameter. Current cut-off is {args.ALN_CUT}')	
	else:

		print('Preparing output...')
	
		res_df.drop(columns=['span_start', 'span_end', 'pathid', 'spanid', 'len', 'y1', 'x1'], inplace=True)

		res_df['sid'] = res_df['i'].apply(lambda i:db_df.iloc[i]['id'])
		res_df['sdesc'] = res_df['i'].apply(lambda i:db_df.iloc[i]['description'])
		res_df['tlen'] = res_df['i'].apply(lambda i:len(db_df.iloc[i]['sequence']))
		res_df['qlen'] = len(query_seq)
		
		res_df['qstart'] = res_df['indices'].apply(lambda i:i[0][1])
		res_df['qend'] = res_df['indices'].apply(lambda i:i[-1][1])
		res_df['tstart'] = res_df['indices'].apply(lambda i:i[0][0])
		res_df['tend'] = res_df['indices'].apply(lambda i:i[-1][0])

		assert all(res_df['qstart'].apply(lambda i: i <= len(query_seq)-1))
		assert all(res_df['qend'].apply(lambda i: i <= len(query_seq)-1))

		res_df.sort_values(by='score', ascending=False, inplace=True)
		res_df.reset_index(inplace=True)

		# alignment, conservation, etc.
		for idx, row in res_df.iterrows():
			tmp_aln=aln.alignment.draw_alignment(row.indices, 
										   db_df.iloc[row.i].sequence,
										   query_seq,
										   output='str')
	
			tmp_aln=tmp_aln.split('\n')
			res_df.at[idx, 'qseq'] = tmp_aln[2]
			res_df.at[idx, 'tseq'] = tmp_aln[0]
			res_df.at[idx, 'con'] = calc_con(tmp_aln[2], tmp_aln[0])
			res_df.at[idx, 'ident'] = calc_ident(tmp_aln[2], tmp_aln[0])
			res_df.at[idx, 'similarity'] = calc_similarity(tmp_aln[2], tmp_aln[0])
	
		# reset index
		res_df.drop(columns=['index', 'indices', 'i'], inplace=True)
		res_df.index.name = 'index'
 
		# order columns
		res_df = res_df[['score','ident','similarity','sid', 'sdesc','qstart','qend','qseq','con','tseq', 'tstart', 'tend', 'tlen', 'qlen']]
	
		# clip df
		res_df = res_df.head(args.MAX_TARGETS)
	
		# save
		res_df.to_csv(args.output, sep=';')
