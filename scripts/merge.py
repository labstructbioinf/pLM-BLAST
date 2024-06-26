import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description =  
	"""
	Merges hits in output CSV from `plm_blast.py`
	""",
	formatter_class=argparse.RawDescriptionHelpFormatter
	)

parser.add_argument('csv', help='CSV file with hits',
					type=str)
								
parser.add_argument('new_csv', help='CSV file merged hits',
					type=str)

parser.add_argument('-score', help='pLM-BLAST score cut-off (use 0 for no cut-off). Used before merging.',
					type=float, default=0)
					
parser.add_argument('-max_hits', help='The number of best matches to return (use 0 for no limit). Used after merging.',
					type=int, default=0)
					
parser.add_argument('--skip_merge',
                    help='Skips merge but applies score and max hits cutoffs.',
                    action='store_true')
		
args = parser.parse_args()

# get data
hits_df = pd.read_csv(args.csv, sep=';')
if 'score' not in hits_df.columns:
	raise KeyError('missing score column: ', hits_df.columns)
hits_df = hits_df[hits_df.score>=args.score]
print(f'{len(hits_df)} hits after applying {args.score} score cut-off')

if not args.skip_merge:

	# merge
	def merge(current_subg):

		seq_spacer = '~~~~~'

		first_subg = current_subg[0]
		last_subg = current_subg[-1]

		print(f'merging {len(current_subg)} hits to {first_subg.sid}')

		# qid;score;ident;similarity;sid;qstart;qend;qseq;con;tseq;tstart;tend;tlen;qlen;match_len;sdesc
	
		new_subg = np.array([
					first_subg.qid, # qid
					np.round(np.mean([i.score for i in current_subg]), 2), # mean score
					np.round(np.mean([i.ident for i in current_subg]), 2), # mean ident
					np.round(np.mean([i.similarity for i in current_subg]), 2), # mean similarity
					first_subg.sid, # sid
					first_subg.qstart, # query start
					last_subg.qend, # query end
					seq_spacer.join([i.qseq for i in current_subg]), # merged query sequences
					(" "*len(seq_spacer)).join([i.con for i in current_subg]), # merged consensus
					seq_spacer.join([i.tseq for i in current_subg]), # merged subject sequences
					first_subg.tstart, # subjet start
					last_subg.tend, # subject end
					first_subg.tlen, # subject length
					first_subg.qlen, # query length
					last_subg.qend - first_subg.qstart + 1, # match length relative to the query
					first_subg.sdesc, # subject description
					], dtype=object)
						
		return new_subg

	res=[]
	for gidx, g in hits_df.groupby('sid'):

		# more than one hit to a single target 
		if len(g)>1:
			
			# sort hits by starting position in the query	
			g_sorted = g.sort_values(by='qstart', ascending=False).copy()
			assert g_sorted.index.is_unique
			g_sorted_index = g_sorted.index.to_list()
			#print()
			#print(g_sorted.qstart.tolist())
			#print(g_sorted.tstart.tolist())
			# list of hits for potential merging
			current_subg = []
			while g_sorted_index:
	
				hit = g_sorted.loc[g_sorted_index.pop()]

				if len(current_subg) == 0:
					current_subg.append(hit)
					continue
				#print('\t', [i.qstart for i in current_subg], hit.qstart)
				first_subg = current_subg[0]			
				qlen = hit.qstart - first_subg.qstart
				tlen = hit.tstart - first_subg.tstart 
				#print('\t', qlen, tlen, (min(qlen, tlen) / max(qlen, tlen)))
				if (qlen<0 or tlen<0) or (min(qlen, tlen) / max(qlen, tlen) < 0.7):
					if len(current_subg) > 1:
						res.append(merge(current_subg))
					else:
						res.append(current_subg[0].values)					
					current_subg = []
				current_subg.append(hit)
			if len(current_subg) > 1:
				res.append(merge(current_subg))
			else:
				res.append(current_subg[0])	
		else:
			for subg in g.values:
				res.append(subg)

				
	mhits_df = pd.DataFrame(res, columns=hits_df.columns)
	
else:
	print('Merge skipped...')
	mhits_df = hits_df
	
	
print('Preparing output...')	
mhits_df = mhits_df.sort_values(by='score', ascending=False)
#mhits_df.drop(columns=['index'], inplace=True)
mhits_df.index = np.arange(1,len(mhits_df)+1)
mhits_df.index.name = 'index'

if args.max_hits > 0:
	mhits_df = mhits_df.head(args.max_hits)

mhits_df.to_csv(args.new_csv, sep=';')

