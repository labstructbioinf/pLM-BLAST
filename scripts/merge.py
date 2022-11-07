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

parser.add_argument('-score', help='score cut-off',
					type=float, default=0)

				
args = parser.parse_args()



# get data
hits_df = pd.read_csv(args.csv)
hits_df = hits_df[hits_df.score>=args.score]
print(f'{len(hits_df)} hits after applying {args.score} score cut-off')


# merge
def merge(current_subg):

	first_subg = current_subg[0]
	last_subg = current_subg[-1]

	print(f'merging {len(current_subg)} hits to {first_subg.sid}')

	new_subg = np.array([0,
				np.mean([i.score for i in current_subg]),
				0,
				0,
				first_subg.sid,
				first_subg.sdesc,
				first_subg.qstart,
				last_subg.qend,
				'',
				'',
				'',
				first_subg.tstart,
				last_subg.tend
						], dtype=object)
						
	return new_subg

res=[]
for gidx, g in hits_df.groupby('sid'):

	# more than one hit to a target 
	if len(g)>1:
				
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
			
			if len(current_subg)==0:
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
					#print('\twritting one')
					res.append(current_subg[0].values)					
				current_subg = []

			current_subg.append(hit)
		
		if len(current_subg) > 1:
			#print('merge post')
			res.append(merge(current_subg))
		else:
			#print('\twritting one post')
			res.append(current_subg[0])	
				
	else:
		for subg in g.values:
			res.append(subg)

print('Preparing output...')			
mhits_df = pd.DataFrame(res, columns=hits_df.columns)
mhits_df = mhits_df.sort_values(by='score', ascending=False)
mhits_df.drop(columns=['index'], inplace=True)
mhits_df.index = np.arange(1,len(mhits_df)+1)
mhits_df.index.name = 'index'
mhits_df.to_csv(args.new_csv)
