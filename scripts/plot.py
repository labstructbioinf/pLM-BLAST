import argparse
import pandas as pd
import matplotlib.pylab as pl
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser(description =  
	"""
	Plots merged or unmerged results of ECOD database search
	""",
	formatter_class=argparse.RawDescriptionHelpFormatter
	)

parser.add_argument('csv', help='CSV file with hits',
					type=str)
			
parser.add_argument('query', help='query CSV file',
					type=str)
			
parser.add_argument('output', help='output PNG plot',
					type=str)
			
parser.add_argument('-score', help='score cut-off',
					type=float, default=0)
				
parser.add_argument('-mode', help='plot mode',
					type=str, default='qend', choices=('score', 'qstart', 'qend'))

parser.add_argument('-ecod', help='parse ECOD headers', action='store_true')

				
			
args = parser.parse_args()

### FUNCTIONS

def get_group(row):
	s = row.sdesc
	x = s[s.find(', X: '):s.find(', H:')][2:]
	h = s[s.find(', H: '):s.find(', T:')][2:]
	t = s[s.find(', T: '):s.find(', F:')][2:]
	assert x!="" and h!="" and t!=""
	return x, h, t

### GET DATA    

# read query 
query_index = args.query
query_df = pd.read_csv(query_index)
assert len(query_df)==1
query_seq = query_df.iloc[0].sequence
print(f'query sequence length is {len(query_seq)}')

# read results
hits_df = pd.read_csv(args.csv)
hits_df = hits_df[hits_df.score >= args.score]

if args.ecod:

	tmp_df=pd.DataFrame(hits_df.apply(get_group, axis=1).tolist(), columns=['X', 'H', 'T'])
	hits_df = pd.concat([hits_df, tmp_df], axis=1)

	# group X/T ECOD groups
	print('-'*20)
	cmap = pl.cm.get_cmap('tab20')
	colors={}
	cidx=0
	order = []
	for idx, g in hits_df.groupby('X'):
		print(idx)
		if idx=='X:NO_X_NAME': 
			c = 'grey'
		else:
			c = cmap(cidx)
			cidx+=1

		for t in g['T'].unique().tolist():
			print(f'\t{t}')
			assert not t in colors, t
			colors[t] = c
			order.append(t)
	print('-'*20)

	if len(colors)>20:
		print(f'warning: palette has 20 colors but {len(colors)} X groups were identified')       
else:
	cmap = pl.cm.get_cmap('coolwarm')

# plot

fig, ax = pl.subplots(1, 1, figsize=(10, len(hits_df)*7/100), dpi=100)

if args.mode == 'score':
	a = False
else:
	a = True

hits_idx_sorted = hits_df.sort_values(by=args.mode, ascending=a).copy()

hits_idx_sorted['score'] = hits_idx_sorted['score'].astype(float)
hits_idx_sorted['done'] = False
pos=0
lastend=0

for _ in range(len(hits_idx_sorted)):
	while True:
		next_hit = hits_idx_sorted[(hits_idx_sorted.qstart >= lastend) & 
								   (~hits_idx_sorted.done)]
		if len(next_hit) == 0:
			pos+=1
			lastend=0
			continue
		break

	next_hit = next_hit.iloc[0]
	lastend = next_hit.qend

	hits_idx_sorted.at[next_hit.name, 'done'] = True
	if hits_idx_sorted.done.all(): break
	
	if args.ecod:
		a = {"color":colors[next_hit['T']]}
	else:
		c = (next_hit.score - hits_idx_sorted.score.min()) / (hits_idx_sorted.score.max() - hits_idx_sorted.score.min())
		a = {"color":cmap(c)}
		
	ax.plot([next_hit.qstart+1, next_hit.qend+1], [pos+1, pos+1], lw=5, **a)

if args.mode in ['qend', 'qstart', 'score']:	
	ax.invert_yaxis()

#	order = order[::-1]
ax.set_xlim(1, len(query_seq))

if args.ecod:
	h = [mpatches.Patch(color=colors[d], label=d) for d in order]  
	pl.legend(handles=h, bbox_to_anchor=(1, -0.1))
	
pl.gca().axes.get_yaxis().set_visible(False)
pl.savefig(args.output, bbox_inches='tight')

