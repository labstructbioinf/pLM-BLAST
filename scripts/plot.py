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
					
parser.add_argument('-maxseqs', help='the maximal number of sequences to plot (0=plot all)',
					type=int, default=0)
				
parser.add_argument('-mode', help='plot mode',
					type=str, default='qend', choices=('score', 'qstart', 'qend'))

parser.add_argument('-ecod', help='parse ECOD headers', action='store_true')
		
args = parser.parse_args()

assert args.maxseqs >= 0

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
hits_df['len'] = hits_df['qend'] - hits_df['qstart']


if args.maxseqs > 0:
	hits_df = hits_df.head(args.maxseqs)
	print(f'only top {len(hits_df)} hits will be ploted')


if args.ecod:

	tmp_df=pd.DataFrame(hits_df.apply(get_group, axis=1).tolist(), columns=['X', 'H', 'T'])
	hits_df = pd.concat([hits_df, tmp_df], axis=1)

	# group X/T ECOD groups
	print('-'*20)
	cmap = pl.cm.get_cmap('tab10')
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

# PLOT

tick_font_size = 10 # 10
label_font_size = 12 # 9
bar_size = 5 # 5

fig, ax = pl.subplots(1, 1, figsize=(10, len(hits_df)*(bar_size*2)/100), dpi=200)

if args.mode == 'score':
	by = 'score'
	a = False
else:
	by = 'qend'
	a = True

hits_idx_sorted = hits_df.sort_values(by=by, ascending=a).copy()

hits_idx_sorted['score'] = hits_idx_sorted['score'].astype(float)
hits_idx_sorted['done'] = False

assert all(hits_idx_sorted.qstart>=0)

pos=0
lastend=0

for _ in range(len(hits_idx_sorted)):
	
	while True:
		assert not hits_idx_sorted.done.all()
		next_hit = hits_idx_sorted[(hits_idx_sorted.qstart >= lastend) & 
								   (~hits_idx_sorted.done)]
		if len(next_hit) == 0:
			pos+=1
			lastend=0
			continue
		break

	next_hit = next_hit.iloc[0]
	lastend = next_hit.qend
		
	if args.ecod:
		a = {"color":colors[next_hit['T']]}
	else:
		c = (next_hit.score - hits_idx_sorted.score.min()) / (hits_idx_sorted.score.max() - hits_idx_sorted.score.min())
		a = {"color":cmap(c)}
	
	# solid_capstyle='round'
	ax.plot([next_hit.qstart+1, next_hit.qend+1], [pos+1, pos+1], lw=bar_size, **a)
	
	hits_idx_sorted.at[next_hit.name, 'done'] = True
	if hits_idx_sorted.done.all(): break


ax.spines.top.set_visible(False)
ax.spines.left.set_visible(False)
ax.spines.right.set_visible(False)

if args.mode in ['qend', 'qstart', 'score']:	
	ax.invert_yaxis()

#	order = order[::-1]
ax.set_xlim(0, len(query_seq)+1)
ax.tick_params(axis='both', which='major', labelsize=tick_font_size)

if args.ecod:
	h = [mpatches.Patch(color=colors[d], label=d) for d in order]  
	pl.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=label_font_size,
	shadow=False)
	
pl.gca().axes.get_yaxis().set_visible(False)
pl.savefig(args.output, bbox_inches='tight')
