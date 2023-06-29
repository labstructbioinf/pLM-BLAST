import sys, os
import pandas as pd

assert len(sys.argv)>1, 'usage: csv2nice.py output_csv_file'
infile = sys.argv[1]
assert os.path.isfile(infile), f'file {infile} not found'

df = pd.read_csv(infile, sep=';')
df.reset_index(inplace=True)
# Reset the index starting from 1
df.index = range(1, len(df) + 1)

print(' No Hit						 Score   Query	   Template')
for pos, (idx, row) in enumerate(df.iterrows()):
	q_str = f'{row.qstart+1}-{row.qend}'
	t_str = f'{row.tstart+1}-{row.tend}'
	print(f'{pos+1:>3} {row.sid:<27} {row.score:<5} {q_str:^10} {t_str:^8} ({row.tlen})')

for pos, (idx, row) in enumerate(df.iterrows()):
	print(f'\n\nNo {pos+1}')
	print(f'>{row.sdesc}')
	print(f'Score={row.score} Identities={round(row.ident*100)}% Similarity={round(row.similarity, 2)}\n')
	
	if row.qseq.find('~~~~~')>-1:
		print('Warning: this is a merged match. The linking points are indicated with "~~~~~"\n')
	
	print(f'Q {row.qstart+1:>6} {row.qseq} {row.qend:<6}')
	print(f'         {row.con}')
	print(f'T {row.tstart+1:>6} {row.tseq} {row.tend:<6}')
