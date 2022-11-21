
import sys, os
import pandas as pd

assert len(sys.argv)>1, 'usage: csv2nice.py output_csv_file'
infile = sys.argv[1]
assert os.path.isfile(infile), f'file {infile} not found'

df = pd.read_csv(infile, sep=';')
df.set_index('index', inplace=True)


for pos, (idx, row) in enumerate(df.iterrows()):
    print(f'\n\nNo {pos+1}')
    print(f'>{row.sdesc}')
    print(f'Score={row.score} Identities={round(row.ident*100)}% Similarity={round(row.similarity, 2)}\n')
    print(f'Q {row.qstart+1:>6} {row.qseq} {row.qend:<6}')
    print(f'         {row.con}')
    print(f'T {row.tstart+1:>6} {row.tseq} {row.tend:<6}')
