import argparse, os
from Bio import SeqIO
import pandas as pd

### CONFIGURATION ###
emb_scr = '../embeddings.py'
#os.environ['TRANSFORMERS_CACHE'] = '/ebio/abt1_share/toolkit_support1/code/bioprogs/tools/pLM-BLAST/cache'
#os.environ['TRANSFORMERS_OFFLINE']= "1"

parser = argparse.ArgumentParser(description =  
	"""
	Calculates embedding and index for a single query sequence in FASTA format
	""",
	formatter_class=argparse.RawDescriptionHelpFormatter
	)
	
parser.add_argument('query_file', help='input sequence file in FASTA format',
				    type=str)
				    
parser.add_argument('query_emb_file', help='output embedding',
				    type=str)
				    
parser.add_argument('query_index_file', help='output index CSV',
				    type=str)
				    
args = parser.parse_args()

seqs = list(SeqIO.parse(args.query_file, 'fasta'))
assert len(seqs) == 1, 'please provide a FASTA file with 1 sequence'
seq = seqs[0]

df = pd.DataFrame([[seq.description, seq.seq]], columns=['desc', 'sequence'])
df.set_index('desc', inplace=True)
df.to_csv(args.query_index_file)

os.system(f'python {emb_scr} {args.query_index_file} {args.query_emb_file} -cname sequence')

