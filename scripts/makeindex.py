import argparse
from Bio import SeqIO
import pandas as pd
import warnings

parser = argparse.ArgumentParser(description =  
	"""
	Calculates sorted index for a given FASTA file
	""",
	formatter_class=argparse.RawDescriptionHelpFormatter
	)

parser.add_argument('fasta_file', help='input sequences in FASTA format',
					type=str)
								
parser.add_argument('index_file', help='output index CSV',
					type=str)
			
parser.add_argument('--uniprot', '--up', help='Use Uniprot headers parser',
						dest='uniprot', default=False, action='store_true')

parser.add_argument('-max_seq_len', help='longer sequences will be split (default: %(default)s)',
					 type=int, default=1000, dest='MAX_SEQ_LEN')	
				 
parser.add_argument('-min_seq_len', help='min allowed seq len (default: %(default)s)',
					 type=int, default=15, dest='MIN_SEQ_LEN')	

args = parser.parse_args()


### Warnings class

class ToShortSeq(UserWarning):
    pass


class UnexpectedCharSeq(UserWarning):
    pass


### MAIN

gres = []

for seq in SeqIO.parse(args.fasta_file, 'fasta'):

        if len(seq.seq)<args.MIN_SEQ_LEN:
            warnings.warn(f"{seq.id} is to short. The Sequence has not been added", ToShortSeq)
            continue
        if set(seq.seq) - set('QWERTYIPASDFGHKLCVNM') != set():
            warnings.warn(f"{seq.id} has characters that do not encode amino acids. The Sequence has not been added", UnexpectedCharSeq)
            continue
	if args.uniprot:
		new_id = ' '.join(seq.id.split('|')[1:])
		new_desc = ' '.join(seq.description.split('|')[1:])
	else:
		new_id = seq.id
		new_desc = seq.description

	if len(seq.seq)>args.MAX_SEQ_LEN:

		for pos, subseq in enumerate(
			[seq.seq[i:i+args.MAX_SEQ_LEN] for i in range(0, len(seq.seq), args.MAX_SEQ_LEN)]
		):
			if len(subseq)>=args.MIN_SEQ_LEN:
				gres.append((f'{new_id}_{pos+1} {pos*args.MAX_SEQ_LEN+1}-{pos*args.MAX_SEQ_LEN+len(subseq)}', new_desc, subseq))
			
	else:
		gres.append(((new_id, new_desc, str(seq.seq))))

gres_df = pd.DataFrame(gres, columns=['id', 'description', 'sequence'])
gres_df = gres_df.sort_values(by='sequence', key=lambda x:x.str.len())
gres_df.reset_index(inplace=True, drop=True)
gres_df.to_csv(args.index_file)

print(f'{args.index_file} has been written. This file can be used to calculate embeddings with `embeddings.py`')
