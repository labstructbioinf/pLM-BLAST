from Bio import SeqIO
import pandas as pd
import warnings

### Warnings class
class ToShortSeq(UserWarning):
	pass


class UnexpectedCharSeq(UserWarning):
	pass


### MAIN function
def make_index(fasta_file, output_file=None, uniprot=False, max_seq_len=1000, min_seq_len=15):
	
	gres = []

	for seq in SeqIO.parse(fasta_file, 'fasta'):

		if len(seq.seq)<min_seq_len:
			warnings.warn(f"{seq.id} is to short. The Sequence has not been added", ToShortSeq)
			continue
		if set(seq.seq) - set('QWERTYIPASDFGHKLCVNMX') != set():
			warnings.warn(f"{seq.id} has characters that do not encode amino acids. The Sequence has not been added", UnexpectedCharSeq)
			continue
		if uniprot:
			new_id = ' '.join(seq.id.split('|')[1:])
			new_desc = ' '.join(seq.description.split('|')[1:])
		else:
			new_id = seq.id
			new_desc = seq.description

		if len(seq.seq)>max_seq_len:
			for pos, subseq in enumerate(
				[seq.seq[i:i+max_seq_len] for i in range(0, len(seq.seq), max_seq_len)]
			):
				if len(subseq)>=min_seq_len:
					gres.append((f'{new_id}_{pos+1} {pos*max_seq_len+1}-{pos*max_seq_len+len(subseq)}', new_desc, subseq))
				
		else:
			gres.append(((new_id, new_desc, str(seq.seq))))

	gres_df = pd.DataFrame(gres, columns=['id', 'description', 'sequence'])
	gres_df = gres_df.sort_values(by='sequence', key=lambda x: x.str.len())
	gres_df.reset_index(inplace=True, drop=True)

	if output_file:
		gres_df.to_csv(output_file)
	else:
		return gres_df


if __name__ == "__main__":
	import argparse

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
	
	gres_df = make_index(args.fasta_file, args.uniprot, max_seq_len=args.MAX_SEQ_LEN, min_seq_len=args.MIN_SEQ_LEN)
	gres_df.to_csv(args.index_file)

	print(f'{args.index_file} has been written. This file can be used to calculate embeddings with `embeddings.py`')
