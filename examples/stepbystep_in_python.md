```python
import sys
import os

import torch
from Bio import SeqIO

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import alntools as aln

def compare(emb1, emb2, win, span, gap, bfactor, sigma):
    """
    Compares two embeddings emb1 and emb2 using the provided parameters and 
    returns detailed results.
    """
    module = aln.Extractor( 
                    enh=False,
                    norm=False, 
                    bfactor=bfactor,
                    sigma_factor=sigma,
                    gap_penalty=gap,
                    min_spanlen=span,
                    window_size=win,
					filter_results=True
    )

    results, densitymap, paths, scorematrix = module.embedding_to_span(emb1, emb2, mode='all')

    if module.FILTER_RESULTS:
        results = aln.postprocess.filter_result_dataframe(results)
    
    return results, densitymap, paths, scorematrix


if __name__ == '__main__':

	# read embeddings and sequences
	emb_file1 = 'data/output/cupredoxin.pt'
	emb_file2 = 'data/output/immunoglobulin.pt'
	
	emb1 = torch.load(emb_file1)[0].numpy()
	emb2 = torch.load(emb_file2)[0].numpy()
	
	seq = list(SeqIO.parse('data/input/cupredoxin.fas', format='fasta'))
	seq1 = str(seq[0].seq)

	seq = list(SeqIO.parse('data/input/immunoglobulin.fas', format='fasta'))
	seq2 = str(seq[0].seq)

	# define parameters for local alignment
	params = 15, 25, 0.5, 2, 2.5 # win, span, gap, bfactor, sigma, enh

	# run comparision
	results, densitymap, paths, scorematrix = compare(emb1, emb2, *params)
	
	# results contains all the detected alignments
	print(results)
	
	# pick best hit
	row = results.iloc[results.score.argmax()]
	
	# print alignment
	aln = aln.draw_alignment(row.indices, seq2, seq1, output='str')
								
	print(aln)
	
```
