
Advanced example:

```python
import torch
import alntools as aln
import pandas as pd
from Bio import SeqIO

# Get embeddings of cupredoxin sequence
# you can generate it via embeddings.py or any other method
emb_file = '../scripts/output/cupredoxin.pt'
embs = torch.load(emb_file).float().numpy()
# A self-comparison is performed
emb1, emb2 = embs[0], embs[0]

seq = list(SeqIO.parse('./scripts/input/cupredoxin.fas', format='fasta'))
seq = str(seq[0].seq)
seq1, seq2 = seq, seq

# Parameters
bfactor = 1 # local alignment
sigma_factor = 2 
window = 10 # scan window length
min_span = 25 # minimum alignment length
gap_opening = 0 # Gap opening penalty
column = 'score' # Another option is "len" column used to sort results

# Run pLM-BLAST
# calculate per residue substitution matrix
sub_matrix = aln.base.embedding_local_similarity(emb1, emb2)
# gather paths from scoring matrix
paths = aln.alignment.gather_all_paths(sub_matrix, gap_opening=gap_opening, bfactor=bfactor)
# seach paths for possible alignment
spans_locations = aln.prepare.search_paths(sub_matrix,
                                             paths=paths,
                                             window=window,
                                             sigma_factor=sigma_factor,
                                             mode='local' if bfactor==1 else 'global',
                                             min_span=min_span)
							
results = pd.DataFrame(spans_locations.values())
# remove redundant hits
results['i'] = 0
results = aln.postprocess.filter_result_dataframe(results, column='score')

# Print best alignment
row = results.iloc[0]

aln = aln.alignment.draw_alignment(row.indices, seq1, seq2, output='str')

print(aln)
```
