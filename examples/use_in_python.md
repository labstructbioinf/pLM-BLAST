
Simple example:

```python
import torch
from alntools.base import Extractor
import os

emb_file = './scripts/output/cupredoxin.pt'
embs = torch.load(emb_file)

# A self-comparison is performed
seq1_emb, seq2_emb = embs[0].numpy(), embs[0].numpy()

# Create multiple local alignments
extr = Extractor(filter_results=True)
results = extr.full_compare(seq1_emb, seq2_emb)

print(results)
# Create a single global alignment
extr.BFACTOR = 'global'
# one alignment per protein pair
results = extr.embedding_to_span(seq1_emb, seq2_emb)

print(results)
```