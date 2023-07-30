# pLM-BLAST

pLM-BLAST is a sensitive remote homology detection tool based on the comparison of residue embeddings obtained from protein language models such as ProtTrans5. It is available as a stand-alone package as well as an easy-to-use web server within the [MPI Bioinformatics Toolkit](https://toolkit.tuebingen.mpg.de/tools/plmblast), where pre-computed databases can be searched.

## Table of contents
* [ Installation ](#Installation)
* [ Usage ](#Usage)
* [ Remarks ](#Remarks)

## Installation
For local use, use the `requirements.txt` file to create an environment.

**pip**

Create a new conda environment:
```
conda create --name plmblast python=3.9
conda activate plmblast
```

Install pip in the environment:
```
conda install pip
```

Install pLM-BLAST (note to use pip from the environment, not the globally installed one):
```
pip install -r requirements.txt
```

## Usage
### Databases

Pre-computed databases can be downloaded from http://ftp.tuebingen.mpg.de/pub/protevo/toolkit/databases/plmblast_dbs. 

To create a custom database, use the `embeddings.py` script and an index file that defines sequences and their descriptions. For example, the first lines of the ECOD database index are shown below:

```
,id,description,sequence
0,ECOD_000151743_e4aybQ1,"ECOD_000151743_e4aybQ1 | 4146.1.1.2 | 4AYB Q:33-82 | A: alpha bundles, X: NO_X_NAME, H: NO_H_NAME, T: YqgQ-like, F: RNA_pol_Rpo13 | Protein: DNA-DIRECTED RNA POLYMERASE",FPKLSIQDIELLMKNTEIWDNLLNGKISVDEAKRLFEDNYKDYEKRDSRR
1,ECOD_000399743_e3nmdE1,"ECOD_000399743_e3nmdE1 | 5027.1.1.3 | 3NMD E:3-53 | A: extended segments, X: NO_X_NAME, H: NO_H_NAME, T: Preprotein translocase SecE subunit, F: DD_cGKI-beta | Protein: cGMP Dependent PRotein Kinase",LRDLQYALQEKIEELRQRDALIDELELELDQKDELIQMLQNELDKYRSVI
2,ECOD_002164660_e6atuF1,"ECOD_002164660_e6atuF1 | 927.1.1.1 | 6ATU F:8-57 | A: few secondary structure elements, X: NO_X_NAME, H: NO_H_NAME, T: Elafin-like, F: WAP | Protein: Elafin",PVSTKPGSCPIILIRCAMLNPPNRCLKDTDCPGIKKCCEGSCGMACFVPQ
```

The index file can be generated from a FASTA file using `scripts/makeindex.py`:

```
python makeindex.py database.fas database.csv 
```

Now you can use the `embeddings.py` script to create a database. Use `-cname` to specify in which column of the `database.csv` file the sequences are stored.

```
embeddings.py database.csv database -embedder pt -cname sequence --gpu -bs -1 --asdir
```

It will create a directory `database` in which each file is a separate sequence embedding. Use `bs -1` for adaptive batch size when using `--gpu`. The use of `--gpu` is highly recommended.

The last step is to create an additional file with flattened embeddings for the chunk cosine similarity scan, a procedure used to speed up database searches. To do this, use the `dbtofile.py` script with the database name as the only parameter:

```
python dbtofile.py database 
```

A new file `emb.64` should appear in the database directory.

### Searching a database

Suppose we want to search the database `database` with a FASTA sequence stored in `query.fas`. First, we need to create an index file for the query:

```
python makeindex.py query.fas query.csv
```

Then an embedding for the query:

```
embeddings.py query.fas query.pt
```

Finally, the `run_plm_blast.py` script can be used to search the database:

```
python ./scripts/run_plm_blast.py database query output.csv -use_chunks
```

Note that only the base filename should be specified for the query. The `-use_chunks` option enables the use of chunk cosine similarity pre-screening. Please follow `scripts/example.sh` for more examples and run `run_plm_blast.py -h` for more options.


### Use in Python

pLM-BLAST can also be used in Python scripts. 

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
extr = Extractor()
extr.FILTER_RESULTS = True # Removes redundant paths
results = extr.full_compare(seq1_emb, seq2_emb)

print(results)

# Create a single global alignment
extr.BFACTOR = 'global'
# one alignment per protein pair
results = extr.embedding_to_span(seq1_emb, seq2_emb)

print(results)
```

Advanced example:

```python
import torch
import alntools.density as ds
import alntools as aln
import pandas as pd
from Bio import SeqIO

# Get embeddings and sequences
emb_file = './scripts/output/cupredoxin.pt'
embs = torch.load(emb_file)
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
column = 'score' # Another option is "len"

# Run pLM-BLAST
densitymap = ds.embedding_similarity(emb1, emb2)
arr = densitymap.cpu().numpy()

paths = aln.alignment.gather_all_paths(densitymap, gap_opening=gap_opening, bfactor=bfactor)

spans_locations = aln.prepare.search_paths(arr, paths=paths, window=window, sigma_factor=sigma_factor, mode='local' if bfactor==1 else 'global', min_span=min_span)
							
results = pd.DataFrame(spans_locations.values())
results['i'] = 0
results = aln.postprocess.filter_result_dataframe(results, column='score')

# Print best alignment
row = results.iloc[0]

aln = aln.alignment.draw_alignment(row.indices, seq1, seq2, output='str')

print(aln)
```


## Remarks

### How to cite?
If you find the `pLM-BLAST` useful, please cite the preprint:

"*pLM-BLAST – distant homology detection based on direct comparison of sequence representations from protein language models*" \
Kamil Kaminski, Jan Ludwiczak, Kamil Pawlicki, Vikram Alva, and Stanislaw Dunin-Horkawicz \
bioRxiv https://www.biorxiv.org/content/10.1101/2022.11.24.517862v1

### Contact
If you have any questions, problems, or suggestions, please contact [us](https://ibe.biol.uw.edu.pl/en/835-2/research-groups/laboratory-of-structural-bioinformatics/).

### Funding
This work was supported by the First TEAM program of the Foundation for Polish Science co-financed by the European Union under the European Regional Development Fund.


