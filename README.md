# pLM-BLAST

pLM-BLAST is a sensitive remote homology detection tool based on the comparison of residue embeddings obtained from protein language models such as ProtTrans5. It is available as a stand-alone package as well as an easy-to-use web server within the [MPI Bioinformatics Toolkit](https://toolkit.tuebingen.mpg.de/tools/psiblast), where pre-computed databases can be searched.

## Table of contents
* [ Installation ](#Installation)
* [ Usage ](#Usage)
* [ Remarks ](#Remarks)

## Installation
For local use, use the `requirements.txt` or `environment.yml` file to create a an environment. \

**pip**

```
pip install -r requirements.txt
```
**conda**
```
conda env create -f environment.yml
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
python ../pLM-BLAST/scripts/run_plm_blast.py database query output.csv -use_chunks
```

Note that only the base filename should be specified for the query. The `-use_chunks` option enables the use of chunk cosine similarity pre-screening. Please follow `scripts/example.sh` for more examples and run `run_plm_blast.py -h` for more options.


### Use in Python
```python
import torch
from alntools.base import Extractor
import os

fasta_file = './scripts/input/cupredoxin.fas'
emb_file = './scripts/output/cupredoxin.pt'
emb_scr = './embeddings.py'

os.system(f'python {emb_scr} {fasta_file} {emb_file}')

emb_file = './scripts/output/cupredoxin.pt'
embs = torch.load(emb_file)

# a self-comparison will be performed
seq1_emb, seq2_emb = embs[0].numpy(), embs[0].numpy()

# all at once - local alignments
extr = Extractor()
results = extr.full_compare(seq1_emb, seq2_emb)

#all at once - global alignment
extr.BFACTOR = 'global'
# one alignment per protein pair
results = extr.embedding_to_span(seq1_emb, seq2_emb)
```


## Remarks

### How to cite?
If you find the `pLM-BLAST` useful, please cite the preprint:

"*pLM-BLAST – distant homology detection based on direct comparison of sequence representations from protein language models*" \
Kamil Kaminski, Jan Ludwiczak, Vikram Alva, and Stanislaw Dunin-Horkawicz \
bioRxiv https://www.biorxiv.org/content/10.1101/2022.11.24.517862v1

### Contact
If you have any questions, problems or suggestions, please contact [us](https://ibe.biol.uw.edu.pl/en/835-2/research-groups/laboratory-of-structural-bioinformatics/).

### Funding
This work was supported by the First TEAM program of the Foundation for Polish Science co-financed by the European Union under the European Regional Development Fund.


