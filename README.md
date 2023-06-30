# pLM-BLAST

pLM-BLAST is a sensitive remote homology detection tool that is based on the comparison of residue embeddings obtained from the protein language model ProtTrans5. \
It is available as a standalone package as well as an easy-to-use web server within the [MPI Bioinformatics Toolkit](https://toolkit.tuebingen.mpg.de/tools/psiblast), where several precomputed databases (e.g., ECOD, InterPro, and PDB) can be searched.

Note: the method is being actively developed, please expect soon new features, such as speed improvement, global alignment, and others.

## Table of contents
* [ Installation ](#Installation)
* [ Usage ](#Usage)
* [ Remarks ](#Remarks)

## Installation
For the local use, use the `requirements.txt` file or `environment.yml` to create a new conda environment.  \
**pip**

```
pip install -r requirements.txt
```
**conda**
```
conda env create -f environment.yml
```
Alternatively, the packages listed below can be installed manually: 
```
python==3.9
Bio==1.5.9
fairscale==0.4.13
matplotlib==3.7.1
numba==0.57.1
pandas==2.0.2
numpy==1.24.3
pytest==7.4.0
scikit_learn==1.2.2
torch==2.0.1
tqdm==4.65.0
transformers==4.30.2
sentencepiece==0.1.99
```

## Usage
### Databases

Pre-calculated databases can be downloaded from http://ftp.tuebingen.mpg.de/pub/protevo/toolkit/databases/plmblast_dbs. 

To create a custom database, use `embeddings.py` script:

```
embeddings.py fasta.fas output_file.pt
```
or
```
embeddings.py database.csv database -embedder pt -cname column_name --gpu -bs -1 --asdir
```

`database.csv` is an index file defining sequences and their descriptions. 
For example, the first lines of the ECOD database index are:

```
,id,description,sequence
0,ECOD_000151743_e4aybQ1,"ECOD_000151743_e4aybQ1 | 4146.1.1.2 | 4AYB Q:33-82 | A: alpha bundles, X: NO_X_NAME, H: NO_H_NAME, T: YqgQ-like, F: RNA_pol_Rpo13 | Protein: DNA-DIRECTED RNA POLYMERASE",FPKLSIQDIELLMKNTEIWDNLLNGKISVDEAKRLFEDNYKDYEKRDSRR
1,ECOD_000399743_e3nmdE1,"ECOD_000399743_e3nmdE1 | 5027.1.1.3 | 3NMD E:3-53 | A: extended segments, X: NO_X_NAME, H: NO_H_NAME, T: Preprotein translocase SecE subunit, F: DD_cGKI-beta | Protein: cGMP Dependent PRotein Kinase",LRDLQYALQEKIEELRQRDALIDELELELDQKDELIQMLQNELDKYRSVI
2,ECOD_002164660_e6atuF1,"ECOD_002164660_e6atuF1 | 927.1.1.1 | 6ATU F:8-57 | A: few secondary structure elements, X: NO_X_NAME, H: NO_H_NAME, T: Elafin-like, F: WAP | Protein: Elafin",PVSTKPGSCPIILIRCAMLNPPNRCLKDTDCPGIKKCCEGSCGMACFVPQ
```
Index can be generated from a FASTA file using `scripts/makeindex.py`.

Program will generate directory `database` in which each file is a separate sequence embedding. `bs -1` for adaptive batch size - especially helpful when using `--gpu`.
 
Use `-cname` to specify in which column of the `database.csv` file sequences are stored \
The resulting embeddings will be stored in `database.pt` \
Usage of `--gpu` is highly recommended (cpu calculations are orders of magnitude slower)

### Searching a database

To search a pre-calculated or custom database, follow `scripts/example.sh` 

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


