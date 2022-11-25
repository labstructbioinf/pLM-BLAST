# pLM-BLAST

pLM-BLAST is a sensitive remote homology detection tool that is based on the comparison of residue embeddings obtained from the protein language model ProtTrans5. It is available as a standalone package as well as an easy-to-use web server within the MPI Bioinformatics Toolkit, where several precomputed databases (e.g., ECOD, InterPro, and PDB) can be searched.

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
pandas
pytorch
scikit-learn
biopython 
tqdm
numba
transformers
sentencepiece 
matplotlib
```

## Usage
### Databases

Pre-calculated databases can be downloaded from http://ftp.tuebingen.mpg.de/pub/protevo/toolkit/databases/plmblast_dbs. To create a custom database, use `embeddings.py` script:

```
embeddings.py -embedder pt -cname column_name database.csv database.pt_emb.p --gpu
```

`database.csv` is an index file defining sequences and their descriptions. For example, the first lines of the ECOD database index are:

```
,id,description,sequence
0,ECOD_000151743_e4aybQ1,"ECOD_000151743_e4aybQ1 | 4146.1.1.2 | 4AYB Q:33-82 | A: alpha bundles, X: NO_X_NAME, H: NO_H_NAME, T: YqgQ-like, F: RNA_pol_Rpo13 | Protein: DNA-DIRECTED RNA POLYMERASE",FPKLSIQDIELLMKNTEIWDNLLNGKISVDEAKRLFEDNYKDYEKRDSRR
1,ECOD_000399743_e3nmdE1,"ECOD_000399743_e3nmdE1 | 5027.1.1.3 | 3NMD E:3-53 | A: extended segments, X: NO_X_NAME, H: NO_H_NAME, T: Preprotein translocase SecE subunit, F: DD_cGKI-beta | Protein: cGMP Dependent PRotein Kinase",LRDLQYALQEKIEELRQRDALIDELELELDQKDELIQMLQNELDKYRSVI
2,ECOD_002164660_e6atuF1,"ECOD_002164660_e6atuF1 | 927.1.1.1 | 6ATU F:8-57 | A: few secondary structure elements, X: NO_X_NAME, H: NO_H_NAME, T: Elafin-like, F: WAP | Protein: Elafin",PVSTKPGSCPIILIRCAMLNPPNRCLKDTDCPGIKKCCEGSCGMACFVPQ
```

Index can be generated from a FASTA file using `scripts/makeindex.py`. 

Use `-cname` to specify in which column of the `database.csv` file sequences are stored \
The resulting embeddings will be stored in `database.pt_emb.p` \
Usage of `--gpu` is highly recommended (cpu calculations are orders of magnitude slower)

### Searching a database

To search a pre-calculated or custom database, follow `scripts/example.sh` 

### Use in Python
```
import torch
import alntools as aln
from alntools.base import Extractor
import pandas as pd

# load the embedding; for the embedding calculation, refer to `scripts/example.sh`
emb_file = './scripts/output/A9A4Y8.pt_emb.p'
embs = torch.load(emb_file)

# a self-comparison will be performed
seq1_emb, seq2_emb = embs[0], embs[0]

# calculate embedding similarity aka substitution matrix
densitymap = aln.density.embedding_similarity(seq1_emb, seq2_emb)
# convert to numpy array
densitymap = densitymap.cpu().numpy()
# find all alignment possible paths (traceback from borders)
paths = aln.alignment.gather_all_paths(densitymap)
# score those paths
results = aln.prepare.search_paths(densitymap, paths=paths, as_df=True)
# remove redundant hits
results = aln.postprocess.filter_result_dataframe(results)
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


