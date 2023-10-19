# pLM-BLAST

pLM-BLAST is a sensitive remote homology detection tool based on the comparison of residue embeddings obtained from protein language models such as ProtTrans5. It is available as a stand-alone package as well as an easy-to-use web server within the [MPI Bioinformatics Toolkit](https://toolkit.tuebingen.mpg.de/tools/plmblast), where pre-computed databases can be searched.

## Table of contents
* [ Installation ](#Installation)
* [ Usage ](#Usage)
    + [Databases](#databases)
    + [Searching a database](#searching-a-database)
    + [Use in Python](#use-in-python)
* [ Remarks ](#Remarks)
    + [How to cite](#how-to-cite)
    + [Funding](#funding)
    + [Contact](#contact)
    + [Changelog](#changelog)

# Installation

Create a conda environment:
```bash
conda create --name plmblast python=3.9
conda activate plmblast
```

Install pip in the environment:
```bash
conda install pip
```

Install pLM-BLAST using `requirements.txt`:
```bash
pip install -r requirements.txt
```

# Usage
## Databases

Pre-computed databases can be downloaded from http://ftp.tuebingen.mpg.de/pub/protevo/toolkit/databases/plmblast_dbs. 

The `embeddings.py` script can be used to create a custom database from a CSV or FASTA file. For example, the first lines of the CSV file for the ECOD database are:

```
,id,description,sequence
0,ECOD_000151743_e4aybQ1,"ECOD_000151743_e4aybQ1 | 4146.1.1.2 | 4AYB Q:33-82 | A: alpha bundles, X: NO_X_NAME, H: NO_H_NAME, T: YqgQ-like, F: RNA_pol_Rpo13 | Protein: DNA-DIRECTED RNA POLYMERASE",FPKLSIQDIELLMKNTEIWDNLLNGKISVDEAKRLFEDNYKDYEKRDSRR
1,ECOD_000399743_e3nmdE1,"ECOD_000399743_e3nmdE1 | 5027.1.1.3 | 3NMD E:3-53 | A: extended segments, X: NO_X_NAME, H: NO_H_NAME, T: Preprotein translocase SecE subunit, F: DD_cGKI-beta | Protein: cGMP Dependent PRotein Kinase",LRDLQYALQEKIEELRQRDALIDELELELDQKDELIQMLQNELDKYRSVI
2,ECOD_002164660_e6atuF1,"ECOD_002164660_e6atuF1 | 927.1.1.1 | 6ATU F:8-57 | A: few secondary structure elements, X: NO_X_NAME, H: NO_H_NAME, T: Elafin-like, F: WAP | Protein: Elafin",PVSTKPGSCPIILIRCAMLNPPNRCLKDTDCPGIKKCCEGSCGMACFVPQ
```

If the input file is in CSV format, use `-cname` to specify in which column the sequences are stored. 

It is recommended to sort the input sequences by length.

```bash
# CSV input
python embeddings.py start database.csv database -embedder pt -cname sequence --gpu -bs 0 --asdir
# FASTA input
python embeddings.py start database.fasta database -embedder pt --gpu -bs 0 --asdir
```

In the examples above, `database` defines a directory where sequence embeddings are stored.

The batch size (number of sequences per batch) is set with the `-bs` option. Setting `-bs` to `0` activates the adaptive mode, in which the batch size is set so that all included sequences have no more than 3000 residues (this value can be changed with `--res_per_batch`).

The use of `--gpu` is highly recommended for large datasets. To run `.embeddings.py` on multiple GPUs, specify `-proc X` where `X` is the number of GPU devices you want to use.

The last step is to create an additional file with flattened embeddings for the chunk cosine similarity scan, a procedure used to speed up database searches. To do this, use the `dbtofile.py` script with the database name as the only parameter:

```bash
python scripts/dbtofile.py database 
```

A new file `emb.64` will appear in the database directory. The database is now ready for use.

### Checkpointing feature

When dealing with large databases, it may be helpful to resume previously stopped or interrupted computations. When `embeddings.py` encounters an exception or keyboard interrupt, the main process captures the actual computation steps in the checkpoint file. If you want to resume, type:

```bash
python embeddings.py resume database
``` 
where `database` is the output directory for interrupted calculations.

## Searching a database

To search the database `database` with a FASTA sequence stored in `query.fas`, a query embedding must be computed:

```bash
python embeddings.py query.fas query.pt
```

Then the `plmblast.py` script can be used to search the database:

```bash
python ./scripts/plmblast.py database query output.csv --use_chunks
```
Note that only the base filename should be specified for the query (`csv` and `pt` extensions are automatically added). The `--use_chunks` option enables the use of cosine similarity pre-screening, which improves search speed. Follow `scripts/example.sh` for more examples and run `plmblast.py -h` for more options. 

## Use in Python

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
import alntools as aln
import pandas as pd
from Bio import SeqIO

# Get embeddings and sequences
emb_file = './scripts/output/cupredoxin.pt'
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
spans_locations = aln.prepare.search_paths(sub_matrix, paths=paths, window=window, sigma_factor=sigma_factor, mode='local' if bfactor==1 else 'global', min_span=min_span)
							
results = pd.DataFrame(spans_locations.values())
results['i'] = 0
# remove redundant hits
results = aln.postprocess.filter_result_dataframe(results, column='score')

# Print best alignment
row = results.iloc[0]

aln = aln.alignment.draw_alignment(row.indices, seq1, seq2, output='str')

print(aln)
```

# Remarks

## How to cite?
If you find the `pLM-BLAST` useful, please cite:

"*pLM-BLAST – distant homology detection based on direct comparison of sequence representations from protein language models*" \
Kamil Kaminski, Jan Ludwiczak, Kamil Pawlicki, Vikram Alva, and Stanislaw Dunin-Horkawicz \
bioinformatics https://doi.org/10.1093/bioinformatics/btad579

## Contact
If you have any questions, problems, or suggestions, please contact [us](https://ibe.biol.uw.edu.pl/en/835-2/research-groups/laboratory-of-structural-bioinformatics/).

## Funding
This work was supported by the First TEAM program of the Foundation for Polish Science co-financed by the European Union under the European Regional Development Fund.

# Changelog

* 26/09/2023 improved embedding extraction script, calculations can now be resumed if interrupted, see databases section for more info.
* 26/09/2023 improved adaptive batching strategy for `-bs 0` option, batch size is now divisible by 4 for better performance and `-res_per_batch` options have been added.
* 9/10/2023 added support for `hdf5` files for embedding generation, soon we will add support for `run_plmblast.py` script.
* 9/10/2023 added multi-processing feature to embedding generation, `-nproc X` options will now spawn `X` independent processes.
