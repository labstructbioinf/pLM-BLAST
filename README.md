# pLM-BLAST

pLM-BLAST is a sensitive remote homology detection tool that is based on the comparison of residue embeddings obtained from the protein language model ProtTrans5. It is available as a standalone package as well as an easy-to-use web server within the MPI Bioinformatics Toolkit, where several precomputed databases (e.g., ECOD, InterPro, and PDB) can be searched.

## Table of contents
* [ Installation ](#Installation)
* [ Usage ](#usage)
* [ Parameters ](#params-explanation)

## Installation
For the local use, please use the requirements.txt file to create a new conda environment:
```
conda create --name <env> --file requirements.txt
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

Pre-calculated databases can be downloaded from (http://ftp.tuebingen.mpg.de/pub/protevo/toolkit/databases/plmblast_dbs)

obtain sequence embeddings from dataframe
use `-cname` to specify in which column residue sequence is stored (default `seq`)  \
use `-embedder` to tell which embedder to use, it will be automatically downloaded if nessesery
specify it's full name or set `pt` for `prot_t5_xl_half_uniref50-enc` and `esm` for `esm2_t33_650M_UR50D`  \
use `-gpu` to utilize cuda device is available  \
example
```bash
python embeddings.py seq.csv seq.emb -cname sequence
```
Type `-h` for more detail description

### other embeddings

Method will work on any sequence based embeddings which can be converted to (n,m) array. To utilize other embedders store them as a python list of `torch.FloatTensor`, above examples should work with them as well.


### scripts
When precalculed embeddings are available use the above
* query - database search `scripts/example.sh`
* all vs all search based on input dataframe and embeddings `scripts/search_frame.py`

### use in python
all steps at once
```python
import torch
import alntools as aln

# only embeddings are needed
emb_file = ...# file in which embeddings are stored
# load both
embs = torch.load(emb_file)
seq1_emb, seq2_emb = embs[0], embs[1]

# all at once 
extractor = aln.Extractor()
results = extractor.embedding_to_span(seq1_emb, seq2_emb)
# remove redundant hits                                                    
results = aln.postprocess.filter_result_dataframe(results)
```
step by step
```python
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


### params explanation

`window` - width of moving average sliding window when extracting alignment from the path. Kind of scale parameter - the bigger the window the wider context of the alignment etc. Different window values will lead to various results. 

`min_span` - minimal width of the captured alignment, values below window - don't make sense.

`bfactor` - reside step when extracting paths, values should be integers bigger than zero. Because embedding values are continuous and embedding similarities are rather smooth there is no need in extracting a path from each possible alignment because close residues will generate almost the same alignments. For large scale runs bfactor > 1 will should improve speed.

**not well tested yet**  \
`gap_opening` - penalty for gap opening while generating path, this is one to one factor that is added function which defines route.

`gap_extension` - multiplicative factor penalizing gap prolongation, the bigger values the bigger penalty for creating long gaps etc.




