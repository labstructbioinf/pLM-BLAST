# localaln

### tldr 

requirements: 
```
pytorch
pandas
numpy
numba
```

* `embedding.py` - generate embeddings from sequences, type `python embeddings.py -h` for use examples 
* `search_parallel.py` script for all vs all search over `datafull.p` rosmmann dataset. Change `limit_records` to increase search range.
* `explore_results.ipynb` - analysis of `search_rossmanns.py` results
* `example_low.ipynb`  usage example with description

### get embeddings

obtain sequence embeddings, use `-cname` to specify in which column sequence is stored if other then `seq`
Type `-h` for more details
```bash
python embeddings.py seq.csv seq.emb
```
### example use as script

use `scripts/example.sh` for query - database search


### example use in python

```
emb_file = # file in which embeddings are stored
csv_file = # coresponding frame
# load both
embs = torch.load(emb_file)
df = pd.read_csv(csv_file)

# calculate similarity
densitymap = aln.density.embedding_similarity(x, y)
# convert to numpy array
densitymap = densitymap.cpu().numpy()

# find all alignment possible paths
paths = aln.alignment.gather_all_paths(densitymap)

# score those paths
spans_locations = aln.prepare.search_paths(densitymap, paths=paths)

# convert to frame
results = pd.DataFrame(spans_locations.values())
# remove redundant alignments                                                    
results = aln.postprocess.filter_result_dataframe(results)
```



### params


`window` - width of moving average sliding window when extracting alignment from the path. Kind of scale parameter - the bigger the window the wider context of the alignment etc. Different window values will lead to various results. 

`min_span` - minimal width of the captured alignment, values below window - don't make sense.

`bfactor` - reside step when extracting paths, values should be integers bigger than zero. Because embedding values are continuous and embedding similarities are rather smooth there is no need in extracting a path from each possible alignment because close residues will generate almost the same alignments. For large scale runs bfactor > 1 will should improve speed.

`gap_opening` - penalty for gap opening while generating path, this is one to one factor that is added function which defines route.

`gap_extension` - multiplicative factor penalizing gap prolongation, the bigger values the bigger penalty for creating long gaps etc.




