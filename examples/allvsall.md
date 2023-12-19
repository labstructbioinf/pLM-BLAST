# perform all vs all search for a database

As an example data we will use Rossmann-like example sequences `examples/data/input/rossmannsdb.fas`
First we need to obtain sequence embeddings, via:

```bash 
python embeddings.py start examples/data/input/rossmannsdb.fas examples/data/output/rossmannsdb.fas -embedder pt --gpu -bs 0 --asdir
```
### using custom embeddings
plm-Blast supports any type of embeddings with shape: `[seqlen, embdim]`. If you want to use your own embeddings skip above step and generate single
embedding file `examples/data/output/rossmannsdb.pt` which content should be a python list where each element is appropriate embedding of `rossmannsdb.fas` file.
Another way is to create a directory `examples/data/output/rossmanndb` in which each sequence representation is stored as separate embedding with name conevention
`[0-num_sequences].pt`

Make sure that both embeddings and `.fas` file are in the same directory
```bash
cp examples/data/input/rossmannsdb examples/data/output/rossmannsdb
```

Now we are able to perform search
```bash
python scripts/plmblast.py examples/data/output/rossmannsdb examples/data/output/rossmannsdb allvsall.csv --use_chunks
```
All above computations are also available as script here `allvsall.sh`