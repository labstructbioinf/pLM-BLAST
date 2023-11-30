## scope databaset benchmark

download database
```bash
wget -O data/input/scope40.fas https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
```

calculate database
```bash
python embeddings.py start examples/data/input/scope40.fas examples/data/output/scope40  -embedder pt --gpu -bs 0 --asdir -t 1500
```
