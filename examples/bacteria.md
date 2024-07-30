# Analysis of Nostoc punctiforme PCC 73102 Bacterial Proteome Using pLM-BLAST

## Introduction
In this report, we describe the analysis of the proteome of the bacterium Nostoc punctiforme PCC 73102 using the pLM-BLAST. We utilize a tool that allows us to identify protein homologs in the sequences of this bacterium.

## Tool pLM_BLAST
Our tool is based on an advanced homology search algorithm, enabling the discovery of related proteins, even when sequences differ significantly. The program facilitates the analysis of multiple sequences simultaneously and generates results in a readable format.

## Sequence data

For the analysis of the proteome of Nostoc punctiforme PCC 73102, we collected a set of protein sequences specific to this bacterium from the NCBI database.

## Data Preparation
1. Download protein sequences of Nostoc punctiforme PCC 73102 from the [NCBI database](https://www.ncbi.nlm.nih.gov/datasets/taxonomy/63737/) in FASTA format. It contains 6690 protein sequences.
2. To streamline the analysis process, we've broken down the file into smaller segments, with each segment containing 1000 sequences. You'll find a script named "split_fasta.py" in the scripts folder, which can be employed for this task. (It's possible to conduct the analysis without splitting the files, but please bear in mind that longer sequences will demand more GPU memory. For graphics cards with 11 GB capacity, the maximum sequence length is approximately 3,500 amino acids. Consequently, sequences that exceed this length should be processed using the CPU.)

    ```bash
    python examples/scripts/split_fasta.py examples/data/input/protein.fas protein_split -cs 1000 -ml 3500
    ```
3. The fasta files will be used to calculate embeddings. To do this, you should utilize the 'embeddings.py' script.
    ```bash
    python embeddings.py start examples/data/input/protein_split_1.fas examples/data/output/protein_split_1.pt -embedder pt --gpu -bs 1 -t 6000
    ```
    The utilized flags are as follows:
    * `-embedder` -> This flag specifies which model to use for creation
    * `--gpu` -> With this flag, computations will be performed on the graphics card.
    * `-bs` -> The number of sequences analyzed in each batch (a higher number speeds up the analysis but increases RAM usage).
    * `-t` -> This value determines the maximum embedding size. In our analysis, we should set a value greater than the longest sequence.
    
    For more useful flags, type:
    ```bash
    python embeddings.py start -h
    ```

## Analysis
The data prepared in this manner will be used to search a database for homologous proteins. The database can be prepared independently or downloaded ready-made from http://ftp.tuebingen.mpg.de/pub/protevo/toolkit/databases/plmblast_dbs. In this analysis, we will use the pre-built ECOD30 database. You should use the 'scripts/plmblast.py' script, which searches the database and then returns a CSV file containing similar proteins for each query.

Usage:
```bash
python scripts/plmblast.py /path/to/database/ecod30db_20220902 examples/data/input/protein_split_1 examples/data/output/protein_split_1.hits.csv -cpc 90 -alignment_cutoff 0.25 -sigma_factor 2
```
The utilized flags are as follows:
* `-cpc` -> Percentile cutoff for chunk cosine similarity pre-screening. The lower the value, the more sequences will be passed through the pre-screening procedure and then aligned with the more accurate but slower pLM-BLAST'
* `-alignment_cutoff` -> pLM-BLAST alignment score cut-off aka average cosine similarity of each position in alignment
* `-sigma_factor` -> The Sigma factor defines the greediness of the local alignment search procedure

For more useful flags, type:
```bash
python scripts/plmblast.py -h
```
