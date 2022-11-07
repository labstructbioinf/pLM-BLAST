'''generate embeddings from sequence'''
import re
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
import torch


regex_aa = re.compile(r"[UZOB]")
EMBEDDER = 'Rostlab/prot_t5_xl_half_uniref50-enc'
BATCH_SIZE = 64


parser = argparse.ArgumentParser(description =  
    f'''Embedding script \ncreate embeddings from sequences via {EMBEDDER}'''
    )
parser.add_argument('-i', '-in', help='csv/pickle (.csv or .p) with `seq` column',
                    required=True, dest='infile', type=str)
parser.add_argument('-o', '-out', help='resulting list of embeddings',
                    required=True, dest='outfile', type=str)
parser.add_argument('-cname', help='custom sequence column name',
                     dest='cname', type=str, default='seq')
args = parser.parse_args()

if args.infile.endswith('csv'):
    df = pd.read_csv(args.infile)
elif args.infile.endswith('.p'):
    df = pd.read_pickle(args.infile)
else:
    raise FileNotFoundError(f'invalid input infile extension {args.infile}')

if args.cname != '':
    if args.cname not in df.columns:
        raise KeyError(f'no column: {args.cname} available in file: {args.infile}')
    else:
        # case when cname == seq
        if 'seq' in df.columns:
            df.rename(columns={'seq': 'seq_backup'}, inplace=True)
        df.rename(columns={args.cname: 'seq'}, inplace=True)

print('loading models')
tokenizer = T5Tokenizer.from_pretrained(EMBEDDER, do_lower_case=False)
model = T5EncoderModel.from_pretrained(EMBEDDER, torch_dtype=torch.float32)
num_records = df.shape[0]
residues = num_records % BATCH_SIZE
num_batches = int(num_records/BATCH_SIZE)

if residues > 0:
    num_batches += 1 
embedding_stack = list()
for batch_id in tqdm(range(num_batches)):
    seqlist = []
    lenlist = []
    for i, (idx, row) in enumerate(df.tail(num_records - batch_id*BATCH_SIZE).iterrows()):
        if i > 0 and i % BATCH_SIZE:
            continue
        sequence = regex_aa.sub("X", row.seq)
        lenlist.append(len(sequence))
        sequence = " ".join(list(sequence))
        seqlist.append(sequence)
        
    ids = tokenizer.batch_encode_plus(seqlist, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])
    
    with torch.no_grad():
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = embeddings[0].float().cpu()
    # remove sequence padding
    embeddings = [emb[:seq_len].T for emb, seq_len in zip(embeddings, lenlist)]
    embedding_stack.extend(embeddings)

torch.save(embeddings, args.outfile)