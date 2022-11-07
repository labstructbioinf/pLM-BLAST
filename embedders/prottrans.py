import os
import re
import argparse
import warnings
import tempfile
import shutil
import gc

import pandas as pd
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5EncoderModel

from .parser import save_as_separate_files
regex_aa = re.compile(r"[UZOB]")
EMBEDDER = 'Rostlab/prot_t5_xl_half_uniref50-enc'


def main_prottrans(df: pd.DataFrame, args: argparse.Namespace, num_batches: int):
    print('loading model')
    tokenizer = T5Tokenizer.from_pretrained(EMBEDDER, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(EMBEDDER, torch_dtype=torch.float32)
    # set device
    if args.gpu:
        device = torch.device('cuda')
        model.to(device)
    else:
        device = torch.device('cpu')
    model.eval()
    gc.collect()
    if df.seqlens.max() > 1000:
        warnings.warn('''dataset poses sequences longer then 1000 aa, this may lead to memory overload and long running time''')
    batch_files = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        for batch_id in tqdm(range(num_batches)):
            seqlist = []
            lenlist = []
            batch_index = []
            for i, (idx, row) in enumerate(df.iterrows()):
                # use only current batch sequences
                if batch_id*args.batch_size <= i < (batch_id + 1)*args.batch_size:
                    sequence = row.seq
                    sequence = regex_aa.sub("X", row.seq)
                    sequence_len = len(sequence)
                    lenlist.append(sequence_len)
                    sequence = " ".join(list(sequence))
                    seqlist.append(sequence)
                    batch_index.append(idx)
                
            ids = tokenizer.batch_encode_plus(seqlist, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device, non_blocking=True)
            attention_mask = torch.tensor(ids['attention_mask']).to(device, non_blocking=True)
            
            with torch.no_grad():
                embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = embeddings[0].float().cpu()
            # remove sequence padding
            num_batch_embeddings = len(embeddings)
            assert num_batch_embeddings == len(seqlist)
            embeddings_filt = []
            for i in range(num_batch_embeddings):
                seq_len = lenlist[i]
                emb = embeddings[i]
                if emb.shape[1] < seq_len:
                    raise KeyError(f'sequence is longer then embedding {emb.shape[1]} and {seq_len} ')       
                embeddings_filt.append(emb[:seq_len])
            # store each batch depending on save mode
            if args.asdir:
                save_as_separate_files(embeddings_filt, batch_index=batch_index, directory=tmpdirname)
            else:
                batch_id_filename = os.path.join(tmpdirname, f"emb_{batch_id}")
                torch.save(embeddings_filt, batch_id_filename)
                batch_files.append(batch_id_filename)
        # creating output
        if args.asdir:
            print(f'copying embeddings to: {args.output}')
            if not os.path.isdir(args.output):
                os.mkdir(args.output)
            for file in os.listdir(tmpdirname):
                shutil.copy(src=os.path.join(tmpdirname, file), dst=os.path.join(args.output, file))
        # merge batch_data if `asdir` is false
        else:
            stack = []
            for fname in batch_files:
                stack.extend(torch.load(fname))
            torch.save(stack, args.output)