import os
import re
import argparse
import warnings
import tempfile
import gc
from typing import Union, List

import pandas as pd
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5EncoderModel

from .parser import save_as_separate_files
from .parser import calculate_adaptive_batchsize
regex_aa = re.compile(r"[UZOB]")
# default embedder
EMBEDDER = 'Rostlab/prot_t5_xl_half_uniref50-enc'


def main_prottrans(df: pd.DataFrame, args: argparse.Namespace, iterator: List[slice]):
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
    if args.asdir and not os.path.isdir(args.output):
        os.mkdir(args.output)
    seqlist_all = df['seq'].tolist()
    lenlist_all = df['seqlens'].tolist()
    with tempfile.TemporaryDirectory() as tmpdirname:
        for batch_id_filename, batchslice in tqdm(enumerate(iterator), total=len(iterator)):
            seqlist = seqlist_all[batchslice]
            lenlist = lenlist_all[batchslice]
            # add empty character between all residues
            # his is mandatory for pt5 embedders
            seqlist = [' '.join(list(seq)) for seq in seqlist]
            batch_index = list(range(batchslice.start, batchslice.stop))
            ids = tokenizer.batch_encode_plus(seqlist, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device, non_blocking=True)
            attention_mask = torch.tensor(ids['attention_mask']).to(device, non_blocking=True)
            with torch.no_grad():
                embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = embeddings.last_hidden_state.float().cpu()
            # remove sequence padding
            num_batch_embeddings = len(embeddings)
            assert num_batch_embeddings == len(seqlist)
            embeddings_filt = []
            for i in range(num_batch_embeddings):
                seq_len = lenlist[i]
                emb = embeddings[i]
                if emb.shape[0] < seq_len:
                    raise KeyError(f'sequence is longer then embedding {emb.shape} and {seq_len} ')       
                embeddings_filt.append(emb[:seq_len])
            # store each batch depending on save mode
            if args.asdir:
                save_as_separate_files(embeddings_filt, batch_index=batch_index, directory=args.output)
            else:
                batch_id_filename = os.path.join(tmpdirname, f"emb_{batch_id_filename}")
                torch.save(embeddings_filt, batch_id_filename)
                batch_files.append(batch_id_filename)
            del embeddings
            del embeddings_filt
            gc.collect()
        # merge batch_data if `asdir` is false
        if not args.asdir:
            stack = []
            for fname in batch_files:
                stack.extend(torch.load(fname))
            torch.save(stack, args.output)