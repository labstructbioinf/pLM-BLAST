from modulefinder import Module
import os
import re
import warnings
import tempfile
import shutil

from tqdm import tqdm
import pandas as pd
import torch

from .parser import save_as_separate_files
regex_aa = re.compile(r"[UZOB]")
EMBEDDER = 'esm2_t33_650M_UR50D'

def fsdb_wrappered_setup(embedder_name) -> torch.nn.Module:
    # based on https://github.com/guruace/esm2-esm-1v/blob/main/examples/esm2_infer_fairscale_fsdp_cpu_offloading.py
    # initialize the model with FSDP wrapper

    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
    from fairscale.nn.wrap import enable_wrap, wrap

    fsdp_params = dict(
    mixed_precision=True,
    flatten_parameters=True,
    state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
    cpu_offload=True,  # enable cpu offloading
    )
    with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
        model, alphabet = torch.hub.load("facebookresearch/esm:main", embedder_name)
        batch_converter = alphabet.get_batch_converter()
        model.eval()


        # Wrap each layer in FSDP separately
        for name, child in model.named_children():
            if name == "layers":
                for layer_name, layer in child.named_children():
                    wrapped_layer = wrap(layer)
                    setattr(child, layer_name, wrapped_layer)
    model = wrap(model)

    return model, batch_converter


def main_esm(df: pd.DataFrame, args, num_batches):
    
    embedder_name = args.embedder if args.embedder != "esm" else EMBEDDER
    print('loading model: ', embedder_name)
    if embedder_name == 'esm2_t48_15B_UR50D':
        model, batch_converter = fsdb_wrappered_setup(embedder_name)
    else:
        model, alphabet = torch.hub.load("facebookresearch/esm:main", embedder_name)
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic resultsS
    if args.gpu:
        model = model.cuda()
    # stats
    seqlens = df['seq'].apply(len)

    if seqlens.max() > 1000:
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
                    seqlist.append(sequence)
                    batch_index.append(idx)
                
            data = [
            (f'seq_{i}', seq)
            for i, seq in enumerate(seqlist)]
            # encode as batch
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            if args.gpu:
                batch_tokens = batch_tokens.to(device='cuda', non_blocking=True)
            with torch.no_grad():
                repr_layers = 11
                results = model(batch_tokens, repr_layers=[repr_layers], return_contacts=False)
                #print(results['logits'].shape)
                token_representations = results["representations"][repr_layers]
                # expected size
                # [batch_size, seqlen, embdim]
                if token_representations.ndim > 3:
                    token_representations = token_representations.squeeze()
                if args.gpu:
                    token_representations = token_representations.to(device='cpu')
            # remove sequence padding
            num_batch_embeddings = token_representations.shape[0]
            assert num_batch_embeddings == len(seqlist)
            embeddings_filt = []
            # remove batch padding
            for i in range(num_batch_embeddings):
                seq_len = lenlist[i]
                # batch shape [args.batch_size, max_seqlen, 1280]
                emb = token_representations[i, 1 : seq_len + 1, :]
                embeddings_filt.append(emb.clone())

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