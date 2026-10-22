import argparse
import sys
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import pandas as pd
import pytest
import torch

from embedders.base import create_parser, make_iterator
from embedders.schema import BatchIterator
from embedders.prottrans import main_prottrans
from embedders.esm3 import main_esm3, embed_esm3_batch


class Tokenizer:
    pad_token_id = 0

    def batch_encode_plus(self, seqs, **kwargs):
        rows = [[50 if t == '<AA2fold>' else ord(t) for t in s.split()] + [1] for s in seqs]
        return self.pack(rows)

    def __call__(self, seqs, **kwargs):
        rows = [[2] + [ord(t) for t in s] + [3] for s in seqs]
        return {'input_ids': torch.tensor(self.pack(rows)['input_ids'])}

    def pack(self, rows):
        n = max(map(len, rows))
        return {'input_ids': [r + [0] * (n - len(r)) for r in rows],
                'attention_mask': [[1] * len(r) + [0] * (n - len(r)) for r in rows]}


class T5Model:
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))


class ESM3Model(T5Model):
    tokenizers = SimpleNamespace(sequence=Tokenizer())

    def __call__(self, sequence_tokens, sequence_id, function_tokens,
                 residue_annotation_tokens, structure_coords):
        assert torch.isnan(structure_coords).all()
        assert not function_tokens.any() and not residue_annotation_tokens.any()
        mask = sequence_id == 0
        mean = (sequence_tokens * mask).sum(1) / mask.sum(1)
        emb = sequence_tokens.float() + mean[:, None]
        return SimpleNamespace(embeddings=emb.unsqueeze(-1))


def args_for(path, embedder, mode='pt', nproc=1):
    return argparse.Namespace(output=str(path), embedder=embedder, gpu=False,
                              use_fastt5=False, nproc=nproc,
                              asdir=mode == 'dir', h5py=mode == 'h5')


def test_parser_defaults_and_new_models():
    for model in [None, 'pt', 'prost', 'prostt5', 'esm', 'esm3']:
        argv = ['embeddings.py', 'start', 'input.fasta', 'out']
        if model:
            argv += ['-e', model]
        with patch.object(sys, 'argv', argv):
            args = create_parser()
        assert args.embedder == (model or 'pt')
        assert args.batch_size == 32 and args.nproc == 1


@pytest.mark.parametrize('embedder', ['pt', 'prost', 'prostt5'])
def test_t5_residue_alignment(tmp_path, embedder):
    df = pd.DataFrame({'sequence': ['AUOBZ', 'CD'], 'seqlens': [5, 2]})
    args = args_for(tmp_path / 'out.pt', embedder)
    with patch('embedders.prottrans.T5Tokenizer.from_pretrained', return_value=Tokenizer()), patch('embedders.prottrans.T5EncoderModel.from_pretrained', return_value=T5Model()):
        main_prottrans(df, args, BatchIterator([slice(0, 2)]))
    embeddings = torch.load(args.output)
    expected = ['AUOBZ', 'CD'] if embedder == 'pt' else ['AXXXX', 'CD']
    for emb, seq in zip(embeddings, expected):
        assert emb[:, 0].tolist() == [ord(c) for c in seq]


@pytest.mark.parametrize('mode', ['pt', 'dir', 'h5'])
@pytest.mark.parametrize('nproc', [1, 2])
def test_esm3_outputs(tmp_path, mode, nproc):
    if mode == 'pt' and nproc == 2:
        return
    fake_module = SimpleNamespace(ESM3=SimpleNamespace(from_pretrained=lambda *a, **k: ESM3Model()))
    df = pd.DataFrame({'sequence': ['ACDE', 'FG', 'HI', 'KLM'], 'seqlens': [4, 2, 2, 3]})
    output = tmp_path / ('out.h5' if mode == 'h5' else 'out')
    args = args_for(output, 'esm3', mode, nproc)
    batches = make_iterator(df.seqlens.tolist(), 0, 8)
    with patch.dict(sys.modules, {'esm.models.esm3': fake_module}):
        for rank in range(nproc):
            iterator = BatchIterator(batches)
            if nproc > 1:
                iterator.set_local_rank(rank, nproc)
            main_esm3(df, args, iterator, rank_id=rank)
    if mode == 'pt':
        actual = torch.load(output)
    elif mode == 'dir':
        actual = [torch.load(output / (str(i) + '.emb')).float() for i in range(4)]
    else:
        with h5py.File(output) as hf:
            actual = [torch.from_numpy(hf['embeddings']['emb_' + str(i)][:]) for i in range(4)]
    for seq, emb in zip(df.sequence, actual):
        expected = embed_esm3_batch(ESM3Model(), [seq], torch.device('cpu'))[0]
        torch.testing.assert_close(emb, expected, atol=0.05 if mode == 'dir' else 0, rtol=0)


def test_esm3_resume_and_dispatch(tmp_path):
    import embeddings
    args = args_for(tmp_path / 'out', 'esm3', 'dir', 2)
    df = pd.DataFrame({'sequence': ['AC'], 'seqlens': [2]})
    with patch.dict('os.environ'), patch.object(embeddings.dist, 'init_process_group'), patch.object(embeddings, 'validate_args', return_value=(args, df)), patch.object(embeddings, 'prepare_dataframe', return_value=(df, None)), patch.object(embeddings, 'main_esm3') as call:
        embeddings.mp_process(1, 2, args)
    call.assert_called_once_with(df, args, None, rank_id=1)
    iterator = BatchIterator([slice(0, 1), slice(1, 2)], start_batch=1)
    assert list(iterator) == [(1, slice(1, 2))]


@pytest.mark.parametrize('mode', ['dir', 'h5'])
@pytest.mark.parametrize('parallel', [False, True])
def test_checkpoint_paths(tmp_path, monkeypatch, mode, parallel):
    import json
    from embedders.checkpoint import find_and_load_checkpoint_file, checkpoint_from_json
    output = tmp_path / ('out' if mode == 'dir' else 'out.h5')
    if mode == 'dir':
        output.mkdir()
    else:
        output.touch()
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    paths = []
    for rank in range(2 if parallel else 1):
        name = 'emb_checkpoint_mp_%d.json' % rank if parallel else 'emb_checkpoint.json'
        path = output / name if mode == 'dir' else output.with_name(output.name + '_' + name)
        path.write_text(json.dumps({'embedder': 'esm3', 'nproc': 2 if parallel else 1, 'last_batch': rank + 2}))
        paths.append(path)
    found = find_and_load_checkpoint_file(str(output))
    assert found.embedder == 'esm3' and found.last_batch == 2
    for rank, path in enumerate(paths):
        if parallel:
            monkeypatch.setenv('WORLD_SIZE', '2')
            monkeypatch.setenv('LOCAL_RANK', str(rank))
        assert checkpoint_from_json(str(output)).last_batch == rank + 2
        assert checkpoint_from_json(str(path)).last_batch == rank + 2
        if parallel:
            assert checkpoint_from_json(str(paths[0])).last_batch == rank + 2
