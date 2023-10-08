import os
from Bio import SeqIO
import pytest
import pandas as pd
import subprocess

import torch

DIR = os.path.dirname(__file__)
PLMBLAST_SCRIPT = os.path.join("scripts/run_plm_blast.py")
PLMBLAST_DB = os.path.join(DIR, 'test_data/db.emb')
PLMBLAST_OUTPUT = os.path.join(DIR, 'test_data/output.csv')
INPUT_FASTA_MUTLTI = os.path.join(DIR, 'test_data/seq.fasta')
INPUT_EMB_MULTI = os.path.join(DIR, 'test_data/seq.emb')
INPUT_FASTA_SINGLE = os.path.join(DIR, 'test_data/seq_single.fasta')
INPUT_EMB_SINGLE = os.path.join(DIR, 'test_data/seq_single.emb')


@pytest.fixture(autouse=True)
def remove_outputs():
	if os.path.isfile(PLMBLAST_OUTPUT):
		os.remove(PLMBLAST_OUTPUT)


@pytest.mark.dependency()
def test_prepare_data():
    data = SeqIO.parse(INPUT_FASTA_MUTLTI, 'fasta')
	# unpack
    data = [[record.description, str(record.seq)] for record in data]
    # single embedding case
    seqlen = len(data[0][1])
    temp_emb = [torch.rand((seqlen, 1024))]
    torch.save(temp_emb, INPUT_FASTA_SINGLE)
    # multi embedding case
    seqlens = ([len(s[1]) for s in data])
    temp_emb_multi = [torch.rand((seqlen, 1024)) for seqlen in seqlens]
    torch.save(temp_emb_multi, INPUT_EMB_MULTI)


@pytest.mark.dependency(depends='test_prepare_data')
@pytest.mark.parametrize('win', ["10", "20", "30"])
@pytest.mark.parametrize('gap_ext', ["0", "0.1"])
def test_single_query(win: str, gap_ext: str):
    
    proc = subprocess.run(["python", PLMBLAST_SCRIPT, PLMBLAST_DB,
                            INPUT_FASTA_SINGLE, PLMBLAST_OUTPUT,
                            "-win", win, '-gap_ext', gap_ext],
            stderr=subprocess.PIPE,
	        stdout=subprocess.PIPE)
    # check process error code
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr, proc.stderr

    assert os.path.isfile(PLMBLAST_OUTPUT)
    output = pd.read_csv(PLMBLAST_OUTPUT)


def test_multi_query(win: str, gap_ext: str):
    pass


def test_single_query_flow():
    pass


def test_multi_query_flow():
    pass