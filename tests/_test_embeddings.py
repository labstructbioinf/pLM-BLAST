import os
import subprocess
import pytest

import pandas as pd
import torch as th

DIR = os.path.dirname(__file__)
EMBEDDING_SCRIPT = "embeddings.py"
EMBEDDING_DATA = os.path.join(DIR, "test_data/seq.p")
EMBEDDING_OUTPUT = os.path.join(DIR, "output/seq.emb")


@pytest.mark.parametrize("embedder", ["pt", "esm", "prost"])
@pytest.mark.parametrize("truncate", ["200", "500"])
@pytest.mark.parametrize("batchsize", ['16', '0'])
def test_embedding_generation(embedder, truncate, batchsize):
	if not os.path.isdir("tests/output"):
		os.mkdir("tests/output")
	if not os.path.isfile(EMBEDDING_SCRIPT):
		raise FileNotFoundError(f'no embedder script in: {EMBEDDING_SCRIPT}')
	embdata = pd.read_pickle(EMBEDDING_DATA)
	seqlist = embdata['seq'].tolist()
	proc = subprocess.run(["python", "embeddings.py",
	EMBEDDING_DATA, EMBEDDING_OUTPUT,
	"-embedder", embedder,
	"--truncate", truncate,
	"-bs", batchsize],
	stderr=subprocess.PIPE,
	stdout=subprocess.PIPE)
	# chech process error code
	assert proc.returncode == 0, proc.stderr
	assert proc.stderr, proc.stderr
	# check process output file/dir
	assert os.path.isfile(EMBEDDING_OUTPUT), f'missing embedding output file, {EMBEDDING_OUTPUT} {proc.stderr}'
	# check output consistency
	embout = th.load(EMBEDDING_OUTPUT)
	assert len(embout) == embdata.shape[0], proc.stderr
	# check embedding size of each sequence
	for i in range(embdata.shape[0]):
		emblen = embout[i].shape
		seqlen = len(seqlist[i])
		assert emblen[0] == seqlen, f'{emblen[0]} != {seqlen}, emb full shape: {emblen}'
	# remove output
	os.remove(EMBEDDING_OUTPUT)


def test_checkpointing()