import os
import json
import subprocess
import shutil
import pytest

import h5py
import numpy as np
from Bio import SeqIO
import pandas as pd
import torch as th

from embedders.base import calculate_adaptive_batchsize_div4
from embedders.dataset import HDF5Handle
from embedders.schema import BatchIterator
from embedders.base import calculate_adaptive_batchsize_div4

DIR = os.path.dirname(__file__)
EMBEDDING_SCRIPT: str = "embeddings.py"
EMBEDDING_DATA: os.PathLike = os.path.join(DIR, "test_data/seq.p")
EMBEDDING_FASTA: os.PathLike = os.path.join(DIR, "test_data/seq.fasta")
EMBEDDING_OUTPUT: os.PathLike = os.path.join(DIR, "test_data/output/seq.emb")
EMBEDDING_OUTPUT_DIR: os.PathLike = os.path.join(DIR, 'output')
NUM_EMBEDDING_FILES: int = pd.read_pickle(EMBEDDING_DATA).shape[0]
DEVICE: str = 'cuda' if th.cuda.is_available() else 'cpu'

@pytest.fixture(autouse=True)
def remove_outputs():
	if os.path.isfile(EMBEDDING_OUTPUT):
		os.remove(EMBEDDING_OUTPUT)
	if os.path.isdir(EMBEDDING_OUTPUT_DIR):
		shutil.rmtree(EMBEDDING_OUTPUT_DIR)
		os.mkdir(EMBEDDING_OUTPUT_DIR)

@pytest.mark.dependency()
def test_files():
	assert os.path.isfile(EMBEDDING_DATA)
	assert os.path.isfile(EMBEDDING_SCRIPT)
	assert os.path.isfile(EMBEDDING_FASTA)

@pytest.mark.dependency()
@pytest.mark.parametrize('seqlen_list',
						  [th.randint(50, 1000, size=(10000, )).tolist() for _ in range(5)])
@pytest.mark.parametrize('res_per_batch', 
						 th.randint(1500, 5000, size=(5,)).tolist())
def test_batching(seqlen_list, res_per_batch):
	num_seq = len(seqlen_list)
	batch_list = calculate_adaptive_batchsize_div4(seqlen_list, res_per_batch)
	assert isinstance(batch_list, list)
	assert len(batch_list) > 0
	for batch in batch_list:
		seqlen_batch = seqlen_list[batch]
		assert len(seqlen_batch) > 0, batch
		assert sum(seqlen_batch) <= res_per_batch, batch
		assert len(seqlen_batch) % 4 == 0 or len(seqlen_batch) < 4, batch
		assert batch.stop <= num_seq, batch

#@pytest.mark.dependency(depends=['test_files', 'test_batching'])
@pytest.mark.parametrize("embedder", ["pt", "esm", "prost"])
@pytest.mark.parametrize("truncate", ["200", "500"])
@pytest.mark.parametrize("batchsize", ['16', '0'])
def test_embedding_generation(embedder, truncate, batchsize):
	embdata = pd.read_pickle(EMBEDDING_DATA)
	seqlist = embdata['seq'].tolist()
	# cmd
	proc = subprocess.run(["python", "embeddings.py", "start",
	EMBEDDING_DATA, EMBEDDING_OUTPUT, "-embedder", embedder,
	"-truncate", truncate, "-bs", batchsize],
	stderr=subprocess.PIPE, stdout=subprocess.PIPE)
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


def test_h5py_dataset():
	# write first batch
	num_embs1 = 128
	dataset = [th.rand(120, 512) for _ in range(num_embs1)]
	HDF5Handle(EMBEDDING_OUTPUT).write_batch(dataset, 0)
	# write second batch
	num_embs2 = 256
	dataset = [th.rand(150, 512) for _ in range(num_embs2)]
	HDF5Handle(EMBEDDING_OUTPUT).write_batch(dataset, num_embs1)
	# read
	with h5py.File(EMBEDDING_OUTPUT, 'r') as hf:
		embgroup = hf['embeddings']
		assert len(embgroup.keys()) == num_embs1 + num_embs2
	d = HDF5Handle(EMBEDDING_OUTPUT).read_batch(0, num_embs1 + num_embs2)
	assert len(d) == num_embs1 + num_embs2


def test_batch_spliting():
	
	num_seq = 30000
	seqlen_list = np.random.randint(10, 1000, size=(num_seq, )).tolist()
	batch_list = calculate_adaptive_batchsize_div4(seqlen_list, 3000)
	total_batches = len(batch_list)
	iterator_1 = BatchIterator(batch_list, 0)
	iterator_2 = BatchIterator(batch_list, 0)
	iterator_1.set_local_rank(0, 2)
	iterator_2.set_local_rank(1, 2)

	assert total_batches == (len(iterator_1) + len(iterator_2))
	assert len(iterator_1) > 0
	assert len(iterator_2) > 0
	assert iterator_1.current_batch == 0
	assert iterator_1.current_batch + len(iterator_1) == iterator_2.current_batch
	assert len(iterator_1) == iterator_2.current_batch

	# reconstruct
	re_seq_len = list()
	for i, sl in iterator_1:
		re_seq_len += seqlen_list[sl]
	for i, sl in iterator_2:
		re_seq_len += seqlen_list[sl]
	assert len(seqlen_list) == len(re_seq_len)
	for i in range(num_seq):
		assert seqlen_list[i] == re_seq_len[i], i
	


def test_h5py_feature():
	proc = subprocess.run(["python", "embeddings.py", "start",
	EMBEDDING_DATA, EMBEDDING_OUTPUT, "-embedder", "pt", "-bs", "16", "--h5py"],
	stderr=subprocess.PIPE, stdout=subprocess.PIPE)
	assert proc.returncode == 0, proc.stderr
	assert proc.stderr, proc.stderr
	assert os.path.isfile(EMBEDDING_OUTPUT), proc.stderr
	embdata = pd.read_pickle(EMBEDDING_DATA)
	seqlist = embdata['seq'].tolist()
	embs = HDF5Handle(EMBEDDING_OUTPUT).read_batch(0)
	for i, emb in enumerate(embs):
		assert emb.shape[0] == len(seqlist[i])


#@pytest.mark.dependency(depends=['test_files', 'test_batching'])
@pytest.mark.parametrize("embedder", ["pt", "esm", "prost"])
@pytest.mark.parametrize("truncate", ["200"])
@pytest.mark.parametrize("batchsize", ['16', '0'])
def test_embedding_generation_fasta(embedder: str, truncate: int, batchsize: int):
	# read sequences from fasta file
	data = SeqIO.parse(EMBEDDING_FASTA, 'fasta')
	seq_list = [record.seq for record in data]
	num_seq = len(seq_list)
	proc = subprocess.run(["python", "embeddings.py", "start",
	EMBEDDING_FASTA, EMBEDDING_OUTPUT, "-embedder", embedder,
	"-truncate", truncate, "-bs", batchsize, '--gpu'], # type: ignore
	stderr=subprocess.PIPE, stdout=subprocess.PIPE)
	# chech process error code
	assert proc.returncode == 0, proc.stderr
	assert proc.stderr, proc.stderr
	# check process output file/dir
	assert os.path.isfile(EMBEDDING_OUTPUT), f'missing embedding output file, {EMBEDDING_OUTPUT} {proc.stderr}'
	# check output consistency
	embout = th.load(EMBEDDING_OUTPUT)
	assert len(embout) == num_seq, proc.stderr
	# check embedding size of each sequence
	for i in range(num_seq):
		emblen = embout[i].shape
		seqlen = min(len(seq_list[i]), int(truncate))
		assert emblen[0] == seqlen, f'{emblen[0]} != {seqlen}, emb full shape: {emblen} for index: {i}'


#@pytest.mark.dependency(depends=['test_files', 'test_batching'])
@pytest.mark.parametrize('checkpoint_file',[
	'test_data/emb_checkpoint.json',
	'test_data/emb_checkpoint_middle.json',
	'test_data/emb_checkpoint_end.json'])
def test_checkpointing(checkpoint_file):
	checkpoint_file = os.path.join(DIR, checkpoint_file)
	# move to newdir
	fname = os.path.basename(checkpoint_file)
	new_location = os.path.join(EMBEDDING_OUTPUT_DIR, fname)
	with open(checkpoint_file, 'rt') as fp:
		checkpoint_data = json.load(fp)
	# force output dir
	checkpoint_data['output'] = EMBEDDING_OUTPUT_DIR
	with open(new_location, 'wt') as fp:
		json.dump(checkpoint_data, fp)
	# cmd
	proc = subprocess.run(["python", "embeddings.py", 'resume',
						new_location], stderr=subprocess.PIPE,
						stdout=subprocess.PIPE)
	# chech process error code
	assert proc.returncode == 0, proc.stderr
	assert proc.stderr, proc.stderr
	# search for files
	files = os.listdir(checkpoint_data['output'])
	files = [os.path.join(checkpoint_data['output'], file) for file in files]
	# input file have fixed number if sequenes = 100
	expected_files = NUM_EMBEDDING_FILES - checkpoint_data['last_batch']*checkpoint_data['batch_size']
	found_files = sum([1  for file in files if file.endswith('.emb')])
	assert expected_files == found_files, proc.stdout


#@pytest.mark.dependency(depends=['test_files', 'test_batching'])
@pytest.mark.parametrize('embedder',['pt'])
def test_parallelism(embedder):
	assert th.cuda.device_count() > 1, 'cannot run test'
		
	embdata = pd.read_pickle(EMBEDDING_DATA)
	seqlist = embdata['seq'].tolist()
	# cmd
	proc = subprocess.run(["python", "embeddings.py", "start",
	EMBEDDING_DATA, EMBEDDING_OUTPUT, "-embedder", embedder,
	  "-bs", "0", '-nproc', '2'],
	stderr=subprocess.PIPE, stdout=subprocess.PIPE)
	# chech process error code
	assert proc.returncode == 0, proc.stderr
	assert proc.stderr, proc.stderr
	# check process output file/dir
	assert os.path.isfile(EMBEDDING_OUTPUT), f'missing embedding output file, {EMBEDDING_OUTPUT} {proc.stderr}'
	# check output consistency
	embout = th.load(EMBEDDING_OUTPUT)
	assert len(embout) == embdata.shape[0], proc.stderr


@pytest.mark.parametrize('checkpoint_file', ['test_data/seq.emb_emb_checkpoint_mp_0.json'])
def test_parallelism_checkpoint(checkpoint_file):
	checkpoint_file = os.path.join(DIR, checkpoint_file)
	assert th.cuda.device_count() > 1, 'no enough cuda devices'
	fname = os.path.basename(checkpoint_file)
	new_location = os.path.join(EMBEDDING_OUTPUT_DIR, fname)
	with open(checkpoint_file, 'rt') as fp:
		checkpoint_data = json.load(fp)
	output = checkpoint_data['output']
	proc = subprocess.run(["python", "embeddings.py", "resume", output],
	stderr=subprocess.PIPE, stdout=subprocess.PIPE)
	# chech process error code
	assert proc.returncode == 0, proc.stderr
	assert proc.stderr, proc.stderr
	# check process output file/dir
	assert os.path.isfile(EMBEDDING_OUTPUT), f'missing embedding output file, {EMBEDDING_OUTPUT} {proc.stderr}'
	# check output consistency
	embs = HDF5Handle(EMBEDDING_OUTPUT).read_batch(0, None)
	assert len(embs) > 0