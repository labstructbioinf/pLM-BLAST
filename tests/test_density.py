'''test torch operations'''
import pytest
import torch
import numpy as np

from alntools.numeric import embedding_local_similarity
from alntools import gather_all_paths
from alntools.density.local import chunk_score
from alntools.density import chunk_cosine_similarity
from alntools.density.local import chunk_score_batch
from alntools.density.local import chunk_cosine_similarity

PATH_SYMMETRIC_TEST = 'tests/test_data/asymetric.pt'
ATOL=1e-6
EMB_DIM = 1024
embs = torch.load(PATH_SYMMETRIC_TEST)
embsym1, embsym2 = embs[0].numpy(), embs[1].numpy()


@pytest.mark.parametrize('emb1', [
	embsym1,
	10*np.random.rand(150, EMB_DIM),
	np.random.rand(200, EMB_DIM),
	np.random.rand(300, EMB_DIM),
	10 + np.random.rand(213, EMB_DIM) # big values
	])
@pytest.mark.parametrize('emb2', [
	embsym2,
	np.random.rand(110, EMB_DIM),
	np.random.rand(300, EMB_DIM),
	10 + np.random.rand(213, EMB_DIM) # big values
	])
def test_embedding_similarity(emb1, emb2):

	emb1 = emb1.astype(np.float32)
	emb2 = emb2.astype(np.float32)
	density = embedding_local_similarity(emb1, emb2)
	density_T = embedding_local_similarity(emb2, emb1)
	density_mask = density > 1.01
	density_over_norm = density_mask.sum()
	# values are greater then one
	assert density_over_norm == 0, f'calculated density exeed cosine similarity norm in {density_over_norm} elements'
	# nan values
	assert not np.isnan(density).any(), f'nan values in density matrix'
	# asymmetric results
	if not np.allclose(density, density_T.T, atol=ATOL):
		max_diff = density - density_T.T
		raise ValueError(f'density matrix is asymmetrix a(X,Y) != a(Y,X).T max diff: {max_diff.max()} for {emb1.shape} - {emb2.shape}')



@pytest.mark.parametrize('emb1', [
	embsym1,
	10*np.random.rand(150, EMB_DIM),
	np.random.rand(200, EMB_DIM),
	np.random.rand(300, EMB_DIM),
	10 + np.random.rand(213, EMB_DIM) # big values
	])
@pytest.mark.parametrize('emb2', [
	embsym2,
	np.random.rand(110, EMB_DIM),
	np.random.rand(300, EMB_DIM),
	10 + np.random.rand(213, EMB_DIM) # big values
	])
@pytest.mark.parametrize('norm', [False, True])
def test_path_gathering(emb1, emb2, norm):

	emb1 = emb1.astype(np.float32)
	emb2 = emb2.astype(np.float32)
	density = embedding_local_similarity(emb1, emb2)
	density_cp = density.copy()
	density_T = embedding_local_similarity(emb2, emb1)
	paths = gather_all_paths(density, norm=norm,
								 minlen=10,
								 bfactor=1,
								 gap_opening=0,
								 gap_extension=0)
	# check if path grathering overwrites density values
	if not np.allclose(density, density_cp):
		max_diff = density - density_cp
		raise ValueError(f'density values where illegaly overwriten, max diff: {max_diff.max()}')
	density_mask = density > 1.01
	density_over_norm = density_mask.sum()
	# values are greater then one
	assert density_over_norm == 0, f'calculated density exeed cosine similarity norm in {density_over_norm} elements'
	# nan values
	assert not np.isnan(density).any(), f'nan values in density matrix'
	# asymmetric results
	if not np.allclose(density, density_T.T, atol=ATOL):
		max_diff = density - density_T.T
		raise ValueError(f'density matrix is asymmetrix a(X,Y) != a(Y,X).T max diff: {max_diff.max()} for {emb1.shape} - {emb2.shape}')


@pytest.mark.parametrize('emb1', [20, 50, 500])
@pytest.mark.parametrize('emb2', [20, 50, 500])
@pytest.mark.parametrize('kernelsize', [10, 30])
@pytest.mark.parametrize('stride', [5, 10])
def test_chunk_cosine_similarity(emb1, emb2, kernelsize, stride):
	query = torch.rand((emb1, 64))
	targets = [torch.rand((emb2, 64)) for _ in range(37)] 
	datasetfiles = [f'filestr_{n}' for n in range(37)]
	results = chunk_cosine_similarity(query, targets, quantile=0.0, dataset_files=datasetfiles, stride=stride, kernel_size=kernelsize)
	assert isinstance(results, list)
	assert isinstance(results[0], dict)
	assert len(results[0]) == len(targets), f"{len(results[0])} != {len(targets)}"
	

@pytest.mark.parametrize('kernelsize', [10, 20, 50])
@pytest.mark.parametrize('stride', [1, 5, 10])
def test_chunk_cosine_similarity_batch(kernelsize, stride):
	embdim = 64
	queries = [torch.rand(embsize, embdim) for embsize in torch.randint(50, 500, size=(10, ))]
	targets = [torch.rand(embsize, embdim) for embsize in torch.randint(50, 500, size=(100, ))]
	results = chunk_score_batch(queries, targets, stride=stride, kernel_size=kernelsize)
	assert results.shape[0] == len(targets)
	assert results.shape[1] == len(queries)


def test_chunk_result_equality():
	embdim = 64
	queries = [torch.rand(embsize, embdim) for embsize in torch.randint(50, 500, size=(10, ))]
	targets = [torch.rand(embsize, embdim) for embsize in torch.randint(50, 500, size=(100, ))]
	results = chunk_score_batch(queries, targets, stride=10, kernel_size=20)
	for i, q in enumerate(queries):
		result_single = chunk_score(q, targets, stride=10, kernel_size=20).view(-1)
		result_flat = results[:, i].view(-1)
		assert result_single.shape[0] == result_flat.shape[0]
		if not torch.allclose(result_flat, result_single, atol=1e-3):
			result_diff = (result_flat - result_single).abs()
			raise ValueError(f"results are differnt by avg {result_diff.mean()} and max {result_diff.max()}")