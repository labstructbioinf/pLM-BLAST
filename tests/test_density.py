import pytest
import torch

from alntools.density import embedding_similarity
from alntools.density import chunk_cosine_similarity

@pytest.mark.parametrize('emb1', [
    10*torch.rand((150, 512)),
    torch.rand((200, 512)),
    torch.rand((300, 512)),
    10 + torch.rand((213, 512)) # big values
    ])
@pytest.mark.parametrize('emb2', [
    torch.rand((100, 512)),
    torch.rand((300, 512)),
    10 + torch.rand((213, 512)) # big values
    ])
def test_embedding_similarity(emb1, emb2):

    density = embedding_similarity(emb1, emb2)
    density_mask = density > 1.01
    density_over_norm = density_mask.sum()
    assert density_over_norm == 0, f'calculated density exeed cosine similarity norm in {density_over_norm} elements'
    assert not torch.isnan(density).any(), f'nan values in density matrix'



@pytest.mark.parametrize('emb1', [
    10*torch.rand((150, 512)),
    torch.rand((200, 512)),
    torch.rand((300, 512)),
    10 + torch.rand((213, 512)) # big values
    ])
@pytest.mark.parametrize('emb2', [
    torch.rand((100, 512)),
    torch.rand((300, 512)),
    10 + torch.rand((213, 512)) # big values
    ])
@pytest.mark.parametrize('kernelsize', [10, 20, 50])
def test_chunk_cosine_similarity(emb1, emb2, kernelsize):
    datasetfiles = ['filestr']
    quantile = 0.95
    chunk_cosine_similarity(emb1, [emb2], quantile=quantile, dataset_files=datasetfiles, kernel_size=kernelsize)