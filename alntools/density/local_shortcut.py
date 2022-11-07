import torch

@torch.jit.script
def embedding_similarity(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    '''
    compute X, Y similarity by matrix multiplication
    result shape [num X residues, num Y residues]
    Args:
        X, Y - (torch.Tensor) protein embeddings as 2D tensors [num residues, embedding size]
    Returns:
        density (torch.Tensor) 
    '''

    assert X.ndim == 2 and Y.ndim == 2, 'input tensors must have 2 dims [num residues, embedding dim]'
    assert X.shape[1] == Y.shape[1], f'embedding size is different for X, Y - {X.shape[1]} and {Y.shape[1]}'
    #normalize
    emb1_normed = X / X.pow(2).sum(1, keepdim=True).sqrt()
    emb2_normed = Y / Y.pow(2).sum(1, keepdim=True).sqrt()
    
    density = torch.matmul(emb1_normed, emb2_normed.T).T

    return density