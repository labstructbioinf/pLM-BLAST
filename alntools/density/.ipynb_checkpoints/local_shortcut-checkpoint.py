import torch

#@torch.jit.script
def embedding_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    
    #normalize
    emb1_norm = emb1.pow(2).sum(1, keepdim=True).sqrt()
    emb2_norm = emb2.pow(2).sum(1, keepdim=True).sqrt()
    
    emb1 /= emb1_norm
    emb2 /= emb2_norm
    
    density = torch.matmul(emb1, emb2.T)
    return density