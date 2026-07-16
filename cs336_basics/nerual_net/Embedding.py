import torch


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        super().__init__()
        embedding_m = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.W = torch.nn.Parameter(embedding_m)
        torch.nn.init.trunc_normal_(self.W, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor):
        return self.W[token_ids]
