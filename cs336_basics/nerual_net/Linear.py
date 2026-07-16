from math import sqrt

import einops
import torch


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        empty_tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.W = torch.nn.Parameter(empty_tensor)
        sigma = sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.W, "... d_in,d_out d_in -> ... d_out")
