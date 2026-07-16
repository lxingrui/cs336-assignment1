import torch


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device, dtype) -> None:
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = torch.nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        temp1 = x @ self.w1.T
        SiLu = temp1 * torch.sigmoid(temp1)
        temp2 = x @ self.w3.T
        return SiLu * temp2 @ self.w2.T
