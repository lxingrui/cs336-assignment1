import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.W = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        # 1. 将转换后的 float32 新张量赋值给一个新变量
        x_fp32 = x.to(torch.float32)

        # 2. 后续所有的计算都使用这个 float32 版本的张量
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(var + self.eps)

        # 3. 此时 rms 本身就是 float32 的，W 也可以统一转为 float32
        result = x_fp32 * rms * self.W.to(torch.float32)

        # 4. 最后统一转回原始类型
        return result.to(in_type)
