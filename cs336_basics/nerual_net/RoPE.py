import torch


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        idx = torch.arange(0, d_k, 2, device=device)
        BaseFrequencies = theta ** -(idx / d_k)
        pos_idx = torch.arange(0, max_seq_len, device=device)
        angles = torch.outer(pos_idx, BaseFrequencies)
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = self.cos_cached[token_positions]  # type: ignore
        sin = self.sin_cached[token_positions]  # type: ignore
        x_even_rot = x_even * cos - x_odd * sin  # 算出 [x_0', x_2']
        x_odd_rot = x_even * sin + x_odd * cos  # 算出 [x_1', x_3']
        # 2. 创建一个形状、类型、设备完全相同的空张量
        output = torch.empty_like(x)

        # 3. 利用切片把数据“填”进对应的格子里
        output[..., 0::2] = x_even_rot  # 把 [x_0', x_2'] 填入 0, 2 位置
        output[..., 1::2] = x_odd_rot  # 把 [x_1', x_3'] 填入 1, 3 位置
        return output

    # 在这里实现：
    # 1. 用 token_positions 索引 self.cos_cached 和 self.sin_cached
    # 2. 提取 x 的偶数分量和奇数分量
    # 3. 计算旋转后的偶数分量和奇数分量
    # 4. 将它们交替合并成一个和 x 形状完全相同的新张量并返回
