import torch


# 假设我们在 3D 标注中定义这个类型提示（这里用标准 Tensor 演示）
def run_softmax_with_trace(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    print("=" * 60)
    print(f"输入矩阵 shape: {list(in_features.shape)}")
    print(f"输入矩阵值:\n{in_features}\n")

    # 1. 寻找最大值
    x_max_tuple = torch.max(in_features, dim=dim, keepdim=True)
    x_max = x_max_tuple[0]
    print(f"--- 步骤 1: 寻找最大值 (dim={dim}, keepdim=True) ---")
    print(f"最大值矩阵 shape: {list(x_max.shape)}")
    print(f"最大值矩阵值:\n{x_max}\n")

    # 2. 减去最大值
    x_max_reduced = in_features - x_max
    print("--- 步骤 2: 减去最大值 (防止溢出) ---")
    print(f"减去最大值后 shape: {list(x_max_reduced.shape)}")
    print(f"减去最大值后值:\n{x_max_reduced}\n")

    # 3. 计算指数
    x_max_reduced_exp = torch.exp(x_max_reduced)
    print("--- 步骤 3: 计算指数 (exp) ---")
    print(f"指数后 shape: {list(x_max_reduced_exp.shape)}")
    print(f"指数后值:\n{x_max_reduced_exp}\n")

    # 4. 计算指数和
    x_sum = x_max_reduced_exp.sum(dim=dim, keepdim=True)
    print(f"--- 步骤 4: 计算指数和 (dim={dim}, keepdim=True) ---")
    print(f"指数和 shape: {list(x_sum.shape)}")
    print(f"指数和值:\n{x_sum}\n")

    # 5. 归一化相除
    out = x_max_reduced_exp / x_sum
    print("--- 步骤 5: 最终归一化输出 (Softmax 结果) ---")
    print(f"输出 shape: {list(out.shape)}")
    print(f"输出值:\n{out}\n")
    print("=" * 60)

    return out


if __name__ == "__main__":
    # 构造一个形状为 [2, 3] 的测试张量
    # 第一行包含一个较大值 10.0，第二行是全同元素
    test_tensor = torch.tensor([[1.0, 2.0, 10.0], [5.0, 5.0, 5.0]], dtype=torch.float32)

    # 在 dim=1 维度上运行带有追踪的 Softmax
    run_softmax_with_trace(test_tensor, dim=1)
