import torch

# 调整 PyTorch 的打印设置，让浮点数显示更整洁
torch.set_printoptions(precision=4, sci_mode=False)


def run_cross_entropy_verbose(inputs: torch.Tensor, targets: torch.Tensor):
    print("=" * 60)
    print("【初始输入数据】")
    print(f"1. 随机初始化的未归一化得分 (Logits) [形状: {list(inputs.shape)}]:\n{inputs}")
    print("   (每一行代表一个样本对词表中 4 个词的预测得分)")
    print(f"2. 随机生成的真实标签 (Targets) [形状: {list(targets.shape)}]:\n{targets}")
    print(
        f"   (表示第 0 个样本的正确答案是第 {targets[0].item()} 个词，第 1 个样本的正确答案是第 {targets[1].item()} 个词)"
    )
    print("=" * 60 + "\n")

    # =========================================================================
    # Step 1: 数值稳定处理 (防止 exp 溢出)
    # =========================================================================
    x_max = torch.max(inputs, dim=1, keepdim=True)[0]
    print("【Step 1: 数值稳定处理】")
    print(f"-> 每行的最大值 x_max:\n{x_max}")

    x_stable = inputs - x_max
    print(f"-> 减去最大值后的 x_stable (最大值所在位置会变成 0.0000):\n{x_stable}")
    print("   *原理说明：减去最大值后，所有数都变成 <= 0，exp(x_stable) 结果在 (0, 1] 之间，防止 NaN 溢出。*")
    print("-" * 50 + "\n")

    # =========================================================================
    # Step 2: 计算分母的 log-sum-exp
    # =========================================================================
    print("【Step 2: 计算分母的 log-sum-exp】")
    exp_x = torch.exp(x_stable)
    print(f"-> 1. 先计算自然指数 exp(x_stable):\n{exp_x}")

    sum_exp_x = exp_x.sum(dim=1, keepdim=True)
    print(f"-> 2. 按行求和 sum(exp(x_stable)) (Softmax 的分母):\n{sum_exp_x}")

    log_sum_exp = torch.log(sum_exp_x)
    print(f"-> 3. 对分母取自然对数 log_sum_exp:\n{log_sum_exp}")
    print("-" * 50 + "\n")

    # =========================================================================
    # Step 3: 计算 log_softmax
    # =========================================================================
    print("【Step 3: 计算全量对数概率 log_softmax】")
    log_softmax = x_stable - log_sum_exp
    print(f"-> 减法得到的对数概率矩阵 (所有值都是负数，越接近 0 概率越大):\n{log_softmax}")
    print("-" * 50 + "\n")

    # =========================================================================
    # Step 4: 根据 targets 提取正确类别的 log 概率
    # =========================================================================
    print("【Step 4: 精准捞出正确答案的概率】")
    batch_size = inputs.shape[0]
    rows = torch.arange(batch_size)
    print(f"-> 行索引 (rows): {rows}")
    print(f"-> 列索引 (targets): {targets}")

    # 高级索引操作：直接把正确答案位置的值“抠”出来
    target_log_probs = log_softmax[rows, targets]
    print(f"-> 提取出的正确词的 log 概率 target_log_probs [形状: {list(target_log_probs.shape)}]:\n{target_log_probs}")
    print("   *具体抓取过程：")
    print(f"    - 从第 0 行抓取第 {targets[0].item()} 列的值: {log_softmax[0, targets[0]].item():.4f}")
    print(f"    - 从第 1 行抓取第 {targets[1].item()} 列的值: {log_softmax[1, targets[1]].item():.4f}")
    print("-" * 50 + "\n")

    # =========================================================================
    # Step 5: 计算负对数似然损失并在 batch 上求平均
    # =========================================================================
    print("【Step 5: 计算最终的 Loss 标量】")
    print(f"-> 1. 给概率值加上负号，转为正数的损失:\n{-target_log_probs}")

    loss = -target_log_probs.mean()
    print(f"-> 2. 对整个 Batch 求平均得到的最终 Loss: {loss.item():.4f}")
    print("=" * 60)

    return loss


# =========================================================================
# 随机数据运行测试
# =========================================================================
if __name__ == "__main__":
    # 固定随机种子，确保你每次运行看到的结果一致（也可以注释掉看不同随机数的结果）
    # torch.manual_seed(42)

    # 1. 初始化输入：生成 batch_size=2, vocab_size=4 的随机浮点数矩阵
    random_logits = torch.randn(2, 4, dtype=torch.float32)

    # 2. 初始化标签：在 0~3 之间随机为每个样本选一个整数作为正确标签
    random_targets = torch.randint(0, 4, (2,), dtype=torch.long)

    # 运行带打印版本的交叉熵函数
    loss_val = run_cross_entropy_verbose(random_logits, random_targets)
