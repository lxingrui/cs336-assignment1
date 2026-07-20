from collections.abc import Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        # 1. 规范的参数校验（防止用户传入非法数值导致数学计算崩溃）
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        # 2. 将参数解构放入 defaults，保持 beta1 和 beta2 清晰
        defaults = {"lr": lr, "weight_decay": weight_decay, "beta1": betas[0], "beta2": betas[1], "eps": eps}
        super().__init__(params, defaults)

    # 3. 使用装饰器，确保整个更新过程不记录梯度，保障内存安全
    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 4. 显式且优雅地初始化状态张量
                # 这能保证 m1, m2 的设备（CPU/GPU）、dtype（半精度/单精度）与参数 p 完美对齐
                if len(state) == 0:
                    state["t"] = 0
                    state["m1"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["m2"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # 状态累加
                state["t"] += 1
                t = state["t"]
                m1, m2 = state["m1"], state["m2"]

                # ================= 数学更新步骤 =================

                # 步骤一：解耦权重衰减 (Decoupled Weight Decay)
                # 使用就地乘法（mul_）直接修改 p 的值，省去了内存分配
                p.mul_(1 - lr * weight_decay)

                # 步骤二：更新一阶动量 m1
                # 计算公式：m1 = beta1 * m1 + (1 - beta1) * grad
                # add_ 的参数 alpha 会将后面的 grad 乘以 (1 - beta1) 后再加给 m1
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 步骤三：更新二阶动量 m2
                # 计算公式：m2 = beta2 * m2 + (1 - beta2) * grad^2
                # addcmul_ 专门用于计算两个张量相乘后累加：m2 = m2 + value * (grad * grad)
                m2.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 步骤四：计算偏差修正系数
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t

                # 步骤五：计算修正后的一阶/二阶矩
                m1_bias_corrected = m1 / bias_correction1
                m2_bias_corrected = m2 / bias_correction2

                # 步骤六：参数最终更新
                # 计算公式：denom = sqrt(m2_bias) + eps
                # 然后：p = p - lr * (m1_bias / denom)
                denom = m2_bias_corrected.sqrt().add_(eps)

                # addcdiv_：就地执行 p = p + value * (tensor1 / tensor2)
                p.addcdiv_(m1_bias_corrected, denom, value=-lr)

        return loss
