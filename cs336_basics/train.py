# train.py
import argparse
import torch
import numpy as np

# 1. 自动选择最佳设备字符串
if torch.cuda.is_available():
    device_str = "cuda"
elif torch.backends.mps.is_available():
    device_str = "mps"
else:
    device_str = "cpu"
    # 2. 设置全局默认设备
torch.set_default_device(device_str)

print(f"全局默认设备已被设置为: {torch.get_default_device()}")


def main():
    # 1. 创建一个参数解析器对象
    parser = argparse.ArgumentParser(description="CS336 Transformer 训练脚本")

    # 2. 告诉 Python 你希望在终端里接收哪些参数
    # -- 符号代表这是一个可选/指定名称的参数。
    # type 指定了数据类型（argparse 会自动帮你把终端里的字符串转成对应的 int, float 或 str）
    # default 设置了如果用户没输入该参数时的默认值
    # required=True 代表这个参数是必须传入的，不传程序会直接报错并提示用户

    parser.add_argument("--train_data_path", type=str, required=True, help="训练集二进制文件 (.bin) 的路径")
    parser.add_argument("--val_data_path", type=str, required=True, help="验证集二进制文件 (.bin) 的路径")

    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    parser.add_argument("--context_length", type=int, default=256, help="模型的上下文最大长度")
    parser.add_argument("--batch_size", type=int, default=64, help="训练批次大小 (Batch Size)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--total_steps", type=int, default=20000, help="总共训练的梯度步数")

    parser.add_argument("--eval_interval", type=int, default=500, help="每隔多少步进行一次验证集评估")
    parser.add_argument("--log_interval", type=int, default=100, help="每隔多少步记录一次训练日志")

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/baseline", help="保存断点的文件夹路径")
    parser.add_argument("--run_name", type=str, default="tinystories_baseline", help="当前实验运行的名称")

    # 3. 这一步会让 Python 去读取终端传入的所有参数，并将其封装在一个对象里
    args = parser.parse_args()

    # 4. 在代码中，你可以通过 args.参数名 的方式直接访问这些值
    print("=" * 40)
    print(f"成功启动实验: {args.run_name}")
    print(f"训练数据路径: {args.train_data_path}")
    print(f"学习率设为: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 40)

    # 5. 接下来，你就可以把这些参数传递给你自己的函数/类了
    # 例如：
    # model = MyTransformer(
    #     vocab_size=args.vocab_size,
    #     context_length=args.context_length
    # )
    #
    # train(model, args.train_data_path, args.batch_size, args.learning_rate)


if __name__ == "__main__":
    main()
