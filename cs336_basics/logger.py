import csv
import json
import os
import time


class ExperimentTracker:
    def __init__(
        self,
        run_name: str,
        hparams: dict,
        log_dir: str = "logs",
        use_wandb: bool = True,
        wandb_project: str = "cs336-basics",
    ):
        self.run_name = run_name
        self.hparams = hparams
        self.log_dir = log_dir
        self.use_wandb = use_wandb

        # 创建日志文件夹
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, f"{run_name}.csv")
        self.config_path = os.path.join(self.log_dir, f"{run_name}_config.json")

        # 1. 保存当前实验的超参数配置
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(hparams, f, indent=4)

        # 2. 初始化本地 CSV 记录文件
        self.csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "wall_clock_time", "train_loss", "val_loss", "lr"])

        # 3. 初始化 WandB（如果可用）
        if self.use_wandb:
            try:
                import wandb

                # 初始化 WandB 项目
                wandb.init(project=wandb_project, name=run_name, config=hparams)
            except ImportError:
                print("WandB is not installed. Falling back to local CSV logging only.")
                self.use_wandb = False

        # 记录训练起始的绝对时间
        self.start_time = time.time()

    def log(self, step: int, train_loss: float, val_loss: float, lr: float):
        """
        记录核心指标：包含当前的梯度步数、从开始训练起流逝的实际秒数（Wall-clock time）
        """
        # 计算相对流逝时间（秒）
        elapsed_time = time.time() - self.start_time

        # 写入本地 CSV
        self.csv_writer.writerow(
            [
                step,
                f"{elapsed_time:.2f}",
                f"{train_loss:.6f}" if train_loss is not None else "",
                f"{val_loss:.6f}" if val_loss is not None else "",
                f"{lr:.8f}" if lr is not None else "",
            ]
        )
        self.csv_file.flush()  # 强制写入磁盘，防止程序意外中断丢失日志

        # 写入 WandB
        if self.use_wandb:
            import wandb

            metrics = {"step": step, "wall_clock_time": elapsed_time}
            if train_loss is not None:
                metrics["train_loss"] = train_loss
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            if lr is not None:
                metrics["lr"] = lr
            wandb.log(metrics, step=step)

    def close(self):
        self.csv_file.close()
        if self.use_wandb:
            import wandb

            wandb.finish()
