from dataclasses import dataclass


@dataclass
class CLATConfig:
    model: str = "ResNet18"
    dataset: str = "CIFAR10"
    resume: str | None = None
    device: str = "cuda"
    data_root: str = "E:/Jzee4Study/dataset"
    batch_size: int = 128
    seed: int = 0
    lr: float = 0.1
    momentum: float = 0.9
    wd: float = 5e-4
    epochs: int = 100
    pretrain_epochs: int = 50
    clat_epochs: int = 50
    reselect_interval: int = 10
    num_critical_layers: int = 1
    lambda_feat: float = 1.0
    attack_eps: float = 8 / 255
    attack_alpha: float = 2 / 225
    attack_steps: int = 10
    random_start: bool = True
    selection_mode: str = "topk"
    dynamic_layers: bool = True
    lr_scheduler: str = "cosine"
