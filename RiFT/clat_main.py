import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms

from dataloader import create_dataloader
from clat.model_zoo import build_model
from clat.trainer import CLATTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="CLAT training entrypoint")
    parser.add_argument("--model", default="ResNet18", type=str)
    parser.add_argument("--dataset", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--input_size", default=32, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--data_root", default="E:/Jzee4Study/dataset", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--patch", default=4, type=int)
    parser.add_argument("--optim", default="SGDM", type=str)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", default=0.0005, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr_scheduler", default="cosine", choices=["step", "cosine"])
    parser.add_argument("--lr_decay_gamma", default=0.1, type=float)
    parser.add_argument("--pretrain_epochs", default=50, type=int)
    parser.add_argument("--clat_epochs", default=50, type=int)
    parser.add_argument("--reselect_interval", default=10, type=int)
    parser.add_argument("--num_critical_layers", default=1, type=int)
    parser.add_argument("--lambda_feat", default=1.0, type=float)
    parser.add_argument("--attack_eps", default=8 / 255, type=float)
    parser.add_argument("--attack_alpha", default=2 / 225, type=float)
    parser.add_argument("--attack_steps", default=10, type=int)
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--selection_mode", default="topk", choices=["topk", "largest", "smallest", "random"])
    parser.add_argument("--dynamic_layers", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.random_start:
        args.random_start = True
    if not args.dynamic_layers:
        args.dynamic_layers = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if "CIFAR" in args.dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_dict = {"train": transform_train, "test": transform_test}
    trainloader, _, testloader = create_dataloader(args.dataset, args.batch_size, use_val=False, transform_dict=transform_dict, data_root=args.data_root)

    model = build_model(args.model, args.input_size, args.num_classes, args.device, patch_size=args.patch, resume=args.resume)

    trainer = CLATTrainer(args, model, trainloader, testloader)
    trainer.run()


if __name__ == "__main__":
    main()
