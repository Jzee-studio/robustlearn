# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standard supervised training entrypoint.

This script is intended to provide a clean from-scratch training path that
reuses the project's existing model and dataloader utilities while avoiding
RiFT/fine-tuning-specific behavior.
"""

from copy import deepcopy
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from dataloader import create_dataloader
from model import create_model
from optimizer import create_optimizer, create_scheduler
from utils import create_logger, evaluate_cifar_robustness, evaluate_tiny_robustness


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform_dict(dataset: str):
    from torchvision import transforms

    if "CIFAR" in dataset:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return {"train": transform_train, "test": transform_test}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_num += batch_size
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()

    return total_loss / max(total_num, 1), 100.0 * total_correct / max(total_num, 1)


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    use_amp = scaler is not None
    pbar = tqdm(dataloader, desc="train", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_num += batch_size
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()

        pbar.set_postfix(loss=total_loss / max(total_num, 1), acc=100.0 * total_correct / max(total_num, 1))

    return total_loss / max(total_num, 1), 100.0 * total_correct / max(total_num, 1)


def load_checkpoint_if_needed(model, optimizer, scheduler, path, device):
    if path is None:
        return 0, 0.0

    checkpoint = torch.load(path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "net" in checkpoint:
        model.load_state_dict(checkpoint["net"])
    else:
        model.load_state_dict(checkpoint)

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_acc = float(checkpoint.get("best_acc", 0.0))

    if "optimizer" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return start_epoch, best_acc


def main():
    parser = argparse.ArgumentParser(description="Standard supervised training")
    parser.add_argument("--model", default="ResNet18", type=str, help="model name")
    parser.add_argument("--dataset", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "TinyImageNet"], help="dataset")
    parser.add_argument("--num_classes", default=10, type=int, help="number of classes")
    parser.add_argument("--input_size", default=32, type=int, help="input image size")
    parser.add_argument("--patch", default=4, type=int, help="patch size for ViT")

    parser.add_argument("--data_root", default="E:/Jzee4Study/dataset", type=str, help="dataset root")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--workers", default=8, type=int, help="dataloader workers")

    parser.add_argument("--epochs", default=200, type=int, help="training epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--optim", default="SGDM", type=str, choices=["SGDM", "Adam", "RMSprop", "AdamW"], help="optimizer")
    parser.add_argument("--lr_scheduler", default="step", choices=["step", "cosine"], help="lr scheduler")
    parser.add_argument("--lr_decay_gamma", default=0.1, type=float, help="step decay gamma")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="warmup epochs (manual linear warmup)")

    parser.add_argument("--resume", default=None, type=str, help="resume checkpoint")
    parser.add_argument("--pretrained", default=None, type=str, help="load model weights only")
    parser.add_argument("--save_dir", default=None, type=str, help="checkpoint output directory")
    parser.add_argument("--use_val", action="store_true", help="use validation split if available")
    parser.add_argument("--amp", action="store_true", help="use automatic mixed precision")
    parser.add_argument("--benchmark", action="store_true", help="enable cudnn benchmark")

    args = parser.parse_args()

    set_seed(args.seed)
    cudnn.deterministic = False
    cudnn.benchmark = bool(args.benchmark)

    device = get_device(args.device)
    args.device = str(device)

    if args.num_classes is None:
        args.num_classes = 10 if args.dataset == "CIFAR10" else 100 if args.dataset == "CIFAR100" else 200

    if args.save_dir is None:
        proj_name = "std"
        args.save_dir = os.path.join(
            "./results",
            f"{args.model}_{args.dataset}",
            "checkpoint",
            f"{proj_name}_{args.optim}_lr={args.lr}_wd={args.wd}_epochs={args.epochs}",
        )
    os.makedirs(args.save_dir, exist_ok=True)

    logger = create_logger(os.path.join(args.save_dir, "output.log"))
    logger.info(args)

    transform_dict = build_transform_dict(args.dataset)
    trainloader, valloader, testloader = create_dataloader(
        args.dataset,
        args.batch_size,
        use_val=args.use_val,
        transform_dict=transform_dict,
        data_root=args.data_root,
    )

    model = create_model(args.model, args.input_size, args.num_classes, args.device, args.patch, resume=None)
    model = model.to(device)
    if isinstance(model, nn.DataParallel):
        base_model = model
    else:
        base_model = model

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args.optim, base_model, args.lr, args.momentum, weight_decay=args.wd)

    if args.lr_scheduler == "step":
        lr_decays = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        scheduler = create_scheduler(args, optimizer, lr_decays=lr_decays)
    else:
        scheduler = create_scheduler(args, optimizer)

    start_epoch = 0
    best_acc = 0.0

    if args.pretrained is not None:
        ckpt = torch.load(args.pretrained, map_location=device)
        if "model" in ckpt:
            base_model.load_state_dict(ckpt["model"])
        elif "state_dict" in ckpt:
            base_model.load_state_dict(ckpt["state_dict"])
        elif "net" in ckpt:
            base_model.load_state_dict(ckpt["net"])
        else:
            base_model.load_state_dict(ckpt)
        logger.info(f"Loaded pretrained weights from {args.pretrained}")

    if args.resume is not None:
        start_epoch, best_acc = load_checkpoint_if_needed(base_model, optimizer, scheduler, args.resume, device)
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}, best_acc={best_acc:.2f}")

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    test_robust_acc_fn = evaluate_cifar_robustness if "CIFAR" in args.dataset else evaluate_tiny_robustness

    init_state = deepcopy(base_model.state_dict())
    torch.save(init_state, os.path.join(args.save_dir, "init_params.pth"))

    logger.info("==> Initial evaluation...")
    if valloader is not None:
        init_loss, init_acc = evaluate(base_model, valloader, criterion, device)
        logger.info(f"Init val loss: {init_loss:.4f}, val acc: {init_acc:.2f}%")
    else:
        init_loss, init_acc = evaluate(base_model, testloader, criterion, device)
        logger.info(f"Init test loss: {init_loss:.4f}, test acc: {init_acc:.2f}%")

    init_robust_acc = test_robust_acc_fn(args, base_model)
    logger.info(f"Init robust acc: {init_robust_acc:.2f}%")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"==> Epoch {epoch + 1}/{args.epochs}")

        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            warmup_scale = float(epoch + 1) / float(max(args.warmup_epochs, 1))
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * warmup_scale

        train_loss, train_acc = train_one_epoch(base_model, trainloader, optimizer, criterion, device, scaler=scaler)

        if args.warmup_epochs > 0 and epoch + 1 >= args.warmup_epochs:
            scheduler.step()
        elif args.warmup_epochs == 0:
            scheduler.step()

        if valloader is not None:
            val_loss, val_acc = evaluate(base_model, valloader, criterion, device)
            monitor_loss, monitor_acc = val_loss, val_acc
            logger.info(f"Train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%")
            logger.info(f"Val loss: {val_loss:.4f}, val acc: {val_acc:.2f}%")
        else:
            test_loss, test_acc = evaluate(base_model, testloader, criterion, device)
            monitor_loss, monitor_acc = test_loss, test_acc
            logger.info(f"Train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%")
            logger.info(f"Test loss: {test_loss:.4f}, test acc: {test_acc:.2f}%")

        ckpt = {
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pth"))

        if monitor_acc > best_acc:
            best_acc = monitor_acc
            ckpt["best_acc"] = best_acc
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))
            logger.info(f"Saved new best checkpoint with acc={best_acc:.2f}%")

    best_path = os.path.join(args.save_dir, "best.pth")
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device)
        if "model" in best_ckpt:
            base_model.load_state_dict(best_ckpt["model"])
        else:
            base_model.load_state_dict(best_ckpt)

    if valloader is not None:
        final_loss, final_acc = evaluate(base_model, valloader, criterion, device)
        logger.info(f"Final val loss: {final_loss:.4f}, val acc: {final_acc:.2f}%")
    else:
        final_loss, final_acc = evaluate(base_model, testloader, criterion, device)
        logger.info(f"Final test loss: {final_loss:.4f}, test acc: {final_acc:.2f}%")

    final_robust_acc = test_robust_acc_fn(args, base_model)
    logger.info(f"Final robust acc: {final_robust_acc:.2f}%")


if __name__ == "__main__":
    main()
