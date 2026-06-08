# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm

from attack import pgd_attack, trades_loss
from dataloader import create_dataloader, TinyImageNet
from model import create_model
from optimizer import create_optimizer, create_scheduler
from utils import (
    Normalize,
    create_logger,
    evaluate,
    evaluate_cifar_robustness,
    evaluate_tiny_robustness,
    format_eta,
    setup_wandb,
    wandb_log,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_dataset_stats(dataset: str):
    if dataset in ["CIFAR10", "CIFAR100"]:
        return (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616), 32
    if dataset == "TinyImageNet":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 64
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_transforms(dataset: str):
    mean, std, image_size = get_dataset_stats(dataset)
    if dataset in ["CIFAR10", "CIFAR100"]:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return {
        "train": transform_train,
        "test": transform_test,
    }, image_size


def build_dataloaders(args):
    transform_dict, image_size = build_transforms(args.dataset)
    args.input_size = image_size
    trainloader, _, testloader = create_dataloader(
        args.dataset,
        args.batch_size,
        use_val=False,
        transform_dict=transform_dict,
        data_root=args.data_root,
    )
    return trainloader, testloader


def make_model_for_eval(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def evaluate_robust(args, model):
    eval_model = make_model_for_eval(model)
    if "CIFAR" in args.dataset:
        return evaluate_cifar_robustness(args, eval_model)
    return evaluate_tiny_robustness(args, eval_model)


def save_checkpoint(path, state):
    torch.save(state, path)


def train_one_epoch(args, model, trainloader, optimizer, criterion, epoch, logger):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False, dynamic_ncols=True)
    start_time = time.time()

    for step, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        optimizer.zero_grad(set_to_none=True)

        if args.method == "pgd":
            adv_inputs = pgd_attack(
                model,
                inputs,
                targets,
                args.dataset,
                args.eps,
                args.attack_alpha,
                args.attack_steps,
                random_start=args.random_start,
            )
            logits = model(adv_inputs)
            loss = criterion(logits, targets)
            clean_logits = model(inputs)
            robust_logits = logits.detach()
            loss_natural = loss.detach()
            loss_robust = torch.tensor(0.0, device=args.device)
        elif args.method == "trades":
            loss, loss_natural, loss_robust, clean_logits, robust_logits = trades_loss(
                model,
                inputs,
                targets,
                args.dataset,
                args.eps,
                args.attack_alpha,
                args.attack_steps,
                args.beta,
                random_start=args.random_start,
            )
        else:
            raise ValueError(f"Unsupported method: {args.method}")

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = clean_logits.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total += batch_size

        avg_loss = total_loss / total
        avg_acc = 100.0 * total_correct / total
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        iters_left = len(trainloader) - (step + 1)
        eta = (elapsed / max(1, step + 1)) * iters_left

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{avg_acc:.2f}%",
            "lr": f"{lr:.6f}",
            "eta": format_eta(eta),
        })

    return total_loss / total, 100.0 * total_correct / total


def main():
    parser = argparse.ArgumentParser(description="Adversarial Training for RiFT baselines")
    parser.add_argument("--model", default="ResNet18", type=str, help="model used")
    parser.add_argument("--dataset", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "TinyImageNet"], help="dataset")
    parser.add_argument("--num_classes", default=10, type=int, help="num classes")
    parser.add_argument("--input_size", default=32, type=int, help="input_size")
    parser.add_argument("--patch", default=4, type=int, help="num patch (used by vit)")
    parser.add_argument("--resume", default=None, type=str, help="resume from checkpoint")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--data_root", default="E:/Jzee4Study/dataset", type=str, help="dataset root directory")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--optim", default="SGDM", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for SGDM")
    parser.add_argument("--lr_scheduler", default="step", choices=["step", "cosine"], help="lr scheduler")
    parser.add_argument("--lr_decay_gamma", default=0.1, type=float, help="lr_decay_gamma")
    parser.add_argument("--wd", default=0.0005, type=float, help="weight decay")
    parser.add_argument("--epochs", default=30, type=int, help="num of epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--method", default="pgd", choices=["pgd", "trades"], help="adversarial training method")
    parser.add_argument("--eps", default=8 / 255, type=float, help="linf epsilon")
    parser.add_argument("--attack_steps", default=10, type=int, help="attack steps")
    parser.add_argument("--attack_alpha", default=2 / 255, type=float, help="attack step size")
    parser.add_argument("--random_start", action="store_true", help="use random start in attack")
    parser.add_argument("--beta", default=6.0, type=float, help="TRADES beta")
    parser.add_argument("--eval_freq", default=1, type=int, help="evaluation frequency")
    parser.add_argument("--save_best_metric", default="robust", choices=["robust", "clean"], help="metric used for best checkpoint selection")
    parser.add_argument("--wandb_project", default="rift-adv-training", type=str, help="wandb project")
    parser.add_argument("--wandb_name", default=None, type=str, help="wandb run name")
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"], help="wandb mode")
    parser.add_argument("--log_dir", default=None, type=str, help="optional tensorboard log dir")

    args = parser.parse_args()
    if not args.random_start:
        args.random_start = True

    if args.dataset == "CIFAR10":
        args.num_classes = 10
    elif args.dataset == "CIFAR100":
        args.num_classes = 100
    elif args.dataset == "TinyImageNet":
        args.num_classes = 200

    set_seed(args.seed)

    proj_name = "advtrain"
    suffix = f"{proj_name}_{args.method}_{args.optim}_lr={args.lr}_wd={args.wd}_epochs={args.epochs}_{args.model}"
    model_save_dir = os.path.join("./results", f"{args.model}_{args.dataset}", "adv_checkpoint", suffix)
    os.makedirs(model_save_dir, exist_ok=True)

    logger = create_logger(os.path.join(model_save_dir, "output.log"))
    logger.info(args)

    logger.info("==> Preparing data and creating dataloaders...")
    trainloader, testloader = build_dataloaders(args)

    logger.info("==> Building model...")
    model = create_model(args.model, args.input_size, args.num_classes, args.device, args.patch, args.resume)

    logger.info("==> Building optimizer and learning rate scheduler...")
    optimizer = create_optimizer(args.optim, model, args.lr, args.momentum, weight_decay=args.wd)
    lr_decays = [int(args.epochs // 2)]
    scheduler = create_scheduler(args, optimizer, lr_decays=lr_decays)

    criterion = nn.CrossEntropyLoss()
    init_sd = deepcopy(make_model_for_eval(model).state_dict())

    if "CIFAR" in args.dataset:
        eval_robustness = evaluate_cifar_robustness
    else:
        eval_robustness = evaluate_tiny_robustness

    wandb_run = setup_wandb(args)
    wandb.watch(model, criterion, log="gradients", log_freq=100)
    wandb_log({"config/num_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}, step=0)

    best_metric = -1.0
    best_clean = -1.0
    best_robust = -1.0
    best_epoch = -1

    if args.resume is not None:
        logger.info(f"==> Resume checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "net" in checkpoint:
                model.load_state_dict(checkpoint["net"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])

    _, init_train_acc = evaluate(args, model, trainloader, criterion)
    init_test_loss, init_test_acc = evaluate(args, model, testloader, criterion)
    init_robust_acc = eval_robustness(args, make_model_for_eval(model))
    logger.info("==> Init train acc: {:.2f}%, test acc: {:.2f}%, robust acc: {:.2f}%".format(init_train_acc, init_test_acc, init_robust_acc))

    wandb_log({
        "init/train_acc": init_train_acc,
        "init/test_acc": init_test_acc,
        "init/robust_acc": init_robust_acc,
        "init/test_loss": init_test_loss,
    }, step=0)

    for epoch in range(args.epochs):
        logger.info(f"==> Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(args, model, trainloader, optimizer, criterion, epoch, logger)
        logger.info("==> Train loss: {:.4f}, train acc: {:.2f}%".format(train_loss, train_acc))

        test_loss = None
        test_acc = None
        robust_acc = None
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            logger.info("==> Testing clean...")
            test_loss, test_acc = evaluate(args, model, testloader, criterion)
            logger.info("==> Test loss: {:.4f}, test acc: {:.2f}%".format(test_loss, test_acc))

            logger.info("==> Testing robust...")
            robust_acc = eval_robustness(args, make_model_for_eval(model))
            logger.info("==> Robust acc: {:.2f}%".format(robust_acc))

            metric = robust_acc if args.save_best_metric == "robust" else test_acc
            if metric > best_metric:
                best_metric = metric
                best_clean = test_acc
                best_robust = robust_acc
                best_epoch = epoch + 1
                logger.info("==> Saving best params...")
                save_checkpoint(
                    os.path.join(model_save_dir, "best_params.pth"),
                    {
                        "model": make_model_for_eval(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "best_clean_acc": best_clean,
                        "best_robust_acc": best_robust,
                        "args": vars(args),
                        "init_model": init_sd,
                    },
                )

            save_checkpoint(
                os.path.join(model_save_dir, "last_params.pth"),
                {
                    "model": make_model_for_eval(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_clean_acc": best_clean,
                    "best_robust_acc": best_robust,
                    "args": vars(args),
                    "init_model": init_sd,
                },
            )

            wandb_log({
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": test_loss,
                "val/clean_acc": test_acc,
                "val/robust_acc": robust_acc,
                "epoch": epoch + 1,
                "best/metric": best_metric,
                "best/clean_acc": best_clean,
                "best/robust_acc": best_robust,
            }, step=epoch + 1)
        else:
            wandb_log({
                "train/loss": train_loss,
                "train/acc": train_acc,
                "epoch": epoch + 1,
            }, step=epoch + 1)

        scheduler.step()

    logger.info("==> Loading best checkpoint for final report...")
    best_ckpt = torch.load(os.path.join(model_save_dir, "best_params.pth"), map_location=args.device)
    model.load_state_dict(best_ckpt["model"])

    final_test_loss, final_test_acc = evaluate(args, model, testloader, criterion)
    final_robust_acc = eval_robustness(args, make_model_for_eval(model))
    logger.info("==> Finetune test acc: {:.2f}%, robust acc: {:.2f}%".format(final_test_acc, final_robust_acc))
    logger.info("==> Best epoch: {}, best clean acc: {:.2f}%, best robust acc: {:.2f}%".format(best_epoch, best_clean, best_robust))

    wandb_log({
        "final/test_loss": final_test_loss,
        "final/test_acc": final_test_acc,
        "final/robust_acc": final_robust_acc,
        "final/best_epoch": best_epoch,
    }, step=args.epochs)

    wandb.finish()


if __name__ == "__main__":
    main()
