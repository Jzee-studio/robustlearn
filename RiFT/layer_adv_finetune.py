# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchattacks
from tqdm import tqdm

from dataloader import create_dataloader
from model import create_model
from optimizer import create_optimizer, create_scheduler
from utils import Normalize, create_logger, evaluate, evaluate_cifar_robustness, evaluate_tiny_robustness


def _build_transforms(dataset):
    if "CIFAR" in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
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
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return {"train": transform_train, "test": transform_test}


def _normalize_layer_name(layer_name):
    if layer_name.startswith("module."):
        return layer_name[len("module."):]
    if layer_name.startswith("backbone."):
        return layer_name[len("backbone."):]
    if layer_name.startswith("1."):
        return layer_name[len("1."):]
    return layer_name


def _collect_layers(model):
    layers = []
    for name, module in model.named_modules():
        normalized_name = _normalize_layer_name(name)
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if "sub" in normalized_name:
                continue
            if normalized_name == "":
                continue
            layers.append(normalized_name)
    return sorted(set(layers))


def _set_trainable_layers(model, target_layer):
    normalized_target_layer = _normalize_layer_name(target_layer)
    for name, param in model.named_parameters():
        normalized_name = _normalize_layer_name(name)
        param.requires_grad = normalized_target_layer in normalized_name


def _build_norm_model(args, model):
    if "CIFAR" in args.dataset:
        norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    else:
        norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return nn.Sequential(norm_layer, model).to(args.device)


def _create_adv_trainloader(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.dataset == "CIFAR10":
        dataset = datasets.CIFAR10(root=os.path.join(args.data_root, "CIFAR-10"), train=False, download=True, transform=transform_test)
    elif args.dataset == "CIFAR100":
        dataset = datasets.CIFAR100(root=os.path.join(args.data_root, "CIFAR-100"), train=False, download=True, transform=transform_test)
    else:
        from dataloader import TinyImageNet
        dataset = TinyImageNet("train", transform_test, data_root=args.data_root)

    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)


def _create_attack(args, model):
    return torchattacks.PGD(model, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True)


def _evaluate_clean(args, model, dataloader, criterion):
    model.eval()
    total = 0
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total += targets.size(0)
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    return total_loss / total, 100.0 * correct / total


def _train_one_layer_adv(args, model, trainloader, testloader, criterion, optimizer, scheduler, eval_robustness_func):
    best_test_acc = 0.0
    best_state = deepcopy(model.state_dict())

    atk_model = _create_attack(args, model)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(trainloader, desc=f"{args.layer} epoch {epoch}", leave=False):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            adv_inputs = atk_model(inputs, targets)

            optimizer.zero_grad()
            outputs = model(adv_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        scheduler.step()

        clean_loss, clean_acc = _evaluate_clean(args, model, testloader, criterion)
        robust_acc = eval_robustness_func(args, model)
        train_acc = 100.0 * train_correct / train_total
        train_loss /= train_total
        args.logger.info(
            "layer=%s epoch=%d train_loss=%.4f train_acc=%.2f%% clean_loss=%.4f clean_acc=%.2f%% robust_acc=%.2f%%",
            args.layer,
            epoch,
            train_loss,
            train_acc,
            clean_loss,
            clean_acc,
            robust_acc,
        )

        if clean_acc > best_test_acc:
            best_test_acc = clean_acc
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_test_acc, best_state


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Layer-wise PGD adversarial finetuning")
    parser.add_argument("--model", default="ResNet18", type=str, help="model used")
    parser.add_argument("--dataset", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--input_size", default=32, type=int)
    parser.add_argument("--layer", required=True, type=str, help="trainable layer")
    parser.add_argument("--resume", default=None, type=str, help="resume checkpoint")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--data_root", default="E:/Jzee4Study/dataset", type=str)
    parser.add_argument("--patch", default=4, type=int)
    parser.add_argument("--optim", default="SGDM", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--lr_scheduler", default="step", choices=["step", "cosine"])
    parser.add_argument("--lr_decay_gamma", default=0.1, type=float)
    parser.add_argument("--wd", default=0.0005, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save_dir", default=None, type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    save_dir = args.save_dir or os.path.join("./results", f"{args.model}_{args.dataset}", "layer_adv_finetune")
    os.makedirs(save_dir, exist_ok=True)
    logger = create_logger(os.path.join(save_dir, "output.log"))
    args.logger = logger
    logger.info(args)

    transform_dict = _build_transforms(args.dataset)
    trainloader, _, testloader = create_dataloader(
        args.dataset,
        args.batch_size,
        use_val=False,
        transform_dict=transform_dict,
        data_root=args.data_root,
    )

    model = create_model(args.model, args.input_size, args.num_classes, args.device, args.patch, args.resume)
    criterion = nn.CrossEntropyLoss()

    target_layers = _collect_layers(model)
    if args.layer not in target_layers:
        raise ValueError(f"layer '{args.layer}' not found in model")

    _set_trainable_layers(model, args.layer)
    optimizer = create_optimizer(args.optim, model, args.lr, args.momentum, weight_decay=args.wd)
    scheduler = create_scheduler(args, optimizer)

    if "CIFAR" in args.dataset:
        eval_robustness_func = evaluate_cifar_robustness
    else:
        eval_robustness_func = evaluate_tiny_robustness

    _, init_clean_acc = _evaluate_clean(args, model, testloader, criterion)
    init_robust_acc = eval_robustness_func(args, model)
    logger.info("Init clean acc=%.2f%%, robust acc=%.2f%%", init_clean_acc, init_robust_acc)

    init_sd = deepcopy(model.state_dict())
    best_clean_acc, best_state = _train_one_layer_adv(
        args,
        model,
        trainloader,
        testloader,
        criterion,
        optimizer,
        scheduler,
        eval_robustness_func,
    )

    torch.save(
        {
            "model": best_state,
            "layer": args.layer,
            "best_clean_acc": best_clean_acc,
            "init_state": init_sd,
        },
        os.path.join(save_dir, f"{args.layer.replace('.', '_')}_best_advft.pth"),
    )

    final_clean_loss, final_clean_acc = _evaluate_clean(args, model, testloader, criterion)
    final_robust_acc = eval_robustness_func(args, model)
    logger.info("Final clean loss=%.4f clean acc=%.2f%% robust acc=%.2f%%", final_clean_loss, final_clean_acc, final_robust_acc)
    logger.info("Best clean acc during adv finetune: %.2f%%", best_clean_acc)


if __name__ == "__main__":
    main()
