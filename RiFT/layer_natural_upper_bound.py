# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from dataloader import create_dataloader
from model import create_model
from optimizer import create_optimizer, create_scheduler
from utils import create_logger, evaluate


def _build_transforms(dataset):
    if "CIFAR" in dataset:
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
    return {"train": transform_train, "test": transform_test}


def _collect_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if "sub" in name:
                continue
            if name == "":
                continue
            layers.append(name)
    return layers


def _set_trainable_layers(model, target_layer):
    for name, param in model.named_parameters():
        param.requires_grad = target_layer in name


def _train_one_layer(args, model, trainloader, testloader, criterion, optimizer, scheduler):
    best_test_acc = 0.0
    best_state = deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(trainloader, desc=f"{args.layer} epoch {epoch}", leave=False):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        scheduler.step()

        _, test_acc = evaluate(args, model, testloader, criterion)
        train_acc = 100.0 * train_correct / train_total
        train_loss /= train_total
        args.logger.info(
            "layer=%s epoch=%d train_loss=%.4f train_acc=%.2f%% test_acc=%.2f%%",
            args.layer,
            epoch,
            train_loss,
            train_acc,
            test_acc,
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_test_acc, best_state


def main():
    parser = torch.optim.Optimizer

    import argparse

    parser = argparse.ArgumentParser(description="Layer-wise natural accuracy upper bound evaluation")
    parser.add_argument("--model", default="ResNet18", type=str, help="model used")
    parser.add_argument("--dataset", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--input_size", default=32, type=int)
    parser.add_argument("--layer", default=None, type=str, help="if set, only evaluate one layer; otherwise evaluate all layers")
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

    save_dir = args.save_dir or os.path.join(
        "./results",
        f"{args.model}_{args.dataset}",
        "natural_upper_bound",
    )
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

    layers = _collect_layers(model)
    if args.layer is not None:
        if args.layer not in layers:
            raise ValueError(f"layer '{args.layer}' not found in model")
        layers = [args.layer]

    results = []
    for layer_name in layers:
        logger.info("Evaluating layer %s", layer_name)
        layer_model = create_model(args.model, args.input_size, args.num_classes, args.device, args.patch, args.resume)
        _set_trainable_layers(layer_model, layer_name)
        optimizer = create_optimizer(args.optim, layer_model, args.lr, args.momentum, weight_decay=args.wd)
        scheduler = create_scheduler(args, optimizer)

        init_sd = deepcopy(layer_model.state_dict())
        best_test_acc, best_state = _train_one_layer(args, layer_model, trainloader, testloader, criterion, optimizer, scheduler)
        results.append((layer_name, best_test_acc))
        torch.save({"model": best_state, "layer": layer_name, "best_test_acc": best_test_acc, "init_state": init_sd}, os.path.join(save_dir, f"{layer_name.replace('.', '_')}_best.pth"))
        logger.info("layer=%s best_natural_test_acc=%.2f%%", layer_name, best_test_acc)

    results = sorted(results, key=lambda x: x[1], reverse=True)
    logger.info("===== Summary (sorted by best natural test acc) =====")
    for layer_name, acc in results:
        logger.info("%s, best_test_acc=%.2f%%", layer_name, acc)


if __name__ == "__main__":
    main()
