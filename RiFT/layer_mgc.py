# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from dataloader import TinyImageNet
from model import create_model
from optimizer import create_optimizer, create_scheduler
from utils import Normalize, create_logger


def _build_eval_model(args, model):
    if "CIFAR" in args.dataset:
        norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    else:
        norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return nn.Sequential(norm_layer, model).to(args.device)


def _normalize_layer_name(layer_name):
    if layer_name.startswith("1.module."):
        return layer_name[len("1.module."):]
    if layer_name.startswith("module."):
        return layer_name[len("module."):]
    if layer_name.startswith("1."):
        return layer_name[len("1."):]
    return layer_name


def _matching_param_name(layer_name):
    normalized = _normalize_layer_name(layer_name)
    return [
        f"{normalized}.weight",
        f"1.{normalized}.weight",
        f"module.{normalized}.weight",
        f"1.module.{normalized}.weight",
    ]


def _collect_target_layers(model):
    layer_names = []
    for name, module in model.named_modules():
        normalized_name = _normalize_layer_name(name)
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if "sub" in normalized_name:
                continue
            if normalized_name == "" or normalized_name == "module":
                continue
            layer_names.append(normalized_name)
    return sorted(set(layer_names))


def _create_test_loader(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.dataset == "CIFAR10":
        dataset = datasets.CIFAR10(root=os.path.join(args.data_root, "CIFAR-10"), train=False, download=True, transform=transform_test)
    elif args.dataset == "CIFAR100":
        dataset = datasets.CIFAR100(root=os.path.join(args.data_root, "CIFAR-100"), train=False, download=True, transform=transform_test)
    else:
        dataset = TinyImageNet("val", transform_test, data_root=args.data_root)

    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)


def _build_dataset_and_model(args):
    model = create_model(args.model, args.input_size, args.num_classes, args.device, args.patch, args.resume)
    model = _build_eval_model(args, model)
    testloader = _create_test_loader(args)
    return model, testloader


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


def _evaluate_mgc_group(args, model, testloader, criterion, layer_names, epsilon=0.1):
    cloned_model = deepcopy(model)
    normalized_layer_names = {_normalize_layer_name(layer) for layer in layer_names}

    for name, param in cloned_model.named_parameters():
        param.requires_grad = False
        if any(name == candidate for layer in normalized_layer_names for candidate in _matching_param_name(layer)):
            param.requires_grad = True

    trainable_params = [name for name, param in cloned_model.named_parameters() if param.requires_grad]
    if not trainable_params:
        return None

    init_params = {name: param.detach().clone() for name, param in cloned_model.named_parameters() if param.requires_grad}
    optimizer = torch.optim.SGD(cloned_model.parameters(), lr=1)

    max_loss = 0.0
    min_acc = 0.0

    for _ in range(10):
        for inputs, targets in testloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = cloned_model(inputs)
            loss = -1 * criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        state_dict = cloned_model.state_dict()
        for layer_name in normalized_layer_names:
            for candidate in _matching_param_name(layer_name):
                if candidate in state_dict and candidate in init_params:
                    diff = state_dict[candidate] - init_params[candidate]
                    times = torch.linalg.norm(diff) / torch.linalg.norm(init_params[candidate])
                    if times > epsilon:
                        diff = diff / times * epsilon
                        state_dict[candidate] = deepcopy(init_params[candidate] + diff)
                    break
        cloned_model.load_state_dict(state_dict)

        total_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = cloned_model(inputs)
                loss = criterion(outputs, targets)
                total += targets.size(0)
                total_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
        total_loss /= total
        correct /= total

        if total_loss > max_loss:
            max_loss = total_loss
            min_acc = correct

    return max_loss, min_acc


def compute_mgc(args, model, testloader, epsilon=0.1):
    criterion = nn.CrossEntropyLoss()
    origin_loss, origin_acc = _evaluate_clean(args, model, testloader, criterion)
    args.logger.info("{:35}, Clean Loss: {:10.2f}, Clean Acc: {:10.2f}".format("Origin", origin_loss, origin_acc))

    layer_mgc_dict = {}
    target_layers = _collect_target_layers(model)

    if args.layer is None:
        for layer_name in target_layers:
            result = _evaluate_mgc_group(args, model, testloader, criterion, [layer_name], epsilon=epsilon)
            if result is None:
                continue
            max_loss, min_acc = result
            layer_mgc_dict[layer_name] = max_loss - origin_loss
            args.logger.info("{:35}, MGC: {:10.2f}, Dropped Clean Acc: {:10.2f}".format(layer_name, max_loss - origin_loss, origin_acc - min_acc * 100))
    else:
        normalized_target_layers = {_normalize_layer_name(layer): layer for layer in target_layers}
        normalized_specified_layer = _normalize_layer_name(args.layer)
        if normalized_specified_layer not in normalized_target_layers:
            raise ValueError(f"Specified layer '{args.layer}' was not found among eligible Conv2d/Linear layers.")

        canonical_specified_layer = normalized_target_layers[normalized_specified_layer]
        for other_layer in target_layers:
            if _normalize_layer_name(other_layer) == normalized_specified_layer:
                continue
            result = _evaluate_mgc_group(args, model, testloader, criterion, [canonical_specified_layer, other_layer], epsilon=epsilon)
            if result is None:
                continue
            max_loss, min_acc = result
            group_name = f"{args.layer} + {other_layer}"
            layer_mgc_dict[group_name] = max_loss - origin_loss
            args.logger.info("{:35}, MGC: {:10.2f}, Dropped Clean Acc: {:10.2f}".format(group_name, max_loss - origin_loss, origin_acc - min_acc * 100))

    sorted_layer_mgc = sorted(layer_mgc_dict.items(), key=lambda x: x[1])
    for (k, v) in sorted_layer_mgc:
        args.logger.info("{:35}, Clean Loss: {:10.2f}".format(k, v))

    return sorted_layer_mgc


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute Module Generalization Criticality (MGC)")
    parser.add_argument("--model", default="ResNet18", type=str)
    parser.add_argument("--dataset", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--input_size", default=32, type=int)
    parser.add_argument("--layer", default=None, type=str, help="If set, compute pairwise MGC for this layer against all others")
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--data_root", default="E:/Jzee4Study/dataset", type=str)
    parser.add_argument("--patch", default=4, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save_dir", default=None, type=str)
    parser.add_argument("--epsilon", default=0.1, type=float)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    save_dir = args.save_dir or os.path.join("./results", f"{args.model}_{args.dataset}", "mgc")
    os.makedirs(save_dir, exist_ok=True)
    args.logger = create_logger(os.path.join(save_dir, "output.log"))
    args.logger.info(args)

    model, testloader = _build_dataset_and_model(args)
    compute_mgc(args, model, testloader, epsilon=args.epsilon)


if __name__ == "__main__":
    main()
