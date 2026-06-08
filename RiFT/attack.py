# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F


_DATASET_STATS = {
    "CIFAR10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2471, 0.2435, 0.2616),
    },
    "CIFAR100": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2471, 0.2435, 0.2616),
    },
    "TinyImageNet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}


def _get_stats(dataset: str):
    if dataset not in _DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return _DATASET_STATS[dataset]["mean"], _DATASET_STATS[dataset]["std"]


def get_normalized_bounds(dataset: str, device: torch.device | str):
    mean, std = _get_stats(dataset)
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return lower, upper


def clamp_normalized(x: torch.Tensor, dataset: str):
    lower, upper = get_normalized_bounds(dataset, x.device)
    return torch.max(torch.min(x, upper), lower)


def _project_linf(x_adv: torch.Tensor, x_nat: torch.Tensor, eps: float, dataset: str):
    x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)
    return clamp_normalized(x_adv, dataset)


def _linf_random_start(x_nat: torch.Tensor, eps: float, dataset: str):
    x_adv = x_nat + torch.empty_like(x_nat).uniform_(-eps, eps)
    return clamp_normalized(x_adv, dataset)


def pgd_attack(
    model,
    x_nat: torch.Tensor,
    y: torch.Tensor,
    dataset: str,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
):
    """Generate Linf-PGD adversarial examples on normalized inputs."""
    model.eval()
    x_adv = x_nat.detach().clone()
    if random_start:
        x_adv = _linf_random_start(x_adv, eps, dataset)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = _project_linf(x_adv, x_nat, eps, dataset)

    return x_adv.detach()


def trades_loss(
    model,
    x_nat: torch.Tensor,
    y: torch.Tensor,
    dataset: str,
    eps: float,
    alpha: float,
    steps: int,
    beta: float,
    random_start: bool = True,
):
    """Return TRADES loss and detached logits for monitoring.

    The loss follows the standard formulation:
        CE(f(x), y) + beta * KL(f(x) || f(x_adv))
    where x_adv is optimized to maximize the KL divergence.
    """
    model.eval()
    batch_size = x_nat.size(0)
    x_adv = x_nat.detach().clone()
    if random_start:
        x_adv = _linf_random_start(x_adv, eps, dataset)
    else:
        x_adv = clamp_normalized(x_adv, dataset)

    with torch.no_grad():
        logits_nat = model(x_nat)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_adv = model(x_adv)
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_nat, dim=1),
                reduction="batchmean",
            )
        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = _project_linf(x_adv, x_nat, eps, dataset)

    model.train()
    logits_nat = model(x_nat)
    logits_adv = model(x_adv)

    loss_natural = F.cross_entropy(logits_nat, y)
    loss_robust = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_nat.detach(), dim=1),
        reduction="batchmean",
    )
    loss = loss_natural + beta * loss_robust

    return loss, loss_natural.detach(), loss_robust.detach(), logits_nat.detach(), logits_adv.detach()
