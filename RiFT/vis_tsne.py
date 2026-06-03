# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""t-SNE visualization for RiFT models.

This script follows the same command-line style as `main.py` and reuses the
project's model / dataloader / utility helpers. It extracts penultimate
representations from a trained model, runs t-SNE, and saves a scatter plot for
train / test / optional adversarial examples.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms

try:
    from sklearn.manifold import TSNE
except ImportError as exc:
    TSNE = None
    sklearn_import_error = exc
else:
    sklearn_import_error = None

from dataloader import TinyImageNet
from model import create_model
from utils import create_logger


class FeatureModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module

        # ResNet-style backbones in this repo end with a linear classifier named `linear`.
        if hasattr(model, "conv1") and hasattr(model, "linear"):
            out = model.conv1(x)
            if hasattr(model, "bn1"):
                out = model.bn1(out)
            if hasattr(model, "relu"):
                out = model.relu(out)
            if hasattr(model, "maxpool"):
                out = model.maxpool(out)
            if hasattr(model, "layer1"):
                out = model.layer1(out)
            if hasattr(model, "layer2"):
                out = model.layer2(out)
            if hasattr(model, "layer3"):
                out = model.layer3(out)
            if hasattr(model, "layer4"):
                out = model.layer4(out)
            if hasattr(model, "avgpool"):
                out = model.avgpool(out)
            if out.dim() > 2:
                out = torch.flatten(out, 1)
            return out

        # Fallback: traverse children until the classifier-like head.
        feats = x
        for name, module in model.named_children():
            if name in {"linear", "fc", "classifier"}:
                break
            feats = module(feats)
        if feats.dim() > 2:
            feats = torch.flatten(feats, 1)
        return feats


def build_transform(dataset):
    if "CIFAR" in dataset:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def get_dataset(args, split):
    transform = build_transform(args.dataset)
    if args.dataset == "CIFAR10":
        return datasets.CIFAR10(root=os.path.join(args.data_root, "CIFAR-10"), train=(split == "train"), download=True, transform=transform)
    if args.dataset == "CIFAR100":
        return datasets.CIFAR100(root=os.path.join(args.data_root, "CIFAR-100"), train=(split == "train"), download=True, transform=transform)
    return TinyImageNet("train" if split == "train" else "val", transform, data_root=args.data_root)


def build_dataloader(args, split):
    dataset = get_dataset(args, split)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_model_for_vis(args):
    model = create_model(args.model, args.input_size, args.num_classes, args.device, args.patch, args.resume)
    model.eval()
    return model


def collect_features(args, model, dataloader, max_samples=None):
    feature_model = FeatureModel(model).to(args.device)
    feature_model.eval()

    features = []
    labels = []
    collected = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(args.device)
            feats = feature_model(inputs)
            feats = feats.detach().float().cpu()

            features.append(feats)
            labels.append(targets.cpu())
            collected += targets.size(0)

            if max_samples is not None and collected >= max_samples:
                break

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    if max_samples is not None:
        features = features[:max_samples]
        labels = labels[:max_samples]

    return features.numpy(), labels.numpy()


def plot_tsne(embeddings, labels, class_names, title, save_path):
    plt.figure(figsize=(10, 8))
    num_classes = int(labels.max()) + 1 if labels.size > 0 else 0
    cmap = plt.get_cmap("tab20", max(num_classes, 1))

    for cls in np.unique(labels):
        idx = labels == cls
        plt.scatter(
            embeddings[idx, 0],
            embeddings[idx, 1],
            s=12,
            alpha=0.75,
            color=cmap(int(cls) % cmap.N),
            label=class_names[int(cls)] if class_names and int(cls) < len(class_names) else str(int(cls)),
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if len(np.unique(labels)) <= 20:
        plt.legend(markerscale=1.5, fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    if TSNE is None:
        raise ImportError(
            "scikit-learn is required for t-SNE visualization. Please install it with `pip install scikit-learn`."
        ) from sklearn_import_error

    parser = argparse.ArgumentParser(description="RiFT t-SNE Visualization")
    parser.add_argument("--model", default="ResNet18", type=str, help="model used")
    parser.add_argument("--dataset", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
    parser.add_argument("--num_classes", default=10, type=int, help="num classes")
    parser.add_argument("--input_size", default=32, type=int, help="input_size")
    parser.add_argument("--patch", default=4, type=int, help="num patch (used by vit)")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--data_root", default="E:/Jzee4Study/dataset", type=str, help="dataset root directory")
    parser.add_argument("--resume", default=None, type=str, help="checkpoint path")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--num_workers", default=8, type=int, help="dataloader workers")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output_dir", default=None, type=str, help="directory to save visualization")
    parser.add_argument("--max_samples", default=2000, type=int, help="max samples to visualize per split")
    parser.add_argument("--perplexity", default=30, type=float, help="t-SNE perplexity")
    parser.add_argument("--include_test", action="store_true", help="also visualize the test split")
    parser.add_argument("--include_train", action="store_true", help="also visualize the train split")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir is None:
        suffix = f"{args.model}_{args.dataset}_tsne"
        if args.resume is not None:
            resume_name = os.path.splitext(os.path.basename(args.resume))[0]
            suffix += f"_{resume_name}"
        args.output_dir = os.path.join("./results", f"{args.model}_{args.dataset}", "tsne", suffix)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = create_logger(os.path.join(args.output_dir, "output.log"))
    logger.info(args)

    if args.dataset == "CIFAR10":
        args.num_classes = 10
        args.input_size = 32
        class_names = [str(i) for i in range(10)]
    elif args.dataset == "CIFAR100":
        args.num_classes = 100
        args.input_size = 32
        class_names = [str(i) for i in range(100)]
    else:
        args.num_classes = 200
        args.input_size = 64
        class_names = None

    logger.info("==> Building model...")
    model = load_model_for_vis(args)
    logger.info("==> Loading data...")

    splits = ["train"] if args.include_train else []
    if args.include_test or not splits:
        splits.append("test")

    for split in splits:
        dataloader = build_dataloader(args, split)
        feats, labels = collect_features(args, model, dataloader, max_samples=args.max_samples)

        logger.info("==> Running t-SNE for %s split with %d samples", split, feats.shape[0])
        tsne = TSNE(
            n_components=2,
            perplexity=min(args.perplexity, max(5, feats.shape[0] - 1)),
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        )
        embeddings = tsne.fit_transform(feats)

        save_path = os.path.join(args.output_dir, f"tsne_{split}.png")
        plot_tsne(embeddings, labels, class_names, f"{args.model} on {args.dataset} ({split})", save_path)
        logger.info("==> Saved %s", save_path)


if __name__ == "__main__":
    main()
