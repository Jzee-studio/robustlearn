import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets

from model import create_model


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_class_names(dataset_name: str) -> List[str]:
    if dataset_name == "CIFAR10":
        return CIFAR10_CLASSES
    if dataset_name == "CIFAR100":
        return list(datasets.CIFAR100(root="./data", train=False, download=True).classes)
    if dataset_name == "TinyImageNet":
        # Tiny-ImageNet class names are image-folder subdirectory names (wnids)
        return []
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_transform(dataset_name: str):
    if dataset_name in {"CIFAR10", "CIFAR100"}:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        input_size = 32
    elif dataset_name == "TinyImageNet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        input_size = 64
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def find_images(input_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if Path(name).suffix.lower() in exts:
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths


def load_state_dict_from_checkpoint(ckpt_path: str, device: str):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "net"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def normalize_state_dict_keys(state_dict):
    """Return variants with and without DataParallel `module.` prefix."""
    variants = [state_dict]
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

    if has_module_prefix:
        stripped = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
        variants.append(stripped)
    else:
        prefixed = {f"module.{k}": v for k, v in state_dict.items()}
        variants.append(prefixed)

    return variants


def load_model(args):
    model = create_model(
        args.model,
        args.input_size,
        args.num_classes,
        args.device,
        args.patch,
        resume=None,
    )

    state_dict = load_state_dict_from_checkpoint(args.ckpt, args.device)
    candidate_state_dicts = normalize_state_dict_keys(state_dict)

    last_error = None
    for candidate in candidate_state_dicts:
        try:
            model.load_state_dict(candidate)
            break
        except RuntimeError as err:
            last_error = err
            try:
                model.load_state_dict(candidate, strict=False)
                break
            except RuntimeError as err2:
                last_error = err2
    else:
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        shared = len(model_keys & ckpt_keys)
        raise RuntimeError(
            f"Failed to load checkpoint '{args.ckpt}' into model '{args.model}'.\n"
            f"Check whether `--model`, `--num_classes`, and `--input_size` match the training config.\n"
            f"Model params: {len(model_keys)}, checkpoint params: {len(ckpt_keys)}, shared keys: {shared}.\n"
            f"Last loading error: {last_error}"
        )

    model = model.to(args.device)
    model.eval()
    return model


@torch.no_grad()
def infer_folder(args) -> List[Tuple[str, int, str, float]]:
    class_names = get_class_names(args.dataset)
    transform = get_transform(args.dataset)
    model = load_model(args)

    image_paths = find_images(args.input_dir)
    if not image_paths:
        raise RuntimeError(f"No image files found in: {args.input_dir}")

    results = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(args.device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        pred_idx = int(pred.item())
        confidence = float(conf.item())
        pred_name = class_names[pred_idx] if class_names and pred_idx < len(class_names) else str(pred_idx)
        results.append((img_path, pred_idx, pred_name, confidence))

    return results


def main():
    parser = argparse.ArgumentParser(description="Infer all images in a folder with a trained model")
    parser.add_argument("--model", default="ResNet18", type=str, help="model architecture")
    parser.add_argument("--dataset", default="CIFAR10", choices=["CIFAR10", "CIFAR100", "TinyImageNet"], help="dataset type")
    parser.add_argument("--num_classes", default=10, type=int, help="number of classes")
    parser.add_argument("--input_size", default=32, type=int, help="input image size")
    parser.add_argument("--patch", default=4, type=int, help="patch size for ViT")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="device")
    parser.add_argument("--ckpt", required=True, type=str, help="path to checkpoint")
    parser.add_argument("--input_dir", required=True, type=str, help="directory containing images")
    parser.add_argument("--output_csv", default="predictions.csv", type=str, help="path to save results")

    args = parser.parse_args()

    results = infer_folder(args)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "pred_idx", "pred_class", "confidence"])
        for row in results:
            writer.writerow(row)

    print(f"Saved {len(results)} predictions to {args.output_csv}")
    for row in results[:10]:
        print(f"{row[0]} -> {row[2]} ({row[1]}), confidence={row[3]:.4f}")


if __name__ == "__main__":
    main()
