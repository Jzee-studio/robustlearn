# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Model visualization analysis script.
- Randomly samples 5 correctly classified and 5 misclassified test images.
- Copies sampled images to ./vis_demo/
- Generates Grad-CAM heatmaps and saves to ./vis_grad/
"""

import argparse
import os
import shutil
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import create_model
from dataloader import TinyImageNet, create_dataloader
from utils import Normalize


# ---------------------------------------------------------------------------
# Grad-CAM utilities
# ---------------------------------------------------------------------------


class GradCAMHook:
    """Hook to capture feature maps and gradients from a target layer."""

    def __init__(self, module):
        self.feature_maps = None
        self.gradients = None
        self.forward_handle = module.register_forward_hook(self._forward_hook)
        self.backward_handle = module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.feature_maps = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def find_target_layer(model, target_name):
    """Find a module by name (supports dotted paths like 'layer4')."""
    if target_name is None:
        # Auto-select: last conv layer before the classifier
        candidates = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                candidates.append(name)
        if not candidates:
            raise ValueError("No Conv2d layer found in model.")
        target_name = candidates[-1]
        print(f"Auto-selected target layer: {target_name}")
    for name, module in model.named_modules():
        if name == target_name:
            return module, name
    raise ValueError(f"Target layer '{target_name}' not found in model.")


def compute_gradcam(hook, target_class=None):
    """Compute the Grad-CAM heatmap from captured features and gradients."""
    features = hook.feature_maps
    grads = hook.gradients

    if features is None or grads is None:
        raise RuntimeError("No captured features/gradients. Run a forward+backward pass first.")

    # Global average pool the gradients
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

    # Weighted combination of feature maps
    cam = (weights * features).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    cam = F.relu(cam)  # keep only positive contributions

    # Normalize per image
    B = cam.size(0)
    cam_flat = cam.view(B, -1)
    cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    denom = cam_max - cam_min
    denom[denom == 0] = 1.0
    cam = (cam - cam_min) / denom

    return cam


def apply_heatmap(img_tensor, cam_tensor):
    """
    Overlay a Grad-CAM heatmap onto an image tensor.
    img_tensor: [3, H, W] in [0, 1] (denormalized)
    cam_tensor: [1, h, w] in [0, 1]
    Returns: [3, H, W] overlay in [0, 1]
    """
    cam_resized = F.interpolate(
        cam_tensor.unsqueeze(0), size=img_tensor.shape[-2:], mode='bilinear', align_corners=False
    ).squeeze(0).squeeze(0)  # [H, W]

    heatmap = plt.cm.jet(cam_resized.cpu().numpy())[:, :, :3]  # [H, W, 3] RGB in [0,1]
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).to(img_tensor.device)  # [3, H, W]

    overlay = 0.5 * img_tensor.cpu() + 0.5 * heatmap.cpu()
    overlay = torch.clamp(overlay, 0.0, 1.0)
    return overlay


def denormalize(img_tensor, mean, std):
    """Reverse Normalize transform to get viewable image."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return img_tensor.cpu() * std + mean


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------


def sample_images(args, model, dataloader, num_correct=5, num_wrong=5):
    """
    Collect random samples of correctly and incorrectly classified images.
    Returns two lists of dicts with keys: image (tensor), label, pred, index
    """
    model.eval()

    all_correct = []
    all_wrong = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Collecting samples")):
            inputs_dev = inputs.to(args.device)
            targets_dev = targets.to(args.device)
            outputs = model(inputs_dev)
            _, preds = outputs.max(1)

            for i in range(inputs.size(0)):
                entry = {
                    "image": inputs[i].clone(),
                    "label": targets[i].item(),
                    "pred": preds[i].item(),
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                }
                if preds[i] == targets[i]:
                    if len(all_correct) < num_correct * 20:
                        all_correct.append(entry)
                else:
                    if len(all_wrong) < num_wrong * 20:
                        all_wrong.append(entry)

            if len(all_correct) >= num_correct * 20 and len(all_wrong) >= num_wrong * 20:
                break

    # Randomly sample
    random.seed(args.seed)
    sampled_correct = random.sample(all_correct, min(num_correct, len(all_correct)))
    sampled_wrong = random.sample(all_wrong, min(num_wrong, len(all_wrong)))

    if len(sampled_correct) < num_correct:
        print(f"Warning: only {len(sampled_correct)} correct samples available.")
    if len(sampled_wrong) < num_wrong:
        print(f"Warning: only {len(sampled_wrong)} wrong samples available.")

    return sampled_correct, sampled_wrong


# ---------------------------------------------------------------------------
# Main visualization routine
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM model visualization")
    parser.add_argument("--model", default="ResNet18", type=str, help="model architecture")
    parser.add_argument("--dataset", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--input_size", default=32, type=int)
    parser.add_argument("--patch", default=4, type=int, help="ViT patch size")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--weights", default=None, type=str, help="path to model weights (.pth)")
    parser.add_argument("--target_layer", default=None, type=str,
                        help="target conv layer for Grad-CAM (auto-detected if omitted)")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_correct", default=5, type=int)
    parser.add_argument("--num_wrong", default=5, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Dataset-specific settings
    # ------------------------------------------------------------------
    if "CIFAR" in args.dataset:
        mean, std = [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # Transform without normalization for saving raw images
    transform_raw = transforms.Compose([transforms.ToTensor()])

    # Standard test transform (normalized) for model inference
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "CIFAR10":
        testset_raw = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_raw)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == "CIFAR100":
        testset_raw = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_raw)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        testset_raw = TinyImageNet("val", transform=transform_raw)
        testset = TinyImageNet("val", transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    model = create_model(args.model, args.input_size, args.num_classes, args.device, args.patch)

    if args.weights is not None:
        print(f"Loading weights from: {args.weights}")
        ckpt = torch.load(args.weights, map_location=args.device)
        # Handle various checkpoint formats
        if "net" in ckpt:
            sd = ckpt["net"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt

        # Remove 'module.' prefix if present (DataParallel)
        new_sd = {}
        for k, v in sd.items():
            new_sd[k.replace("module.", "")] = v
        model.load_state_dict(new_sd, strict=False)

    model.eval()

    # ------------------------------------------------------------------
    # Sample correct / wrong predictions
    # ------------------------------------------------------------------
    print("Collecting correct and incorrect predictions...")
    sampled_correct, sampled_wrong = sample_images(
        args, model, testloader, num_correct=args.num_correct, num_wrong=args.num_wrong
    )

    print(f"Sampled correct: {len(sampled_correct)}, sampled wrong: {len(sampled_wrong)}")

    # ------------------------------------------------------------------
    # Prepare output directories
    # ------------------------------------------------------------------
    vis_demo_dir = os.path.join(os.getcwd(), "vis_demo")
    vis_grad_dir = os.path.join(os.getcwd(), "vis_grad")

    for d in [vis_demo_dir, vis_grad_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # ------------------------------------------------------------------
    # Set up Grad-CAM
    # ------------------------------------------------------------------
    target_module, target_name = find_target_layer(model, args.target_layer)
    print(f"Grad-CAM target layer: {target_name}")

    grad_cam_hook = GradCAMHook(target_module)

    all_samples = [("correct", sampled_correct), ("wrong", sampled_wrong)]

    for category, samples in all_samples:
        for idx, entry in enumerate(samples):
            img_raw = testset_raw[entry["batch_idx"] * args.batch_size + entry["sample_idx"]][0]
            img_normalized = entry["image"].unsqueeze(0).to(args.device)

            # ----------------------------------------------------------
            # Copy raw image to vis_demo
            # ----------------------------------------------------------
            img_pil = transforms.ToPILImage()(img_raw)
            demo_name = f"{category}_{idx+1}_label{entry['label']}_pred{entry['pred']}.png"
            img_pil.save(os.path.join(vis_demo_dir, demo_name))

            # ----------------------------------------------------------
            # Grad-CAM
            # ----------------------------------------------------------
            model.zero_grad()
            img_normalized.requires_grad = False

            # Forward
            output = model(img_normalized)
            pred_class = output.argmax(dim=1).item()

            # Backward on the predicted class
            one_hot = torch.zeros_like(output)
            one_hot[0, pred_class] = 1.0
            output.backward(gradient=one_hot)

            cam = compute_gradcam(grad_cam_hook)  # [1, 1, h, w]

            # Denormalize image for overlay
            img_denorm = denormalize(entry["image"].clone(), mean, std)

            overlay = apply_heatmap(img_denorm, cam[0])  # [3, H, W]

            # ----------------------------------------------------------
            # Save Grad-CAM figure
            # ----------------------------------------------------------
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            axes[0].imshow(img_denorm.permute(1, 2, 0).clamp(0, 1))
            axes[0].set_title(f"Original\nLabel: {entry['label']}, Pred: {pred_class}")
            axes[0].axis("off")

            axes[1].imshow(overlay.permute(1, 2, 0))
            axes[1].set_title(f"Grad-CAM ({target_name})\nLabel: {entry['label']}, Pred: {pred_class}")
            axes[1].axis("off")

            grad_name = f"{category}_{idx+1}_label{entry['label']}_pred{entry['pred']}_gradcam.png"
            fig.savefig(os.path.join(vis_grad_dir, grad_name), bbox_inches="tight", dpi=150)
            plt.close(fig)

            verdict = "CORRECT" if category == "correct" else "WRONG"
            print(f"  [{verdict}] {demo_name} -> saved.")

    grad_cam_hook.close()

    print(f"\nDone. Sampled images saved to: {vis_demo_dir}")
    print(f"Grad-CAM visualizations saved to: {vis_grad_dir}")


if __name__ == "__main__":
    main()
