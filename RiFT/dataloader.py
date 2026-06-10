# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


def _apply_augmix_op(img, op_name, severity):
    if op_name == "autocontrast":
        return transforms.functional.autocontrast(img)
    if op_name == "equalize":
        return transforms.functional.equalize(img)
    if op_name == "brightness":
        return transforms.functional.adjust_brightness(img, 1.0 + 0.1 * severity * (random.random() * 2 - 1))
    if op_name == "contrast":
        return transforms.functional.adjust_contrast(img, 1.0 + 0.1 * severity * (random.random() * 2 - 1))
    if op_name == "saturation":
        return transforms.functional.adjust_saturation(img, 1.0 + 0.1 * severity * (random.random() * 2 - 1))
    if op_name == "sharpness":
        return transforms.functional.adjust_sharpness(img, 1.0 + 0.1 * severity * (random.random() * 2 - 1))
    if op_name == "posterize":
        return transforms.functional.posterize(img, max(1, 4 - int(severity / 2)))
    if op_name == "solarize":
        return transforms.functional.solarize(img, 256 - min(255, 30 * severity))
    if op_name == "rotate":
        return transforms.functional.rotate(img, random.uniform(-10, 10) * severity / 3)
    if op_name == "affine":
        return transforms.functional.affine(
            img,
            angle=0,
            translate=(int(random.uniform(-2, 2) * severity / 3), int(random.uniform(-2, 2) * severity / 3)),
            scale=1.0,
            shear=0,
        )
    return img


class AugMix(torch.nn.Module):
    def __init__(self, ops, severity=3, width=3, depth=-1, alpha=1.0):
        super().__init__()
        self.ops = ops
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def _sample_chain(self, img):
        ws = np.random.dirichlet([self.alpha] * self.width).astype(np.float32)
        m = float(np.random.beta(self.alpha, self.alpha))
        mix = torch.zeros_like(self.to_tensor(img))

        for i in range(self.width):
            image_aug = img.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op_name = random.choice(self.ops)
                image_aug = _apply_augmix_op(image_aug, op_name, self.severity)
            mix = mix + ws[i] * self.to_tensor(image_aug)

        image = self.to_tensor(img)
        mixed = (1.0 - m) * image + m * mix
        return self.to_pil(torch.clamp(mixed, 0.0, 1.0))

    def forward(self, img):
        return self._sample_chain(img)


def _cifar_augmix_ops():
    return ["autocontrast", "equalize", "brightness", "contrast", "saturation", "sharpness", "posterize", "solarize", "rotate", "affine"]


def _tiny_augmix_ops():
    return ["autocontrast", "equalize", "brightness", "contrast", "saturation", "sharpness", "posterize", "solarize", "rotate"]


def build_train_transform(dataset, use_augmix=False, augmix_width=3, augmix_depth=-1, augmix_severity=3):
    if dataset in ["CIFAR10", "CIFAR100"]:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        base = [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ]
        if use_augmix:
            base.append(AugMix(_cifar_augmix_ops(), severity=augmix_severity, width=augmix_width, depth=augmix_depth))
        base.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(base)

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    base = [
        transforms.RandomCrop(64, padding=8, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
    ]
    if use_augmix:
        base.append(AugMix(_tiny_augmix_ops(), severity=augmix_severity, width=augmix_width, depth=augmix_depth))
    base.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(base)


class TinyImageNet(Dataset):
    def __init__(self, dataset_type, transform=None, data_root="E:/Jzee4Study/dataset"):
        self.root = os.path.join(data_root, "tiny-imagenet-200")
        data_path = os.path.join(self.root, dataset_type)
        self.dataset = torchvision.datasets.ImageFolder(root=data_path)
        self.transform = transform

    def __getitem__(self, index):
        img, targets = self.dataset[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, targets

    def __len__(self):
        return self.dataset.__len__()


class TinyImageNetC(Dataset):
    def __init__(self, name, data_root="E:/Jzee4Study/dataset", level=1):
        self.corruptions = [
            "gaussian_noise",
            "shot_noise",
            "speckle_noise",
            "impulse_noise",
            "defocus_blur",
            "gaussian_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            "spatter",
            "saturate",
            "frost",
        ]
        assert name in self.corruptions
        self.root = os.path.join(data_root, "Tiny-ImageNet-C")
        data_path = os.path.join(self.root, name + "/" + str(level))
        self.dataset = torchvision.datasets.ImageFolder(root=data_path)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __getitem__(self, index):
        img, targets = self.dataset[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, targets

    def __len__(self):
        return self.dataset.__len__()


class CIFAR100C(Dataset):
    def __init__(self, name, data_root="E:/Jzee4Study/dataset"):
        self.corruptions = [
            "gaussian_noise",
            "shot_noise",
            "speckle_noise",
            "impulse_noise",
            "defocus_blur",
            "gaussian_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            "spatter",
            "saturate",
            "frost",
        ]
        assert name in self.corruptions
        self.root = os.path.join(data_root, "CIFAR-100-C")
        data_path = os.path.join(self.root, name + ".npy")
        target_path = os.path.join(self.root, "labels.npy")
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]
        )

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, targets

    def __len__(self):
        return len(self.data)


class CIFAR10C(Dataset):
    def __init__(self, name, data_root="E:/Jzee4Study/dataset"):
        self.corruptions = [
            "gaussian_noise",
            "shot_noise",
            "speckle_noise",
            "impulse_noise",
            "defocus_blur",
            "gaussian_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            "spatter",
            "saturate",
            "frost",
        ]
        assert name in self.corruptions
        self.root = os.path.join(data_root, "CIFAR-10-C")
        data_path = os.path.join(self.root, name + ".npy")
        target_path = os.path.join(self.root, "labels.npy")
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]
        )

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, targets

    def __len__(self):
        return len(self.data)


class adv_dataset(Dataset):
    def __init__(self):
        self.images = None
        self.labels = None

    def append_data(self, images, labels):
        if self.images is None:
            self.images = images
            self.labels = labels
        else:
            self.images = torch.cat((self.images, images), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)

    def __getitem__(self, item):
        img = self.images[item]
        label = self.labels[item]
        return img, label

    def __len__(self):
        return self.images.shape[0]


class mini_imagenet_dataset(Dataset):
    def __init__(self, csv_dir, transform=None, data_root="E:/Jzee4Study/dataset"):
        self.data_root = data_root
        img_label_pairs = pd.read_csv(csv_dir)
        self.imgs = img_label_pairs["filename"].values
        self.labels = img_label_pairs["label"].values
        self.transform = transform
        label_set = sorted(set(img_label_pairs["label"].drop_duplicates().values))
        self.label2idx = {}
        self.idx2label = {}
        for v, k in enumerate(label_set):
            self.label2idx[k] = v
            self.idx2label[v] = k

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.data_root, "mini-imagenet", "images", self.imgs[item][:9], self.imgs[item])).convert("RGB")
        label = self.label2idx[self.labels[item]]
        return self.transform(img), torch.tensor(label)

    def __len__(self):
        return len(self.imgs)


def create_dataloader(dataset, batch_size, use_val=True, transform_dict=None, resize=None, data_root="E:/Jzee4Study/dataset"):
    if dataset == "TinyImageNet":
        if transform_dict is not None:
            transform_train, transform_test = transform_dict["train"], transform_dict["test"]
        else:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

        train_dataset = TinyImageNet("train", transform_train, data_root=data_root)
        testset = TinyImageNet("val", transform_test, data_root=data_root)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        valloader = None
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    if dataset == "CIFAR10":
        if transform_dict is not None:
            transform_train, transform_test = transform_dict["train"], transform_dict["test"]
        else:
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                ]
            )
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                ]
            )

        train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(data_root, "CIFAR-10"), train=True, download=True, transform=transform_train)

        if use_val:
            valid_dataset = torchvision.datasets.CIFAR10(root=os.path.join(data_root, "CIFAR-10"), train=True, download=True, transform=transform_test)
            train_idx = np.loadtxt(os.path.join(data_root, "train_idx.txt"), dtype=int)
            valid_idx = np.loadtxt(os.path.join(data_root, "val_idx.txt"), dtype=int)
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            trainloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=True,
                num_workers=8,
                worker_init_fn=seed_worker,
                generator=g,
            )
            valloader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=batch_size,
                sampler=valid_sampler,
                num_workers=8,
                worker_init_fn=seed_worker,
                generator=g,
            )
        else:
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
            valloader = None

        testset = torchvision.datasets.CIFAR10(root=os.path.join(data_root, "CIFAR-10"), train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    if dataset == "CIFAR100":
        if transform_dict is not None:
            transform_train, transform_test = transform_dict["train"], transform_dict["test"]
        else:
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                ]
            )
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(resize),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                ]
            )

        train_dataset = torchvision.datasets.CIFAR100(root=os.path.join(data_root, "CIFAR-100"), train=True, download=True, transform=transform_train)

        if use_val:
            valid_dataset = torchvision.datasets.CIFAR100(root=os.path.join(data_root, "CIFAR-100"), train=True, download=True, transform=transform_test)
            indices = list(range(50000))
            np.random.shuffle(indices)
            train_idx, valid_idx = indices[:45000], indices[45000:]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            trainloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=True,
                num_workers=8,
                worker_init_fn=seed_worker,
                generator=g,
            )
            valloader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=batch_size,
                sampler=valid_sampler,
                num_workers=8,
                worker_init_fn=seed_worker,
                generator=g,
            )
        else:
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
            valloader = None

        testset = torchvision.datasets.CIFAR100(root=os.path.join(data_root, "CIFAR-100"), train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, valloader, testloader
