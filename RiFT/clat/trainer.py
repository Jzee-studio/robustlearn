import os
from copy import deepcopy

import torch
from tqdm import tqdm

from dataloader import create_dataloader
from optimizer import create_optimizer, create_scheduler
from utils import create_logger, evaluate, evaluate_cifar_robustness, evaluate_tiny_robustness, interpolation

from .attack import pgd_attack
from .criticality import estimate_criticality
from .feature_extractor import FeatureExtractor
from .losses import clat_total_loss
from .selection import select_critical_layers, freeze_noncritical_params, get_trainable_parameters


def _get_norm_layer(dataset):
    if "CIFAR" in dataset:
        return torch.nn.Sequential()
    return torch.nn.Sequential()


def _get_eval_robustness(dataset):
    return evaluate_cifar_robustness if "CIFAR" in dataset else evaluate_tiny_robustness


def _default_target_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and name != "":
            layers.append(name.replace("module.", ""))
    return layers


class CLATTrainer:
    def __init__(self, args, model, trainloader, testloader):
        self.args = args
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = torch.nn.CrossEntropyLoss()
        self.eval_robustness = _get_eval_robustness(args.dataset)
        self.model_save_dir = os.path.join(
            "./results",
            f"{args.model}_{args.dataset}",
            "clat",
            f"{args.optim}_lr={args.lr}_wd={args.wd}_epochs={args.epochs}",
        )
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.logger = create_logger(os.path.join(self.model_save_dir, "output.log"))
        self.args.logger = self.logger
        self.norm_layer = _get_norm_layer(args.dataset)
        self.target_layers = _default_target_layers(model)
        self.selected_layers = []
        self.feature_layer_names = []
        self.init_sd = deepcopy(model.state_dict())
        torch.save(self.init_sd, os.path.join(self.model_save_dir, "init_params.pth"))
        self.best_acc = 0.0

    def _rebuild_optimizer(self):
        trainable_params = get_trainable_parameters(self.model)
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters selected for CLAT.")
        if self.args.optim == "SGDM":
            optimizer = torch.optim.SGD(trainable_params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd)
        else:
            optimizer = create_optimizer(self.args.optim, self.model, self.args.lr, self.args.momentum, weight_decay=self.args.wd)
        scheduler = create_scheduler(self.args, optimizer, lr_decays=[int(self.args.epochs // 2)])
        return optimizer, scheduler

    def _extract_layer_names(self):
        return self.target_layers

    def _criticality_pass(self, batch_size=10):
        loader_iter = iter(self.trainloader)
        images, labels = next(loader_iter)
        images, labels = images.to(self.args.device), labels.to(self.args.device)
        if images.shape[0] > batch_size:
            images = images[:batch_size]
            labels = labels[:batch_size]

        normed_model = torch.nn.Sequential(self.norm_layer, self.model).to(self.args.device)
        target_layers = self._extract_layer_names()
        _, criticality_dict, ranked = estimate_criticality(
            normed_model,
            images,
            labels,
            target_layers=target_layers,
            eps=self.args.attack_eps,
            alpha=self.args.attack_alpha,
            steps=self.args.attack_steps,
            random_start=self.args.random_start,
        )
        return criticality_dict, ranked

    def _forward_features(self, model, images, layer_names):
        extractor = FeatureExtractor(model, layer_names).register_hooks()
        extractor.clear()
        _ = model(images)
        feats = extractor.get_features().copy()
        extractor.remove_hooks()
        return feats

    def _train_batch(self, optimizer, images, labels, selected_layers):
        normed_model = torch.nn.Sequential(self.norm_layer, self.model).to(self.args.device)
        adv_images = pgd_attack(normed_model, images, labels, eps=self.args.attack_eps, alpha=self.args.attack_alpha, steps=self.args.attack_steps, random_start=self.args.random_start)

        clean_features = self._forward_features(normed_model, images, selected_layers)
        adv_features = self._forward_features(normed_model, adv_images, selected_layers)

        optimizer.zero_grad()
        outputs = normed_model(adv_images)
        total_loss, cls_loss, feat_loss = clat_total_loss(outputs, labels, clean_features, adv_features, selected_layers, lambda_feat=self.args.lambda_feat)
        total_loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        acc = predicted.eq(labels).float().mean().item() * 100
        return total_loss.item(), cls_loss.item(), feat_loss.item(), acc

    def _validate(self):
        _, train_acc = evaluate(self.args, self.model, self.trainloader, self.criterion)
        _, test_acc = evaluate(self.args, self.model, self.testloader, self.criterion)
        robust_acc = self.eval_robustness(self.args, self.model)
        return train_acc, test_acc, robust_acc

    def run(self):
        self.logger.info(self.args)
        self.logger.info("==> Building CLAT context...")

        criticality_dict, ranked = self._criticality_pass()
        self.selected_layers = select_critical_layers(ranked, num_layers=self.args.num_critical_layers, mode=self.args.selection_mode)
        self.feature_layer_names = list(self.target_layers)
        freeze_noncritical_params(self.model, self.selected_layers)
        optimizer, scheduler = self._rebuild_optimizer()

        self.logger.info(f"Initial selected critical layers: {self.selected_layers}")

        start_epoch = 0
        for epoch in range(start_epoch, self.args.epochs):
            if self.args.dynamic_layers and (epoch % self.args.reselect_interval == 0):
                criticality_dict, ranked = self._criticality_pass()
                self.selected_layers = select_critical_layers(ranked, num_layers=self.args.num_critical_layers, mode=self.args.selection_mode)
                freeze_noncritical_params(self.model, self.selected_layers)
                optimizer, scheduler = self._rebuild_optimizer()
                self.logger.info(f"Epoch {epoch}: reselected layers -> {self.selected_layers}")

            self.model.train()
            train_loss = 0.0
            train_cls_loss = 0.0
            train_feat_loss = 0.0
            total = 0
            correct = 0

            for images, labels in tqdm(self.trainloader, desc=f"CLAT epoch {epoch}"):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                total_loss, cls_loss, feat_loss, acc = self._train_batch(optimizer, images, labels, self.selected_layers)
                train_loss += total_loss * labels.size(0)
                train_cls_loss += cls_loss * labels.size(0)
                train_feat_loss += feat_loss * labels.size(0)
                total += labels.size(0)
                correct += acc * labels.size(0) / 100.0

            train_loss /= total
            train_cls_loss /= total
            train_feat_loss /= total
            train_acc = correct / total * 100.0

            test_loss, test_acc = evaluate(self.args, self.model, self.testloader, self.criterion)
            robust_acc = self.eval_robustness(self.args, self.model)
            self.logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, cls_loss={train_cls_loss:.4f}, feat_loss={train_feat_loss:.4f}, train_acc={train_acc:.2f}, test_acc={test_acc:.2f}, robust_acc={robust_acc:.2f}"
            )

            state = {
                "model": self.model.state_dict(),
                "acc": test_acc,
                "robust_acc": robust_acc,
                "epoch": epoch,
                "selected_layers": self.selected_layers,
            }
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                torch.save(state, os.path.join(self.model_save_dir, "best_params.pth"))
                self.logger.info("==> Saving best params...")

            scheduler.step()

        checkpoint = torch.load(os.path.join(self.model_save_dir, "best_params.pth"), map_location=self.args.device)
        self.model.load_state_dict(checkpoint["model"])
        _, test_acc = evaluate(self.args, self.model, self.testloader, self.criterion)
        robust_acc = self.eval_robustness(self.args, self.model)
        self.logger.info(f"==> Finetune test acc: {test_acc:.2f}%, robust acc: {robust_acc:.2f}")
        self.logger.info(interpolation(self.args, self.logger, self.init_sd, deepcopy(self.model.state_dict()), self.model, self.testloader, self.criterion, self.model_save_dir, self.eval_robustness))
        return self.model
