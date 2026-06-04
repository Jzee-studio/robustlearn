from collections import OrderedDict

import torch
import torch.nn as nn


class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = set(target_layers)
        self.features = OrderedDict()
        self.handles = []

    def _get_module(self, name):
        module = self.model
        if isinstance(module, torch.nn.DataParallel):
            module = module.module
        for part in name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _hook_fn(self, name):
        def hook(_module, _inputs, output):
            self.features[name] = output
        return hook

    def register_hooks(self):
        self.remove_hooks()
        for name, module in self.model.named_modules():
            matched = None
            if name in self.target_layers:
                matched = name
            else:
                for target in self.target_layers:
                    if name.endswith(target) or target.endswith(name):
                        matched = name
                        break
            if matched is not None:
                self.handles.append(module.register_forward_hook(self._hook_fn(matched)))
        return self

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def clear(self):
        self.features = OrderedDict()

    def get_features(self):
        return self.features
