from collections import OrderedDict

import torch

from .attack import pgd_attack
from .feature_extractor import FeatureExtractor


def _resolve_feature_key(feature_dict, layer_name):
    if layer_name in feature_dict:
        return layer_name
    candidates = []
    for key in feature_dict.keys():
        if key.endswith(layer_name) or layer_name.endswith(key):
            candidates.append(key)
    if candidates:
        return max(candidates, key=len)
    raise KeyError(layer_name)


def compute_layer_weakness(clean_features, adv_features, layer_name):
    clean_key = _resolve_feature_key(clean_features, layer_name)
    adv_key = _resolve_feature_key(adv_features, layer_name)
    clean_feat = clean_features[clean_key]
    adv_feat = adv_features[adv_key]
    diff = adv_feat - clean_feat
    feature_dim = clean_feat.flatten(start_dim=1).shape[1]
    weakness = torch.norm(diff.flatten(start_dim=1), p=2, dim=1).mean() / max(feature_dim, 1)
    return weakness.item()


def compute_criticality_index(weakness_dict, ordered_layers):
    criticality = OrderedDict()
    prev_weakness = None
    for layer_name in ordered_layers:
        weakness = weakness_dict[layer_name]
        if prev_weakness is None:
            criticality[layer_name] = weakness
        else:
            denom = prev_weakness if prev_weakness != 0 else 1e-12
            criticality[layer_name] = weakness / denom
        prev_weakness = weakness
    return criticality


def estimate_criticality(model, images, labels, target_layers, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True):
    model.eval()
    extractor = FeatureExtractor(model, target_layers).register_hooks()

    with torch.no_grad():
        extractor.clear()
        _ = model(images)
        clean_features = extractor.get_features().copy()

    adv_images = pgd_attack(model, images, labels, eps=eps, alpha=alpha, steps=steps, random_start=random_start)

    with torch.no_grad():
        extractor.clear()
        _ = model(adv_images)
        adv_features = extractor.get_features().copy()

    extractor.remove_hooks()

    weakness_dict = OrderedDict()
    for layer_name in target_layers:
        weakness_dict[layer_name] = compute_layer_weakness(clean_features, adv_features, layer_name)

    criticality_dict = compute_criticality_index(weakness_dict, target_layers)
    ranked_layers = sorted(criticality_dict.items(), key=lambda kv: kv[1], reverse=True)
    return weakness_dict, criticality_dict, ranked_layers
