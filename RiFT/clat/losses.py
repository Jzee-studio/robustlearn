import torch
import torch.nn as nn


_ce = nn.CrossEntropyLoss()


def classification_loss(logits, targets):
    return _ce(logits, targets)


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


def feature_weakness_loss(clean_features, adv_features, selected_layers):
    loss = 0.0
    for layer_name in selected_layers:
        clean_key = _resolve_feature_key(clean_features, layer_name)
        adv_key = _resolve_feature_key(adv_features, layer_name)
        clean_feat = clean_features[clean_key]
        adv_feat = adv_features[adv_key]
        diff = adv_feat - clean_feat
        loss = loss + torch.norm(diff.flatten(start_dim=1), p=2, dim=1).mean() / max(clean_feat.flatten(start_dim=1).shape[1], 1)
    return loss


def clat_total_loss(logits, targets, clean_features, adv_features, selected_layers, lambda_feat=1.0):
    cls = classification_loss(logits, targets)
    feat = feature_weakness_loss(clean_features, adv_features, selected_layers)
    return cls + lambda_feat * feat, cls, feat
