import random


_EXCLUDED_TOKENS = ("conv1", "linear", "shortcut", "downsample", "sub")


def is_middle_layer(name):
    name = name.lower()
    if any(token in name for token in _EXCLUDED_TOKENS):
        return False
    return True


def filter_middle_layers(layer_names):
    return [name for name in layer_names if is_middle_layer(name)]


def select_critical_layers(ranked_layers, num_layers=1, mode="topk"):
    layers = filter_middle_layers([name for name, _score in ranked_layers])
    if not layers:
        layers = [name for name, _score in ranked_layers]
    if mode in ("topk", "largest"):
        return layers[:num_layers]
    if mode == "smallest":
        return layers[-num_layers:]
    if mode == "random":
        return random.sample(layers, k=min(num_layers, len(layers)))
    raise ValueError(f"Unsupported selection mode: {mode}")


def freeze_noncritical_params(model, selected_layers):
    for name, param in model.named_parameters():
        param.requires_grad = any(layer in name or name in layer for layer in selected_layers)
    return model


def get_trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
