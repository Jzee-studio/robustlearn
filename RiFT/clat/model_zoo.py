from model import create_model


_MODEL_NAME_MAP = {
    "RN18": "ResNet18",
    "RN34": "ResNet34",
    "RN50": "ResNet50",
    "DN121": "DenseNet",
    "VGG19": "VGG19",
    "WRN34-10": "WideResNet34",
    "WRN34_10": "WideResNet34",
    "WRN70-16": "WideResNet34",
    "PreActRN18": "ResNet18",
}


def build_model(model_name, input_size, num_classes, device, patch_size=4, resume=None):
    mapped = _MODEL_NAME_MAP.get(model_name, model_name)
    return create_model(mapped, input_size, num_classes, device, patch_size=patch_size, resume=resume)
