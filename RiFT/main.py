# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy
from tqdm import tqdm

from utils import *
from dataloader import *
from dataloader import build_train_transform
from model import create_model
from optimizer import *

from robustbench.utils import load_model


def generate_adv_dataset(args, model):
    adv_train_dataset = adv_dataset()

    model = model.eval()
    atk_model = torchattacks.PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
    transform_test = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    if args.dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=os.path.join(args.data_root, 'CIFAR-10'), train=False, download=True, transform=transform_test)
    elif args.dataset == "CIFAR100":
        train_dataset = datasets.CIFAR100(root=os.path.join(args.data_root, 'CIFAR-100'), train=False, download=True, transform=transform_test)
    else:
        train_dataset = TinyImageNet("train", transform_test, data_root=args.data_root)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)

    for images, labels in trainloader:

        images = images.to(args.device)
        labels = labels.to(args.device)
        
        adv_images = atk_model(images, labels)  
        adv_train_dataset.append_data(adv_images, labels)

    return adv_train_dataset


def layer_sharpness(args, model, epsilon=0.1):
    
    if "CIFAR" in args.dataset:
        norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    else:
        norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model = nn.Sequential(norm_layer, model).to(args.device)
    
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(generate_adv_dataset(args, deepcopy(model)), batch_size=512, shuffle=True, num_workers=0)
    origin_total = 0
    origin_loss = 0.0
    origin_acc = 0
    with torch.no_grad():
        model.eval()
        

        for inputs, targets in trainloader:
            outputs = model(inputs)
            origin_total += targets.shape[0]
            origin_loss += criterion(outputs, targets).item() * targets.shape[0]
            _, predicted = outputs.max(1)
            origin_acc += predicted.eq(targets).sum().item()        
        
        origin_acc /= origin_total
        origin_loss /= origin_total

    args.logger.info("{:35}, Robust Loss: {:10.2f}, Robust Acc: {:10.2f}".format("Origin", origin_loss, origin_acc*100))

    model.eval()
    layer_sharpness_dict = {} 
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # print(name)
            # For WideResNet
            if "sub" in name:
                continue
            layer_sharpness_dict[name] = 1e10

    for layer_name, _ in model.named_parameters():
        if "weight" in layer_name and layer_name[:-len(".weight")] in layer_sharpness_dict.keys():
            # print(layer_name)
            cloned_model = deepcopy(model)
            # set requires_grad sign for each layer
            for name, param in cloned_model.named_parameters():
                # print(name)
                if name == layer_name:
                    # print(name)
                    param.requires_grad = True
                    init_param = param.detach().clone()
                else:
                    param.requires_grad = False
        
            optimizer = torch.optim.SGD(cloned_model.parameters(), lr=1)

            max_loss = 0.0
            min_acc = 0
    
            for epoch in range(10):
                # Gradient ascent
                for inputs, targets in trainloader:
                    optimizer.zero_grad()
                    outputs = cloned_model(inputs)
                    loss = -1 * criterion(outputs, targets) 
                    loss.backward()
                    optimizer.step()
                sd = cloned_model.state_dict()
                diff = sd[layer_name] - init_param
                times = torch.linalg.norm(diff)/torch.linalg.norm(init_param)
                # print(times)
                if times > epsilon:
                    diff = diff / times * epsilon
                    sd[layer_name] = deepcopy(init_param + diff)
                    cloned_model.load_state_dict(sd)

                with torch.no_grad():
                    total = 0
                    total_loss = 0.0
                    correct = 0
                    for inputs, targets in trainloader:
                        outputs = cloned_model(inputs)
                        total += targets.shape[0]
                        total_loss += criterion(outputs, targets).item() * targets.shape[0]
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(targets).sum().item()  
                    
                    total_loss /= total
                    correct /= total

                if total_loss > max_loss:
                    max_loss = total_loss
                    min_acc = correct
            
            layer_sharpness_dict[layer_name[:-len(".weight")]] = max_loss - origin_loss
            args.logger.info("{:35}, MRC: {:10.2f}, Dropped Robust Acc: {:10.2f}".format(layer_name[:-len(".weight")], max_loss-origin_loss, (origin_acc-min_acc)*100))

    sorted_layer_sharpness = sorted(layer_sharpness_dict.items(), key=lambda x:x[1])
    for (k, v) in sorted_layer_sharpness:
        args.logger.info("{:35}, Robust Loss: {:10.2f}".format(k, v))
    
    return sorted_layer_sharpness


def _unwrap_state_dict(checkpoint):
    if "net" in checkpoint:
        return checkpoint["net"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _strip_module_prefix(state_dict):
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            stripped[k[len("module."):]] = v
        else:
            stripped[k] = v
    return stripped


def _add_module_prefix(state_dict):
    prefixed = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            prefixed[k] = v
        else:
            prefixed[f"module.{k}"] = v
    return prefixed


def _load_weights_into_model(model, path, device, strict=True):
    checkpoint = torch.load(path, map_location=device)
    state_dict = _strip_module_prefix(_unwrap_state_dict(checkpoint))

    # Make checkpoint/model naming consistent for DataParallel vs non-DataParallel models.
    model_state_keys = next(iter(model.state_dict().keys()))
    model_expects_module_prefix = model_state_keys.startswith("module.")
    checkpoint_has_module_prefix = next(iter(state_dict.keys())).startswith("module.") if len(state_dict) > 0 else False

    if model_expects_module_prefix and not checkpoint_has_module_prefix:
        state_dict = _add_module_prefix(state_dict)
    elif not model_expects_module_prefix and checkpoint_has_module_prefix:
        state_dict = _strip_module_prefix(state_dict)

    incompatible = model.load_state_dict(state_dict, strict=strict)
    return model, incompatible.missing_keys, incompatible.unexpected_keys


def _get_base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _get_target_param_names(model, layer_prefix):
    base_model = _get_base_model(model)
    return [name for name, _ in base_model.named_parameters() if name.startswith(layer_prefix)]


def _build_teacher_target_map(model, teacher_model, target_param_names):
    student_base = _get_base_model(model)
    teacher_base = _get_base_model(teacher_model)

    student_params = dict(student_base.named_parameters())
    teacher_params = dict(teacher_base.named_parameters())

    target_map = {}
    for name in target_param_names:
        if name not in student_params:
            raise KeyError(f"Student model is missing parameter: {name}")
        if name not in teacher_params:
            raise KeyError(f"Teacher model is missing parameter: {name}")
        if student_params[name].shape != teacher_params[name].shape:
            raise ValueError(
                f"Shape mismatch for parameter {name}: "
                f"student={tuple(student_params[name].shape)}, teacher={tuple(teacher_params[name].shape)}"
            )
        target_map[name] = teacher_params[name].detach().clone()
    return target_map


def _resolve_module_by_prefix(model, layer_prefix):
    base_model = _get_base_model(model)
    modules = dict(base_model.named_modules())
    candidates = [layer_prefix]
    if layer_prefix.endswith(".weight") or layer_prefix.endswith(".bias"):
        candidates.append(layer_prefix.rsplit(".", 1)[0])

    for candidate in candidates:
        if candidate in modules:
            return candidate, modules[candidate]

    available = [name for name in modules.keys() if name.startswith(layer_prefix)]
    if available:
        available.sort(key=len)
        return available[0], modules[available[0]]

    raise KeyError(f"Cannot resolve module for layer prefix: {layer_prefix}")


def _relative_weight_penalty(student_model, teacher_target_map, eps=1e-12):
    student_base = _get_base_model(student_model)
    student_params = dict(student_base.named_parameters())

    penalty = torch.tensor(0.0, device=next(student_base.parameters()).device)
    for name, teacher_param in teacher_target_map.items():
        student_param = student_params[name]
        diff_sq = torch.sum((student_param - teacher_param) ** 2)
        denom = torch.sum(teacher_param.detach() ** 2) + eps
        penalty = penalty + diff_sq / denom
    return penalty


def _feature_penalty(student_feature, teacher_feature, eps=1e-12):
    diff_sq = torch.sum((student_feature - teacher_feature) ** 2)
    denom = torch.sum(teacher_feature.detach() ** 2) + eps
    return diff_sq / denom


def _get_module_output_cache(module):
    cache = {"output": None}

    def hook(_module, _inputs, output):
        cache["output"] = output

    handle = module.register_forward_hook(hook)
    return cache, handle


def train(args, model, teacher_model, teacher_target_map, dataloader, optimizer, criterion, teacher_weight_lambda, teacher_feature_lambda, student_feature_module=None, teacher_feature_module=None):
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    train_loss = 0
    ce_loss_sum = 0
    reg_weight_sum = 0
    reg_feature_sum = 0
    correct = 0
    total = 0

    student_cache, student_handle = (None, None)
    teacher_cache, teacher_handle = (None, None)
    if student_feature_module is not None and teacher_feature_module is not None and teacher_feature_lambda > 0:
        student_cache, student_handle = _get_module_output_cache(student_feature_module)
        teacher_cache, teacher_handle = _get_module_output_cache(teacher_feature_module)

    try:
        for i, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()

            if student_handle is not None and teacher_handle is not None:
                teacher_cache["output"] = None
                student_cache["output"] = None
                with torch.no_grad():
                    _ = teacher_model(inputs)
                outputs = model(inputs)
            else:
                outputs = model(inputs)

            ce_loss = criterion(outputs, targets)

            weight_reg = torch.tensor(0.0, device=inputs.device)
            feature_reg = torch.tensor(0.0, device=inputs.device)
            if teacher_target_map is not None and teacher_weight_lambda > 0:
                weight_reg = _relative_weight_penalty(model, teacher_target_map)
            if (
                teacher_feature_lambda > 0
                and student_cache is not None
                and teacher_cache is not None
                and student_cache["output"] is not None
                and teacher_cache["output"] is not None
            ):
                student_feat = student_cache["output"]
                teacher_feat = teacher_cache["output"].detach()
                if isinstance(student_feat, (tuple, list)):
                    student_feat = student_feat[0]
                if isinstance(teacher_feat, (tuple, list)):
                    teacher_feat = teacher_feat[0]
                feature_reg = _feature_penalty(student_feat, teacher_feat)

            loss = ce_loss + teacher_weight_lambda * weight_reg + teacher_feature_lambda * feature_reg
            train_loss += loss.item() * targets.size(0)
            ce_loss_sum += ce_loss.item() * targets.size(0)
            reg_weight_sum += weight_reg.item() * targets.size(0)
            reg_feature_sum += feature_reg.item() * targets.size(0)

            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    finally:
        if student_handle is not None:
            student_handle.remove()
        if teacher_handle is not None:
            teacher_handle.remove()

    return train_loss / total, ce_loss_sum / total, reg_weight_sum / total, reg_feature_sum / total, correct / total * 100


def main():

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--model', default="ResNet18", type=str, help='model used')
    parser.add_argument('--dataset', default="CIFAR10", type=str, help="dataset", choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
    parser.add_argument('--num_classes', default=10, type=int, help='num classes')
    parser.add_argument('--input_size', default=32, type=int, help='input_size')
    parser.add_argument('--layer', default=None, type=str, help='Trainable layer')
    parser.add_argument('--teacher', default=None, type=str, help='teacher checkpoint for layer guidance')
    parser.add_argument('--teacher_weight_lambda', default=1.0, type=float, help='weight of teacher relative weight regularization')
    parser.add_argument('--teacher_feature_lambda', default=1.0, type=float, help='weight of teacher feature regularization')
    parser.add_argument('--teacher_init_weight_lambda', default=0.0, type=float, help='initial teacher weight regularization')
    parser.add_argument('--teacher_init_feature_lambda', default=0.0, type=float, help='initial teacher feature regularization')
    parser.add_argument('--teacher_warmup_epochs', default=2, type=int, help='epochs to warm up teacher regularization')
    parser.add_argument('--teacher_feature_layer', default=None, type=str, help='module prefix for feature guidance; defaults to target layer prefix')
    parser.add_argument('--use_augmix', action='store_true', help='use AugMix augmentation during finetuning')
    parser.add_argument('--augmix_width', default=3, type=int, help='AugMix mixture width')
    parser.add_argument('--augmix_depth', default=-1, type=int, help='AugMix chain depth; -1 samples randomly')
    parser.add_argument('--augmix_severity', default=3, type=int, help='AugMix severity')
    parser.add_argument("--cal_mrc", action="store_true", help='If to calculate Module Robust Criticality (MRC) value of each module.')
    
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--patch', default=4, type=int, help='num patch (used by vit)')
    parser.add_argument('--optim', default="SGDM", type=str, help="optimizer")

    parser.add_argument('--device', default="cuda", type=str, help='device')
    parser.add_argument('--data_root', default="E:/Jzee4Study/dataset", type=str, help='dataset root directory')
    
    parser.add_argument('--lr_scheduler', default="step", choices=["step", 'cosine'])

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGDM')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=float, help='lr_decay_gamma')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--epochs', default=10, type=int, help='num of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    proj_name = "rift"
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    suffix = '{}_{}_lr={}_wd={}_epochs={}_{}'.format(proj_name, args.optim, args.lr, args.wd, args.epochs, args.layer)
    model_save_dir = './results/{}_{}/checkpoint/'.format(args.model, args.dataset) + suffix + "/"

    for path in [model_save_dir]:
        if not os.path.isdir(path):
            os.makedirs(path)

    logger = create_logger(model_save_dir+'output.log')
    logger.info(args)

    args.logger = logger

    # create dataloader
    logger.info('==> Preparing data and create dataloaders...')
    transform_train = build_train_transform(
        args.dataset,
        use_augmix=args.use_augmix,
        augmix_width=args.augmix_width,
        augmix_depth=args.augmix_depth,
        augmix_severity=args.augmix_severity,
    )

    if "CIFAR" in args.dataset:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    transform_dict = {"train": transform_train, "test": transform_test}

    trainloader, _, testloader = create_dataloader(args.dataset, args.batch_size, use_val=False, transform_dict=transform_dict, data_root=args.data_root)

    logger.info('==> Building dataloaders...')
    logger.info(args.dataset)

    # create model
    logger.info('==> Building model...')
    model = create_model(args.model, args.input_size, args.num_classes, args.device, args.patch, args.resume)
    logger.info(args.model)

    teacher_model = None
    teacher_target_map = None
    if args.teacher is not None:
        logger.info(f"==> Loading teacher model from {args.teacher}")
        teacher_model = create_model(args.model, args.input_size, args.num_classes, args.device, args.patch, None)
        teacher_model, missing_keys, unexpected_keys = _load_weights_into_model(teacher_model, args.teacher, args.device)
        teacher_model = teacher_model.to(args.device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        if missing_keys or unexpected_keys:
            raise ValueError(f"Teacher checkpoint mismatch. missing={missing_keys}, unexpected={unexpected_keys}")

    logger.info('==> Building optimizer and learning rate scheduler...')
    optimizer = create_optimizer(args.optim, model, args.lr, args.momentum, weight_decay=args.wd)
    logger.info(optimizer)
    lr_decays = [int(args.epochs // 2)]
    scheduler = create_scheduler(args, optimizer, lr_decays=lr_decays)
    logger.info(scheduler)

    criterion = nn.CrossEntropyLoss()

    init_sd = deepcopy(model.state_dict())
    torch.save(init_sd, model_save_dir + "init_params.pth")

    if "CIFAR" in args.dataset:
        evalulate_robustness = evaluate_cifar_robustness
    else:
        evalulate_robustness = evaluate_tiny_robustness


    if args.cal_mrc:    
        layer_sharpness(args, deepcopy(model), epsilon=0.1)
        exit()

    assert args.layer is not None

    target_param_names = _get_target_param_names(model, args.layer)
    if len(target_param_names) == 0:
        raise ValueError(f"No parameters matched layer prefix: {args.layer}")
    logger.info(f"==> Trainable parameters: {target_param_names}")

    base_model = _get_base_model(model)
    for name, param in base_model.named_parameters():
        param.requires_grad = name in target_param_names

    student_feature_module = None
    teacher_feature_module = None
    if teacher_model is not None:
        teacher_target_map = _build_teacher_target_map(model, teacher_model, target_param_names)
        logger.info(f"==> Teacher guidance enabled for {len(teacher_target_map)} parameters")

        feature_layer_name = args.teacher_feature_layer or args.layer
        student_feature_layer_name, student_feature_module = _resolve_module_by_prefix(model, feature_layer_name)
        teacher_feature_layer_name, teacher_feature_module = _resolve_module_by_prefix(teacher_model, feature_layer_name)
        logger.info(f"==> Feature guidance layer: student={student_feature_layer_name}, teacher={teacher_feature_layer_name}")

    _, train_acc = evaluate(args, model, trainloader, criterion)
    _, test_acc = evaluate(args, model, testloader, criterion)
    test_robust_acc = evalulate_robustness(args, model)
    logger.info("==> Init train acc: {:.2f}%, test acc: {:.2f}%, robust acc: {:.2f}%".format(train_acc, test_acc, test_robust_acc))


    for epoch in range(start_epoch, start_epoch + args.epochs):

        logger.info("==> Epoch {}".format(epoch))
        logger.info("==> Training...")
        if args.teacher_warmup_epochs > 0:
            weight_lambda = args.teacher_init_weight_lambda + (args.teacher_weight_lambda - args.teacher_init_weight_lambda) * min((epoch + 1) / args.teacher_warmup_epochs, 1.0)
            feature_lambda = args.teacher_init_feature_lambda + (args.teacher_feature_lambda - args.teacher_init_feature_lambda) * min((epoch + 1) / args.teacher_warmup_epochs, 1.0)
        else:
            weight_lambda = args.teacher_weight_lambda
            feature_lambda = args.teacher_feature_lambda
        train_loss, ce_loss, reg_weight, reg_feature, train_acc = train(
            args,
            model,
            teacher_model,
            teacher_target_map,
            trainloader,
            optimizer,
            criterion,
            weight_lambda,
            feature_lambda,
            student_feature_module,
            teacher_feature_module,
        )

        logger.info("==> Train loss: {:.6f}, ce loss: {:.6f}, reg weight: {:.6f}, reg feature: {:.6f}, train acc: {:.2f}%".format(train_loss, ce_loss, reg_weight, reg_feature, train_acc))

        logger.info("==> Testing...")
        test_loss, test_acc = evaluate(args, model, testloader, criterion)

        logger.info("==> Test loss: {:.2f}, test acc: {:.2f}%".format(test_loss, test_acc))

        state = {
            'model': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if test_acc > best_acc:
            best_acc = test_acc
            params = "best_params.pth"
            logger.info('==> Saving best params...')
            torch.save(state, model_save_dir + params)
        else:
            if epoch % 2 == 0:
                params = "epoch{}_params.pth".format(epoch)
                logger.info('==> Saving checkpoints...')
                torch.save(state, model_save_dir + params)

        scheduler.step()

    checkpoint = torch.load(model_save_dir + "best_params.pth")
    model.load_state_dict(checkpoint["model"])

    test_loss, test_acc = evaluate(args, model, testloader, criterion)
    
    test_robust_acc = evalulate_robustness(args, model)

    logger.info("==> Finetune test acc: {:.2f}%, robust acc: {:.2f}".format(test_acc, test_robust_acc))

    logger.info(interpolation(args, logger, init_sd, deepcopy(model.state_dict()), model, testloader, criterion, model_save_dir, evalulate_robustness))


if __name__ == "__main__":
    main()






