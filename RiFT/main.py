# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy
from tqdm import tqdm

from utils import *
from dataloader import *
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


def _build_eval_model(args, model):
    if "CIFAR" in args.dataset:
        norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    else:
        norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return nn.Sequential(norm_layer, model).to(args.device)


def _normalize_layer_name(layer_name):
    if layer_name.startswith("1.module."):
        return layer_name[len("1.module."):]
    if layer_name.startswith("module."):
        return layer_name[len("module."):]
    if layer_name.startswith("1."):
        return layer_name[len("1."):]
    return layer_name


def _matching_param_name(layer_name):
    normalized = _normalize_layer_name(layer_name)
    return [
        f"{normalized}.weight",
        f"1.{normalized}.weight",
        f"module.{normalized}.weight",
        f"1.module.{normalized}.weight",
    ]


def _collect_target_layers(model):
    layer_names = []
    for name, module in model.named_modules():
        normalized_name = _normalize_layer_name(name)
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if "sub" in normalized_name:
                continue
            if normalized_name == "" or normalized_name == "module":
                continue
            layer_names.append(normalized_name)
    return sorted(set(layer_names))


def _evaluate_mrc_group(args, model, trainloader, criterion, layer_names, epsilon=0.1):
    cloned_model = deepcopy(model)
    normalized_layer_names = {_normalize_layer_name(layer) for layer in layer_names}
    for name, param in cloned_model.named_parameters():
        param.requires_grad = False
        if any(name == candidate for layer in normalized_layer_names for candidate in _matching_param_name(layer)):
            param.requires_grad = True

    trainable_params = [name for name, param in cloned_model.named_parameters() if param.requires_grad]
    if not trainable_params:
        return None

    init_params = {name: param.detach().clone() for name, param in cloned_model.named_parameters() if param.requires_grad}
    optimizer = torch.optim.SGD(cloned_model.parameters(), lr=1)

    max_loss = 0.0
    min_acc = 0.0

    for _ in range(10):
        for inputs, targets in trainloader:
            optimizer.zero_grad()
            outputs = cloned_model(inputs)
            loss = -1 * criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        state_dict = cloned_model.state_dict()
        for layer_name in normalized_layer_names:
            for candidate in _matching_param_name(layer_name):
                if candidate in state_dict and candidate in init_params:
                    diff = state_dict[candidate] - init_params[candidate]
                    times = torch.linalg.norm(diff) / torch.linalg.norm(init_params[candidate])
                    if times > epsilon:
                        diff = diff / times * epsilon
                        state_dict[candidate] = deepcopy(init_params[candidate] + diff)
                    break
        cloned_model.load_state_dict(state_dict)

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

    return max_loss, min_acc


def layer_sharpness(args, model, epsilon=0.1):
    model = _build_eval_model(args, model)
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

    args.logger.info("{:35}, Robust Loss: {:10.2f}, Robust Acc: {:10.2f}".format("Origin", origin_loss, origin_acc * 100))

    layer_sharpness_dict = {}
    target_layers = _collect_target_layers(model)

    if args.layer is None:
        for layer_name in target_layers:
            result = _evaluate_mrc_group(args, model, trainloader, criterion, [layer_name], epsilon=epsilon)
            if result is None:
                continue
            max_loss, min_acc = result
            layer_sharpness_dict[layer_name] = max_loss - origin_loss
            args.logger.info("{:35}, MRC: {:10.2f}, Dropped Robust Acc: {:10.2f}".format(layer_name, max_loss - origin_loss, (origin_acc - min_acc) * 100))

        sorted_layer_sharpness = sorted(layer_sharpness_dict.items(), key=lambda x: x[1])
        for (k, v) in sorted_layer_sharpness:
            args.logger.info("{:35}, Robust Loss: {:10.2f}".format(k, v))
        return sorted_layer_sharpness

    normalized_target_layers = {_normalize_layer_name(layer) for layer in target_layers}
    normalized_specified_layer = _normalize_layer_name(args.layer)

    if normalized_specified_layer not in normalized_target_layers:
        raise ValueError(f"Specified layer '{args.layer}' was not found among eligible Conv2d/Linear layers.")

    for other_layer in target_layers:
        if _normalize_layer_name(other_layer) == normalized_specified_layer:
            continue
        result = _evaluate_mrc_group(args, model, trainloader, criterion, [args.layer, other_layer], epsilon=epsilon)
        if result is None:
            continue
        max_loss, min_acc = result
        group_name = f"{args.layer} + {other_layer}"
        layer_sharpness_dict[group_name] = max_loss - origin_loss
        args.logger.info("{:35}, MRC: {:10.2f}, Dropped Robust Acc: {:10.2f}".format(group_name, max_loss - origin_loss, (origin_acc - min_acc) * 100))

    sorted_layer_sharpness = sorted(layer_sharpness_dict.items(), key=lambda x: x[1])
    for (k, v) in sorted_layer_sharpness:
        args.logger.info("{:35}, Robust Loss: {:10.2f}".format(k, v))

    return sorted_layer_sharpness


def train(args, model, dataloader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item() * targets.size(0)

        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss / total, correct / total * 100


def main():

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--model', default="ResNet18", type=str, help='model used')
    parser.add_argument('--dataset', default="CIFAR10", type=str, help="dataset", choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
    parser.add_argument('--num_classes', default=10, type=int, help='num classes')
    parser.add_argument('--input_size', default=32, type=int, help='input_size')
    parser.add_argument('--layer', default=None, type=str, help='Trainable layer')
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
    if "CIFAR" in args.dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

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

    if not args.cal_mrc:
        assert args.layer is not None

        for name, param in model.named_parameters():
            param.requires_grad = False
            if args.layer in name:
                param.requires_grad = True

    _, train_acc = evaluate(args, model, trainloader, criterion)
    _, test_acc = evaluate(args, model, testloader, criterion)
    test_robust_acc = evalulate_robustness(args, model)
    logger.info("==> Init train acc: {:.2f}%, test acc: {:.2f}%, robust acc: {:.2f}%".format(train_acc, test_acc, test_robust_acc))


    for epoch in range(start_epoch, start_epoch + args.epochs):

        logger.info("==> Epoch {}".format(epoch))
        logger.info("==> Training...")
        train_loss, train_acc = train(args, model, trainloader, optimizer, criterion)

        logger.info("==> Train loss: {:.2f}, train acc: {:.2f}%".format(train_loss, train_acc))

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






