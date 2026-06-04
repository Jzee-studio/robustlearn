import torch


class NormalizeWrapper(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


def pgd_attack(model, images, labels, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True, clamp=(0.0, 1.0)):
    """Untargeted PGD attack used for CLAT training and criticality estimation."""
    device = images.device
    ori_images = images.detach()
    if random_start:
        adv_images = ori_images + torch.empty_like(ori_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, clamp[0], clamp[1]).detach()
    else:
        adv_images = ori_images.clone().detach()

    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        adv_images = torch.clamp(ori_images + delta, clamp[0], clamp[1]).detach()

    return adv_images.to(device)
