import torchvision
from torch.utils.data import DataLoader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sample_transforms():
    return torchvision.transforms.Compose([
               torchvision.transforms.Resize(224),
               torchvision.transforms.ToTensor()
            ])
