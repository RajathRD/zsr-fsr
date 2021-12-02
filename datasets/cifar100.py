from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision


def cifar100Original(data_dir, train_transforms, transforms):
    train_data = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=T.Compose(train_transforms+transforms))

    test_data = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=T.Compose(transforms))

    return train_data, test_data