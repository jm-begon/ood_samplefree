from .datasets import CIFAR10, CIFAR100, SVHN, MNIST, FashionMNIST, STL10, \
    Uniform, Gaussian, TinyImageNet, LSUNTestSet, ImageNet
from .custom_transforms import CropAugmented, CropHzFlipAugmented, \
    FlipsAugmented, CropHzVFlipAugmented
from .utils import get_transform

__DATASETS__ = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "svhn": SVHN,
    "mnist": MNIST,
    "fashionMnist": FashionMNIST,
    "stl10": STL10,
    "uniform": Uniform,
    "gaussian": Gaussian,
    "TinyImageNet": TinyImageNet,
    "LSUNTestSet": LSUNTestSet,
    "Imagenet": ImageNet
}

__AUGMENTED_DATASETS__ = {
    "cifar10": CropHzFlipAugmented().partial(CIFAR10),
    "cifar100": CropHzFlipAugmented().partial(CIFAR100),

}

__all__ = [
    "__DATASETS__", "__AUGMENTED_DATASETS__", "get_transform",
]
