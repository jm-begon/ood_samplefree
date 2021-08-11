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
    "Imagenet": ImageNet,
    "cifar10-h1": CIFAR10h1,
    "cifar10-h2": CIFAR10h2,
    "cifar10-h3": CIFAR10h3,
    "cifar10-h4": CIFAR10h4,
}

__AUGMENTED_DATASETS__ = {
    "cifar10": CropHzFlipAugmented().partial(CIFAR10),
    "cifar100": CropHzFlipAugmented().partial(CIFAR100),
    "cifar10-h1": CropHzFlipAugmented().partial(CIFAR10h1),
    "cifar10-h2": CropHzFlipAugmented().partial(CIFAR10h2),
    "cifar10-h3": CropHzFlipAugmented().partial(CIFAR10h3),
    "cifar10-h4": CropHzFlipAugmented().partial(CIFAR10h4),

}

__all__ = [
    "__DATASETS__", "__AUGMENTED_DATASETS__", "get_transform",
]
