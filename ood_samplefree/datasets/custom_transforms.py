from copy import copy
from functools import partial
import torch
from torchvision import transforms

class Shuffle(object):
    def __call__(self, tensor):
        # TODO support gray images
        n_channel, height, width = tensor.size()
        perm = torch.randperm(height*width)
        for c in range(n_channel):
            tensor[c] = tensor[c].view(width*height)[perm].view(height, width)
        return tensor


def make_shuffle_variant(full_dataset):
    d2 = copy(full_dataset)
    d2.ls_transform = transforms.Compose([d2.ls_transform, Shuffle()])
    d2.vs_transform = transforms.Compose([d2.vs_transform, Shuffle()])
    d2.ts_transform = transforms.Compose([d2.ts_transform, Shuffle()])
    return d2


class ShuffleFactory(object):
    @classmethod
    def same_as(cls, full_dataset):
        return make_shuffle_variant(full_dataset)




class InverseFactory(object):
    @classmethod
    def apply_invert(cls, full_dataset, make_copy=True):
        try:
            from PIL.ImageChops import invert
        except ImportError:
            from PIL.ImageOps import invert
        d2 = copy(full_dataset) if make_copy else full_dataset
        d2.ls_transform = transforms.Compose([invert, d2.ls_transform])
        d2.vs_transform = transforms.Compose([invert, d2.vs_transform])
        d2.ts_transform = transforms.Compose([invert, d2.ts_transform])
        return d2


    def __init__(self, full_dataset_factory):
        self.fdf = full_dataset_factory

    def __call__(self, *args, **kwargs):
        full_dataset = self.fdf(*args, **kwargs)
        return self.__class__.apply_invert(full_dataset, make_copy=False)

    def same_as(self, ref_full_dataset):
        fd = self.fdf.same_as(ref_full_dataset)
        return self.__class__.apply_invert(fd, make_copy=False)


# ================================ DATA AUG. ================================= #
class DataAugmentation(object):
    def get_transform(self):
        return transforms.Compose()

    def partial(self, full_dataset_factory, **kwargs):
        return partial(full_dataset_factory,
                       ls_data_augmentation=self.get_transform(), **kwargs)


class CropAugmented(DataAugmentation):
    def __init__(self, size=32, padding=4, padding_mode="reflect"):
        self.kwargs = {"size": size, "padding":padding,
                       "padding_mode":padding_mode}

    def get_transform(self):
        return transforms.RandomCrop(**self.kwargs)


class CropHzFlipAugmented(DataAugmentation):
    def __init__(self, size=32, padding=4, padding_mode="reflect"):
        self.kwargs = {"size": size, "padding": padding,
                       "padding_mode": padding_mode}

    def get_transform(self):
        return transforms.Compose([
            transforms.RandomCrop(**self.kwargs),
            transforms.RandomHorizontalFlip(),
        ])


class CropHzVFlipAugmented(DataAugmentation):
    def __init__(self, size=32, padding=4, padding_mode="reflect"):
        self.kwargs = {"size": size, "padding": padding,
                       "padding_mode": padding_mode}

    def get_transform(self):
        return transforms.Compose([
            transforms.RandomCrop(**self.kwargs),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])


class FlipsAugmented(DataAugmentation):
    def get_transform(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
