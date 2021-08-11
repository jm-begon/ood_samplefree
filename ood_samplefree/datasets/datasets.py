"""
Class |  STL 10  |  CIFAR 10
------+----------+------------
  0   | airplane | airplane
  1   | bird     | automobile
  2   | car      | bird
  3   | cat      | cat
  4   | deer     | deer
  5   | dog      | dog
  6   | horse    | frog
  7   | monkey   | horse
  8   | ship     | ship
  9   | truck    | truck
Need to switch classes 1 and 2 and classes 6 and 7.
Note that monkey (STL 10) is not the same as frog (CIFAR 10)
"""
import os
from abc import ABCMeta, abstractmethod
from copy import copy

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.utils import check_random_state
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, Subset
from torchvision import transforms



def get_data_folder(db_folder):
    import getpass
    username = getpass.getuser()
    path = os.path.join("/scratch", username, "data", db_folder)
    if os.path.exists(path):
        return path
    data_folder = os.path.expanduser("~/data")
    return os.path.join(data_folder, db_folder)





class FullDataset(object, metaclass=ABCMeta):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Compose([])

    @classmethod
    def get_default_shape(cls):
        return 0, 0, 0

    @classmethod
    def get_default_lengths(cls):
        """
        Return
        ------
        train, val, test: triplet of int
            The sizes of the training, validation and testing sets, respectively
        """
        return 0, 0, 0

    @classmethod
    def get_default_n_outputs(cls):
        return 0

    @classmethod
    def same_as(cls, full_dataset, overriding_shape=None,
                overrinding_normalization=None, **kwargs):
        shape = full_dataset.shape if overriding_shape is None else overriding_shape
        normalization = full_dataset.normalization if overrinding_normalization is None else overrinding_normalization
        return cls(shape, normalization, **kwargs)

    @classmethod
    def base_transform(cls):
        return list()

    def __init__(self, shape=None, normalization=None,
                 ls_data_augmentation=None, folder=None):
        def_shape = self.__class__.get_default_shape()
        if shape is None:
            shape = def_shape
        if normalization is None:
            normalization = self.__class__.get_default_normalization()

        # Transform
        # 0. Base transform
        # 1. Data augmentation (PIL)
        # 2. Shape (PIL)
        # 3. Tensor
        # 4. Normalization
        transform_list = self.__class__.base_transform()

        # Shape analysis
        if shape[0] != def_shape[0]:
            transform_list.append(transforms.Grayscale(shape[0]))
        if shape[1] != def_shape[1] or shape[2] != def_shape[2]:
            transform_list.append(transforms.Resize((shape[1], shape[2])))

        transform_list.append(transforms.ToTensor())
        transform_list.append(normalization)

        transform = transforms.Compose(transform_list)
        self.vs_transform = self.ts_transform = transform

        if ls_data_augmentation is not None:
            transform_list = [ls_data_augmentation] + transform_list

        self.ls_transform = transforms.Compose(transform_list)
        # For REPR
        self.shape = shape
        self.normalization = normalization
        self.ls_data_augmentation = ls_data_augmentation
        self._folder = folder

    def __repr__(self):
        return "{}(shape={}, normalizaton={}, ls_data_augmentation={})" \
               "".format(self.__class__.__name__,
                         repr(self.shape), repr(self.normalization),
                         repr(self.ls_data_augmentation))


    @property
    def n_outputs(self):
        return self.__class__.get_default_n_outputs()

    @property
    def folder(self):
        if self._folder is not None:
            return self._folder
        return get_data_folder(self.__class__.__name__.lower())

    @abstractmethod
    def get_ls_vs_ts(self):
        """
        Return
        ------
        ls, vs, ts: cls:`IndexableDataset`
            The learning, validation and test sets
        """
        return tuple()

    def vs_from_ls(self, train_set, ls_prop=0.9):
        # Validation set from Test set
        valid_set = copy(train_set)
        valid_set.transform = self.vs_transform

        indices = np.arange(len(train_set))
        split = int(len(train_set) * ls_prop)

        train_set = Subset(train_set, indices[:split])
        valid_set = Subset(valid_set, indices[split:])

        return train_set, valid_set

    def to_loaders(self, batch_size, *sets, shuffle=True, num_workers=0,
                   pin_memory=True):
        loaders = []
        for set in sets:
            loaders.append(
                torch.utils.data.DataLoader(set,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory)
            )

        return tuple(loaders)

    def get_loaders(self, ls_batch_size, test_batch_size=1024, num_workers=0,
                    pin_memory=True):
        ls, vs, ts = self.get_ls_vs_ts()
        valid_loader, test_loader = self.to_loaders(test_batch_size, vs, ts,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory)


        (train_loader,) = self.to_loaders(ls_batch_size, ls,
                                          num_workers=num_workers,
                                          pin_memory=pin_memory)

        return train_loader, valid_loader, test_loader


class PartialDataset(object):
    # Factory
    def __init__(self, factory, **kwargs):
        self.factory = factory
        self.kwargs = kwargs

    def get_default_normalization(self):
        return self.factory.get_default_normalization()

    def get_default_shape(self):
        return self.factory.get_default_shape()

    def get_default_lengths(self):
        """
        Return
        ------
        train, val, test: triplet of int
            The sizes of the training, validation and testing sets, respectively
        """
        return self.factory.get_default_lengths()

    def same_as(self, full_dataset):
        return self(full_dataset.shape, full_dataset.normalization)

    def __call__(self, shape=None, normalization=None,
                 ls_data_augmentation=None):
        kwargs = copy(self.kwargs)
        kwargs["shape"] = shape
        kwargs["normalization"] = normalization
        kwargs["ls_data_augmentation"] = ls_data_augmentation
        return self.factory(**kwargs)


class CIFAR10(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.247, 0.243, 0.261))

    @classmethod
    def get_default_lengths(cls):
        return 45000, 5000, 10000

    @classmethod
    def get_default_shape(cls):
        return 3, 32, 32

    @classmethod
    def get_default_n_outputs(cls):
        return 10


    def _get_pytorch_class(self):
        return torchvision.datasets.CIFAR10

    def get_ls_vs_ts(self):
        train_set = self._get_pytorch_class()(
            root=self.folder, train=True,
            download=True,
            transform=self.ls_transform,
        )

        test_set = self._get_pytorch_class()(
            root=self.folder, train=False,
            download=True,
            transform=self.ts_transform,
        )

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set


class CIFAR10Sub(CIFAR10):
    @property
    def folder(self):
        if self._folder is not None:
            return self._folder
        return get_data_folder("cifar10")

    def get_idx(self, set):
        return None

    def get_target_transform(self):
        return None

    def get_ls_vs_ts(self):
        train_set = self._get_pytorch_class()(
            root=self.folder, train=True,
            download=True,
            transform=self.ls_transform,
            target_transform=self.get_target_transform()
        )

        test_set = self._get_pytorch_class()(
            root=self.folder, train=False,
            download=True,
            transform=self.ts_transform,
            target_transform=self.get_target_transform()
        )

        train_set = Subset(train_set, self.get_idx(train_set))
        test_set = Subset(test_set, self.get_idx(test_set))

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set


class CIFAR10h1(CIFAR10Sub):
    def get_idx(self, set):
        return np.arange(len(set))[np.array(set.targets) < 5]



class CIFAR10h2(CIFAR10Sub):
    def get_idx(self, set):
        return np.arange(len(set))[np.array(set.targets) >= 5]

    def get_target_transform(self):
        def minus5(y):
            return y-5
        return minus5

class CIFAR10SubByDict(CIFAR10Sub):
    @property
    def class_dict(self):
        return {}

    def get_idx(self, set):
        targets = set.targets
        idx = np.zeros(len(targets), dtype=bool)
        for k in self.class_dict.keys():
            idx[targets == k] = 1
        return np.arange(len(set))[idx]

    def get_target_transform(self):
        def tt(y):
            return self.class_dict[y]
        return tt


class CIFAR10h3(CIFAR10SubByDict):
    @property
    def class_dict(self):
        # Animals
        return {
            2: 0, # Bird
            3: 1, # Cat
            4: 2, # Deer
            5: 3, # Dog
            6: 4, # Frog
            7: 5, # Horse
        }

class CIFAR10h4(CIFAR10SubByDict):
    @property
    def class_dict(self):
        # Transportation
        return {
            0: 0, # Airplane
            1: 1, # Automobile
            8: 2, # Ship
            9: 3, # Truck
        }



class CIFAR100(CIFAR10):

    @classmethod
    def get_default_n_outputs(cls):
        return 100

    def _get_pytorch_class(self):
        return torchvision.datasets.CIFAR100


class SVHN(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5)),

    @classmethod
    def get_default_shape(cls):
        return 3, 32, 32

    @classmethod
    def get_default_lengths(cls):
        return 65931, 7326, 26032

    @classmethod
    def get_default_n_outputs(cls):
        return 10

    def get_ls_vs_ts(self):
        train_set = torchvision.datasets.SVHN(
            root=self.folder, split="train",
            download=True,
            transform=self.ls_transform,
        )

        test_set = torchvision.datasets.SVHN(
            root=self.folder, split="test",
            download=True,
            transform=self.ts_transform,
        )

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set


class MNIST(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.1307,), (0.3081,))

    @classmethod
    def get_default_lengths(cls):
        return 54000, 6000, 10000

    @classmethod
    def get_default_shape(cls):
        return 1, 28, 28

    @classmethod
    def get_default_n_outputs(cls):
        return 10


    def get_ls_vs_ts(self):
        train_set = torchvision.datasets.MNIST(
            root=self.folder, train=True,
            download=True,
            transform=self.ls_transform
        )

        test_set = torchvision.datasets.MNIST(
            root=self.folder, train=False,
            download=True,
            transform=self.ts_transform
        )

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set


class FashionMNIST(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.2860,), (0.3530,))

    @classmethod
    def get_default_lengths(cls):
        return 54000, 6000, 10000

    @classmethod
    def get_default_shape(cls):
        return 1, 28, 28

    @classmethod
    def get_default_n_outputs(cls):
        return 10

    def get_ls_vs_ts(self):
        train_set = torchvision.datasets.FashionMNIST(
            root=self.folder, train=True,
            download=True,
            transform=self.ls_transform
        )

        test_set = torchvision.datasets.FashionMNIST(
            root=self.folder, train=False,
            download=True,
            transform=self.ts_transform
        )

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set



class STL10(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.4311,), (0.2634,))

    @classmethod
    def get_default_lengths(cls):
        return 4500, 500, 8000

    @classmethod
    def get_default_shape(cls):
        return 3, 93, 93

    @classmethod
    def get_default_n_outputs(cls):
        return 10


    def get_ls_vs_ts(self):
        train_set = torchvision.datasets.STL10(
            root=self.folder, split='train',
            download=True,
            transform=self.ls_transform,
        )

        test_set = torchvision.datasets.STL10(
            root=self.folder, split='test',
            download=True,
            transform=self.ts_transform
        )

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set

    def get_monkeys(self):
        ls, vs, ts = self.get_ls_vs_ts()
        all = []
        for samples in ls, ts:
            dataset = samples
            while(hasattr(dataset, "dataset")):
                # In case of concat. dataset
                dataset = dataset.dataset
            monkeys = dataset.labels == 7
            all.append(monkeys)
        return np.concatenate(all)



class Uniform(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.5,), (np.sqrt(1 / 12.),))

    @classmethod
    def get_default_lengths(cls):
        return 36000, 4000, 10000

    @classmethod
    def get_default_shape(cls):
        return 3, 32, 32

    class GeneratedDataset(Dataset):
        def __init__(self, n_instances, shape, n_classes, seed=None,
                     transform=None):
            self.transform = transforms.ToTensor() if transform is None else \
                transform
            rs = check_random_state(seed)
            total_size = tuple([n_instances] + list(shape[1:]) + [shape[0]])

            self.data = (rs.rand(*total_size)*255).astype("uint8")
            self.target = torch.from_numpy(
                rs.randint(0, n_classes, (n_instances,))).float()

        def __getitem__(self, item):
            # To be consistent with other dataset, must return a PIL.Image
            img = self.data[item]
            if img.shape[-1] == 1:
                # PIL does not like having 1 channel
                img = img.squeeze()
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
            return img, self.target[item]

        def __len__(self):
            return len(self.data)

    def __init__(self, shape=(3, 32, 32), normalization=None,
                 ls_data_augmentation=None, n_output=10, n_instances=50000):
        super().__init__(shape=shape, normalization=normalization,
                         ls_data_augmentation=ls_data_augmentation)

        self._n_outputs = n_output
        ls_size = int(n_instances * .8)
        ts_size = n_instances - ls_size
        self.train = self.__class__.GeneratedDataset(ls_size, shape, n_output,
                                                     98, self.ls_transform)
        self.test = self.__class__.GeneratedDataset(ts_size, shape, n_output,
                                                    97, self.ts_transform)

    @property
    def folder(self):
        return get_data_folder("uniform")


    @property
    def n_outputs(self):
        return self._n_outputs


    def get_ls_vs_ts(self):

        train_set, valid_set = self.vs_from_ls(self.train)

        return train_set, valid_set, self.test

class GLikeCif(FullDataset):
    # Like CIFAR 10
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.247, 0.243, 0.261)

    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize(cls.means, cls.stds)

    @classmethod
    def get_default_lengths(cls):
        return 45000, 5000, 10000

    @classmethod
    def get_default_shape(cls):
        return 3, 32, 32

    @classmethod
    def get_default_n_outputs(cls):
        return 10

    class GeneratedDataset(Dataset):
        def __init__(self, n_instances, shape, n_classes, means, stds,
                     seed=None, transform=None):
            self.transform = transforms.ToTensor() if transform is None else \
                transform

            rs = check_random_state(seed)
            total_size = tuple([n_instances] + list(shape[1:]) + [shape[0]])

            data = (rs.normal(means, stds, total_size)) * 255
            self.data = data.clip(0, 255).astype("uint8")
            self.target = torch.from_numpy(
                rs.randint(0, n_classes, (n_instances,))).float()

        def __getitem__(self, item):
            # To be consistent with other dataset, must return a PIL.Image
            img = self.data[item]
            if img.shape[-1] == 1:
                # PIL does not like having 1 channel
                img = img.squeeze()
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
            return img, self.target[item]

        def __len__(self):
            return len(self.data)

    def __init__(self, shape=None, normalization=None,
                 ls_data_augmentation=None):
        super().__init__(shape=shape, normalization=normalization,
                         ls_data_augmentation=ls_data_augmentation)

        ls_size, vs_size, ts_size = self.__class__.get_default_lengths()
        shape = self.__class__.get_default_shape()
        n_output = self.__class__.get_default_n_outputs()

        self.train = self.__class__.GeneratedDataset(ls_size, shape, n_output,
                                                     self.__class__.means,
                                                     self.__class__.stds,
                                                     47, self.ls_transform)

        self.val = self.__class__.GeneratedDataset(vs_size, shape, n_output,
                                                   self.__class__.means,
                                                   self.__class__.stds,
                                                   48, self.vs_transform)

        self.test = self.__class__.GeneratedDataset(ts_size, shape, n_output,
                                                    self.__class__.means,
                                                    self.__class__.stds,
                                                    49, self.ts_transform)

    def get_ls_vs_ts(self):
        return self.train, self.val, self.test





class Gaussian(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Compose([])  # Nothing to do

    @classmethod
    def get_default_lengths(cls):
        return 36000, 4000, 10000

    @classmethod
    def get_default_shape(cls):
        return 3, 32, 32

    class GeneratedDataset(Dataset):
        def __init__(self, n_instances, shape, n_classes, sigma, seed=None,
                     transform=None):
            self.transform = transforms.ToTensor() if transform is None else \
                transform

            rs = check_random_state(seed)
            total_size = tuple([n_instances] + list(shape[1:]) + [shape[0]])


            data = 255*(rs.normal(0, sigma, total_size) + 0.5)
            self.data = data.clip(0, 255).astype("uint8")
            self.target = torch.from_numpy(
                rs.randint(0, n_classes, (n_instances,))).float()

        def __getitem__(self, item):
            # To be consistent with other dataset, must return a PIL.Image
            img = self.data[item]
            if img.shape[-1] == 1:
                # PIL does not like having 1 channel
                img = img.squeeze()
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
            return img, self.target[item]

        def __len__(self):
            return len(self.data)


    def __init__(self, shape=(3, 32, 32), normalization=None,
                 ls_data_augmentation=None, n_output=10, n_instances=10000,
                 sigma=.25):
        # Note sigma = .25 --> Pr(-.5 < x < .5) = 95%
        super().__init__(shape=shape, normalization=normalization,
                         ls_data_augmentation=ls_data_augmentation)

        self._n_outputs = n_output
        self._sigma = sigma
        ls_size = int(n_instances * .8)
        ts_size = n_instances - ls_size
        self.train = self.__class__.GeneratedDataset(ls_size, shape, n_output,
                                                     self._sigma, 73,
                                                     self.ls_transform)
        self.test = self.__class__.GeneratedDataset(ts_size, shape, n_output,
                                                    self._sigma, 79,
                                                    self.ts_transform)

    @property
    def folder(self):
        return get_data_folder("gaussian")

    @property
    def n_outputs(self):
        return self._n_outputs


    def get_ls_vs_ts(self, ls_transform=None, transform=None):
        train_set, valid_set = self.vs_from_ls(self.train)

        return train_set, valid_set, self.test



class Constant(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.5,), (np.sqrt(1 / 12.),))

    @classmethod
    def get_default_lengths(cls):
        return 5, 2, 3

    @classmethod
    def get_default_shape(cls):
        return 3, 32, 32

    class GeneratedDataset(Dataset):
        def __init__(self, n_instances, shape, n_classes, seed=None,
                     transform=None):
            self.transform = transforms.ToTensor() if transform is None else \
                transform

            values = np.linspace(0, 255, n_instances).astype("uint8")

            rs = check_random_state(seed)
            rs.shuffle(values)

            total_size = tuple([n_instances] + list(shape[1:]) + [shape[0]])
            self.data = np.ones(total_size, dtype="uint8")

            for i, v in enumerate(values):
                self.data[i, ...] *= v

            self.target = torch.from_numpy(
                rs.randint(0, n_classes, (n_instances,))).float()

        def __getitem__(self, item):
            # To be consistent with other dataset, must return a PIL.Image
            img = self.data[item]
            if img.shape[-1] == 1:
                # PIL does not like having 1 channel
                img = img.squeeze()
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
            return img, self.target[item]

        def __len__(self):
            return len(self.data)

    def __init__(self, shape=(3, 32, 32), normalization=None,
                 ls_data_augmentation=None, n_output=10, n_instances=10):
        super().__init__(shape=shape, normalization=normalization,
                         ls_data_augmentation=ls_data_augmentation)

        self._n_outputs = n_output

        whole_data = self.__class__.GeneratedDataset(n_instances, shape,
                                                     n_output, 103,
                                                     self.ls_transform)

        ls_size = n_instances //2
        ts_size = (n_instances - ls_size) // 2
        vs_size = n_instances - ts_size - ls_size

        s, e = 0, ls_size
        self.train = torch.utils.data.Subset(whole_data, list(range(s, e)))
        s, e = e, e+vs_size
        self.val = torch.utils.data.Subset(whole_data, list(range(s, e)))
        s, e = e, e + ts_size
        self.test = torch.utils.data.Subset(whole_data, list(range(s, e)))

        self.val.transform = self.vs_transform
        self.test.transform = self.ts_transform


    @property
    def folder(self):
        return get_data_folder("uniform")


    @property
    def n_outputs(self):
        return self._n_outputs


    def get_ls_vs_ts(self):
        return self.train, self.val, self.test


class TinyImageNet(FullDataset):
    """
    From https://tiny-imagenet.herokuapp.com/

    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    """
    @classmethod
    def get_default_normalization(cls):
        # Computed on training set
        return transforms.Normalize((0.4802458005453784, 0.44807219498302625,
                                     0.3975477610692504),
                                    (0.2769864106388343, 0.2690644893481639,
                                     0.2820819105768366))

    @classmethod
    def get_default_lengths(cls):
        return 72000, 8000, 20000

    @classmethod
    def get_default_shape(cls):
        return 3, 64, 64

    @classmethod
    def get_default_n_outputs(cls):
        return 200   # 500 of each

    @property
    def folder(self):
        if self._folder is not None:
            return self._folder
        return get_data_folder("tinyimagenet/tiny-imagenet-200")

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            self.normalization
        ])

    def _get_full_set(self, subset, transform):
        path = os.path.join(self.folder, subset)
        return torchvision.datasets.ImageFolder(root=path,
                                                transform=transform)

    def get_ls_vs_ts(self):
        train_set = self._get_full_set("train", self.ls_transform)
        test_set = self._get_full_set("test", self.ts_transform)

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set


class LSUNTestSet(FullDataset):
    """
    From https://github.com/fyu/lsun

    conda install lmdb
    """
    @classmethod
    def get_default_normalization(cls):
        return transforms.Compose([])  # TODO

    @classmethod
    def get_default_shape(cls):
        return 3, 256, 256

    def __init__(self, shape=None, normalization=None,
                 ls_data_augmentation=None):
        super().__init__(shape=shape, normalization=normalization,
                         ls_data_augmentation=ls_data_augmentation)
        if shape is None:
            self.ls_transform = transforms.Compose([transforms.Resize(256, 256),
                                                    self.ls_transform])
            self.vs_transform = transforms.Compose([transforms.Resize(256, 256),
                                                    self.vs_transform])
            self.ts_transform = transforms.Compose([transforms.Resize(256, 256),
                                                    self.ts_transform])

    @classmethod
    def get_default_n_outputs(cls):
        return 0

    @property
    def n_outputs(self):
        raise NotImplementedError()

    @property
    def folder(self):
        return get_data_folder("lsun")

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            self.normalization
        ])

    def _get_full_set(self, transform):
        return torchvision.datasets.LSUN(root=self.folder,
                                         classes="test",
                                         transform=transform)

    def get_ls_vs_ts(self):
        actual_test_set = self._get_full_set(self.ts_transform)
        # TODO beware of transformation
        train_set, test_set = self.vs_from_ls(actual_test_set, 0.8)
        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set



class ImageNet(FullDataset):
    """
    See http://www.image-net.org/challenges/LSVRC/2012/

        - Download from http://image-net.org/challenges/LSVRC/2012/downloads.php#images
        - Use the Pytorch class to ready everithing

    https://stackoverflow.com/questions/40744700/how-can-i-find-imagenet-data-labels

    """

    @classmethod
    def get_default_normalization(cls):
        # stats from https://pytorch.org/docs/stable/torchvision/models.html
        return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        # stats estimated on training set
        # return transforms.Normalize(
        #     (0.48026870993652526, 0.45750730332850736, 0.4081817554661841),
        #     (0.2807399958925809, 0.27367912125650207, 0.28782503124759895))

    @classmethod
    def get_default_n_outputs(cls):
        return 2  # ~ [1000, 1300] / cls

    @classmethod
    def get_default_lengths(cls):
        return 1281167, 50000, 100000

    @classmethod
    def get_default_shape(cls):
        return 3, 224, 224

    @classmethod
    def base_transform(cls):
        return [transforms.Resize(256), transforms.CenterCrop(224)]

    def get_ls_vs_ts(self):
        ls = torchvision.datasets.ImageNet(
            root=self.folder,
            split="train",
            transform=self.ls_transform
        )

        vs = torchvision.datasets.ImageNet(
            root=self.folder,
            split="val",
            transform=self.vs_transform
        )

        # vs, ts = self.vs_from_ls(vs, 0.5)
        # ts.transform = self.ts_transform

        ts = torchvision.datasets.ImageFolder(
            root=os.path.join(self.folder, "alltest"),
            transform=self.ts_transform,
            target_transform=lambda x:-1  # unknown
        )
        return ls, vs, ts


