import warnings

import torch

from ood_samplefree.datasets import Uniform
from ood_samplefree.features import OneClassSum
from ..features.baselines import BaselineMonitor
from ..features.batchnorm import BatchNormMonitor
from ..features.latent import LatentMonitor, create_latent_saver
from ..features.structures import MultiMonitor, Cache
import numpy as np

class Deviceable(object):
    def __init__(self, use_cuda):
        if use_cuda:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                warnings.warn(
                    "Asking for cuda but not available. Falling back on CPU",
                    ResourceWarning)
                self._device = torch.device("cpu")
        else:
            self._device = torch.device("cpu")

    @property
    def device(self):
        return self._device


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def stream_ood_features(model, ood_dataset, latent_save_folder=None,
                        device=None, batch_size=256, num_workers=0,
                        pin_memory=True, fail_fast=True):

    # Device
    if device is None:
        device = get_device()

    # Model
    model.to(device)
    model.eval()

    # Data
    loader = torch.utils.data.DataLoader(ood_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         pin_memory=pin_memory)

    # Monitors
    monitor = MultiMonitor(BaselineMonitor(), BatchNormMonitor(),
                           LatentMonitor())

    latent_saver = create_latent_saver(latent_save_folder, fail_fast=fail_fast)


    # Compute ood features
    with monitor(model) as cache, latent_saver(model), torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            model(data.to(device))

            completion = (batch_idx + 1.) / len(loader)

            yield completion, cache






class OODStreamerWithSummaries(object):
    @classmethod
    def from_uniform(cls, ref_full_dataset, size=10000, excluded=None,
                 ref_save_path=None, device=None, batch_size=256,
                 num_workers=0, pin_memory=True):

        full_dataset = Uniform.same_as(ref_full_dataset, n_instances=size)
        dataset = torch.utils.data.ConcatDataset(full_dataset.get_ls_vs_ts())

        return cls(dataset, excluded, ref_save_path, device, batch_size,
                   num_workers, pin_memory)

    def __init__(self, reference, excluded=None,
                 ref_save_path=None, device=None, batch_size=256,
                 num_workers=0, pin_memory=True):



        self.reference_dataset = reference
        self.ref_save_path = ref_save_path

        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.exluced = excluded if excluded is not None else set()

        self.summary_models = []
        self.labels = []
        self.indicators = None

    def append_summary(self, fittable_model, label, *labels):
        self.summary_models.append(fittable_model)
        self.labels.append(label)
        self.labels.extend(labels)


    def cache_to_X(self, cache):
        if self.indicators is None:
            self.indicators = [k for k in cache.keys() if k not in self.exluced]
            self.indicators.sort()  # Ensure same ordering

        return np.vstack([cache[key] for key in self.indicators]).T


    def fit_from_model(self, model):
        import os

        if not os.path.exists(self.ref_save_path):
            cache_ = None
            for cplt, cache in stream_ood_features(model,
                                                   self.reference_dataset,
                                                   latent_save_folder=None,
                                                   device=self.device,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory):
                cache_ = cache

            if cache_ is None:
                raise ValueError("Emtpy cache.")

            X = self.cache_to_X(cache_)

            if self.ref_save_path is not None:
                self.ref_save_path = os.path.expanduser(self.ref_save_path)
                folder = os.path.split(self.ref_save_path)[0]
                print("FOLDER:", folder)
                os.makedirs(folder, exist_ok=True)
                np.save(self.ref_save_path, X)
        else:
            X = np.load(self.ref_save_path)

        for model in self.summary_models:
            model.fit(X, None)


    def predict(self, X):
        ls = []
        for model in self.summary_models:
            z = model.predict(X)
            if z.ndim == 1:
                z = z.reshape(-1, 1)
            ls.append(z)

        return np.hstack(ls)

    def ensure(self):
        if len(self.summary_models) == 0:
            self.summary_models.append(OneClassSum())
            self.labels.append("1C-Sum")


    def stream(self, model, ood_dataset, latent_save_folder=None,
               device=None, batch_size=None, num_workers=None,
               pin_memory=None, fail_fast=True):
        if_none = lambda x, y: x if x is None else y
        device = if_none(device, self.device)
        batch_size = if_none(batch_size, self.batch_size)
        num_workers = if_none(num_workers, self.num_workers)
        pin_memory = if_none(pin_memory, self.pin_memory)


        self.ensure()

        self.fit_from_model(model)

        full_cache = Cache()

        for cplt, cache in stream_ood_features(model, ood_dataset,
                                               latent_save_folder,
                                               device, batch_size,
                                               num_workers, pin_memory,
                                               fail_fast=fail_fast):

            # Compute summary
            Z = self.predict(self.cache_to_X(cache))

            # Update local cache with cache
            for k, v in cache.items():
                full_cache.save(k, v)
            cache.clear()

            # Update local cache with summary
            for p, label in zip(range(Z.shape[1]), self.labels):
                full_cache.save(label, Z[:, p])

            # TODO cplt with no memoization
            yield cplt, full_cache






