from abc import ABCMeta, abstractmethod, abstractproperty
from collections import defaultdict

import numpy as np


class MultiHook(object):
    def __init__(self):
        self.handles = []

    def register_handle(self, handle):
        self.handles.append(handle)

    def remove_handles(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_handles()

    def __call__(self, model):
        return self


class Cache(object):
    def __init__(self):
        self.data = defaultdict(list)

    def clear(self):
        self.data = defaultdict(list)

    def __getitem__(self, item):
        return np.hstack(self.data[item])

    def save(self, key, value):
        self.data[key].append(value)

    def keys(self):
        return self.data.keys()

    def __iter__(self):
        for k, v in self.data.items():
            yield k, np.hstack(v)

    def items(self):
        for k, v in self.data.items():
            yield k, v





class AbstractMonitor(MultiHook, metaclass=ABCMeta):
    def __call__(self, model):
        self.watch(model)
        return self

    @abstractmethod
    def watch(self, model):
        pass

    @abstractproperty
    def cache(self):
        pass

    def __enter__(self):
        super().__enter__()
        return self.cache


class Monitor(AbstractMonitor, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self._cache = Cache()

    @property
    def cache(self):
        return self._cache


class MultiMonitor(AbstractMonitor):
    class MultiCache(object):
        def __init__(self):
            self.caches = []
            self.dirty = False
            self.lut = {}

        def append(self, cache):
            self.caches.append(cache)
            self.dirty = True

        def clear(self):
            for cache in self.caches:
                cache.clear()
            self.dirty = True

        def keys(self):
            ks = []
            for cache in self.caches:
                ks.extend(cache.keys())
            return ks

        def __getitem__(self, item):
            if self.dirty:
                self.lut = {}
                for i, cache in enumerate(self.caches):
                    self.lut.update({k:i for k in cache.keys()})
            return self.caches[self.lut[item]][item]

        def __iter__(self):
            for cache in self.caches:
                for x in cache:
                    yield x

        def items(self):
            for cache in self.caches:
                for k, v in cache.items():
                    yield k, v

    def __init__(self, *monitors):
        super().__init__()
        self.monitors = monitors

    @property
    def cache(self):
        m_cache = self.__class__.MultiCache()
        for monitor in self.monitors:
            m_cache.append(monitor.cache)
        return m_cache

    def watch(self, model):
        for monitor in self.monitors:
            monitor.watch(model)

    def remove_handles(self):
        for monitor in self.monitors:
            monitor.remove_handles()

