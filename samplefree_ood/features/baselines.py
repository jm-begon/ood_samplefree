"""
L stands for logit. It is an array of shape [n_samples, n_classes]
"""
from collections import defaultdict

import numpy as np
import torch
from scipy.special import logsumexp

from .structures import Monitor


def entropy(P):
    T = P.copy()
    mask = T == 0
    T[mask] = 1
    L = np.log(T)
    Q = P * L
    return - Q.sum(axis=1)


def softmax(x, T=1., axis=1):
    # Based on scipy.special.softmax

    # compute in log space for numerical stability
    return np.exp(x / T - logsumexp(x / T, axis=axis, keepdims=True))


def mp_and_h(L):
    P = softmax(L)
    return 1 - P.max(axis=1), entropy(P)


def T1000(L):
    P = softmax(L, T=1000.)
    return 1 - P.max(axis=1)


def get_linear_layer(model):
    layer = None
    for label, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if layer is not None:
                raise ValueError("Several `Linear` layer. Use a custom getter to specify the one to use.")
            layer = module

    if layer is None:
        raise ValueError("No `Linear` layer found.")

    return layer



class BaselineMonitor(Monitor):
    def __init__(self, linear_layer_getter=None):
        super().__init__()
        if linear_layer_getter is None:
            linear_layer_getter = get_linear_layer
        self.linear_layer = linear_layer_getter


    def create_hook(self):
        def hook(module, input, output):
            logits = output.data.cpu().numpy()

            mp, h = mp_and_h(logits)
            self.cache.save("mp", mp)
            self.cache.save("h", h)

            t1000 = T1000(logits)
            self.cache.save("T1000", t1000)
        return hook


    def watch(self, model):
        linear_layer = self.linear_layer(model)
        handle = linear_layer.register_forward_hook(self.create_hook())
        self.register_handle(handle)

