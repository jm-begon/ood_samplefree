import os
import shutil
import warnings

import numpy as np

from .structures import Monitor, MultiHook
from .baselines import get_linear_layer

def norm_of(X, axis=-1):
    """
    X: array [n_samples, n_features]
    """
    return np.sqrt(np.sum(X**2, axis=axis))


def predicted_class(Z):
    return Z.argmax(axis=1)


def latent_metrics(U, C, W):

    W_norm = norm_of(W)
    U_norm = norm_of(U)

    M_k = U * W[C]
    A = np.sum(M_k, axis=1)
    Q = A / W_norm[C]

    norm = -U_norm
    proj = -Q
    ang = 1 - (Q / U_norm)
    act = -A

    return norm, act, proj, ang


def compute_ang_p(ang, act_p, act):
    rAct = act_p / act
    return 1 - ((1 - ang) * rAct)


class LatentMonitor(Monitor):
    def __init__(self, linear_layer_getter=None):
        super().__init__()
        if linear_layer_getter is None:
            linear_layer_getter = get_linear_layer
        self.linear_layer = linear_layer_getter

        self.W = None
        self.W_pos = None
        self.W_pos_mask = None


    def create_hook(self):
        def hook(module, input, output):
            latents = input[0].data.cpu().numpy()
            logits = output.data.cpu().numpy()

            C = predicted_class(logits)
            norm, act, proj, ang = latent_metrics(latents, C, self.W)
            self.cache.save("norm", norm)
            self.cache.save("act", act)
            self.cache.save("proj", proj)
            self.cache.save("ang", ang)

            latents *= self.W_pos_mask[C]
            norm_p, act_p, proj_p, ang_pp = latent_metrics(latents, C, self.W_pos)
            self.cache.save("norm+", norm_p)
            self.cache.save("act+", act_p)
            self.cache.save("proj+", proj_p)
            self.cache.save("ang++", ang_pp)

            self.cache.save("ang+", compute_ang_p(ang, act_p, act))

        return hook


    def watch(self, model):
        linear_layer = self.linear_layer(model)
        W = linear_layer.weight.data.cpu().numpy()
        self.W = W
        self.W_pos_mask = (W>0).astype(float)
        self.W_pos = W * self.W_pos_mask

        handle = linear_layer.register_forward_hook(self.create_hook())
        self.register_handle(handle)



class LatentSaver(MultiHook):
    # Hook

    @classmethod
    def load_latent_matrix(cls, folder, force=False):
        files = []
        with os.scandir(os.path.expanduser(folder)) as entries:
            # Get npy files
            files = [entry for entry in entries if entry.is_file() and
                     entry.name.endswith(".npy")]

        if len(files) == 0:
            return list()

        files.sort(key=(lambda x: x.name))

        # Verify timestamp ordering match
        prev_time = files[0].stat().st_mtime_ns
        for entry in files[1:]:
            curr_time = entry.stat().st_mtime_ns
            if curr_time < prev_time:
                if force:
                    warnings.warn("Ordering mismatch")
                else:
                    raise IOError("Ordering mismatch")

            prev_time = curr_time

        arrs = []
        for entry in files:
            arrs.append(np.load(entry.path))

        return np.vstack(arrs)

    @classmethod
    def load_batches_save_whole(cls, folder, force=False):
        folder = os.path.expanduser(folder)
        arr = cls.load_latent_matrix(folder, force=force)
        if len(arr) > 0:
            np.save(folder, arr)
        return len(arr) > 0

    def __init__(self, folder, linear_layer_getter=None, max_n_batch_order=10,
                 auto_remove=False, fail_fast=True):
        super().__init__()
        self.folder = os.path.expanduser(folder)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        if linear_layer_getter is None:
            linear_layer_getter = get_linear_layer
        self.linear_layer = linear_layer_getter

        self.batch_number = 0
        self.suffix_length = max_n_batch_order
        self.auto_remove = auto_remove
        self.fail_fast = fail_fast

    def create_hook(self):
        def hook(module, input, output):
            latents = input[0].data.cpu().numpy()
            fname = "batch_{}".format(str(self.batch_number).zfill(self.suffix_length))
            fpath = os.path.join(self.folder, fname)
            np.save(fpath, latents)
            self.batch_number += 1
        return hook



    def __call__(self, model):
        linear_layer = self.linear_layer(model)
        handle = linear_layer.register_forward_hook(self.create_hook())
        self.register_handle(handle)
        return self

    def concatenate_batches(self, remove_folder=False, force=False):
        saved = self.__class__.load_batches_save_whole(self.folder, force=force)
        if saved and remove_folder:
            shutil.rmtree(self.folder)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.concatenate_batches(self.auto_remove, force=not self.fail_fast)


def create_latent_saver(folder, linear_layer_getter=None, max_n_batch_order=10,
                        fail_fast=True):
    if folder is None:
        return MultiHook()
    return LatentSaver(folder, linear_layer_getter=linear_layer_getter,
                       max_n_batch_order=max_n_batch_order, fail_fast=fail_fast)