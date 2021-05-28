import numpy as np
import torch
from scipy import stats
from pt_inspector.stat import Stat

from .structures import Monitor

# def weighted_norm(weights, vector):
#     print(weights.shape, vector.shape)
#     total_weight = weights.sum()
#     normalized_weights = weights / total_weight
#     return np.sqrt(np.sum((normalized_weights*vector)**2, axis=-1))

__PRINT__ = True


def norms(T, w):
    total_weight = w.sum()
    normalized_weights = w / total_weight

    M = T.sum(axis=1) / total_weight
    # M = np.sqrt(M)
    R = np.sqrt(np.sum(normalized_weights*T, axis=1))


    return M, R


def stouffer_p_value(z_scores, k, rho=0.):
    """
    Parameters
    ----------
    z_scores: array [n, b]

    Return
    ------
    p_value of the double sided gaussian
    """
    if z_scores.ndim == 2:
        z_scores = z_scores.sum(axis=1)
    z_agr = z_scores / np.sqrt(k * (1 + (k - 1) * rho))
    return 2 * (1-stats.norm.cdf(np.abs(z_agr)))



def chi2(chi, df):
    if chi.ndim == 2:
        chi = chi.sum(axis=1)
    return 1-stats.chi2.cdf(chi, df)


def chi2norm(x, df):
    chi = stats.chi2.cdf(x, df)
    z = stats.norm.ppf(chi)
    # Can get +/-inf
    mask = np.isneginf(z)
    z[mask] = -5

    mask = np.isposinf(z)
    z[mask] = 5
    return chi, z



class BatchNormMonitor(Monitor):
    def __init__(self, n_input_channels=3):
        super().__init__()
        self.lut = {"input": 0}
        self.n_channels = np.array([n_input_channels])
        self.z_scores = []
        self.dms = []
        self.other_dms = []

        self.dss = []
        self.dss05 = []
        self.dss95 = []


    def _reset(self):
        def ls():
            return [None for _ in range(len(self.n_channels))]
        self.dms = ls()
        self.other_dms = ls()
        self.dss = ls()
        self.z_scores = ls()
        self.dss05 = ls()
        self.dss95 = ls()


    def create_hook(self, label, is_input):
        def hook(module, input, output):
            idx = self.lut[label]

            with torch.no_grad():
                if is_input:
                    tensor = input[0]
                    mu = 0
                    sigma = 1

                else:
                    tensor = output
                    mu = module.bias
                    sigma = module.weight

                z = tensor.mean(dim=[2, 3])
                samples = (z - mu) / sigma
                wh = tensor.size(2) *  tensor.size(3)
                rho = .5
                samples = samples / np.sqrt(1./wh + rho*(wh-1.)/wh)
                samples = samples.data.cpu().numpy()  # [n x c]

                z_scores = samples.sum(axis=1)
                self.z_scores[idx] = z_scores


                z = (tensor.permute(0, 2, 3, 1) - mu) / sigma
                n, w, h, c = z.size()
                samples = z.view(n, w * h, c).data.cpu().numpy()

                mean = samples.mean(axis=1)
                std = samples.std(axis=1)

                # DMS/DSS
                dms = mean ** 2  # ch2 df1
                aos = (samples ** 2).mean(axis=1)
                self.other_dms[idx] = np.sum(aos, axis=1)
                self.dms[idx] = np.sum(dms, axis=1)
                self.dss[idx] = np.sum((std - 1) ** 2, axis=1)

                # DSS-ext
                dl = (w * h) - 1
                chi_dss, z_dss = chi2norm(dl*(std**2), dl)
                self.dss95[idx] = np.sum(chi_dss > 0.95, axis=1)
                self.dss05[idx] = np.sum(chi_dss < 0.05, axis=1)


        return hook


    def create_end_hook(self):
        def end_hook(*args):
            n_channels_input = self.n_channels[0]
            n_channels_last = self.n_channels[-1]
            n_channels_total = np.sum(self.n_channels)

            def npfy(x):
                """return [n_samples, n_batch_layers]"""
                return np.array(x).T


            # Gaussian hyp.
            self.cache.save("in-nota", 1-stouffer_p_value(self.z_scores[0],
                                                          n_channels_input))
            z_scores = npfy(self.z_scores)
            self.cache.save("nota",
                            1-stouffer_p_value(z_scores,
                                               n_channels_total,
                                               rho=0.2))


            # DMS
            self.cache.save("in-dms", self.dms[0] / n_channels_input)
            dms = npfy(self.dms)
            self.cache.save("dms", dms.sum(axis=1) / n_channels_total)

            # -- other
            self.cache.save("in-dms-aos", self.other_dms[0] / n_channels_input)
            other_dms = npfy(self.other_dms)
            self.cache.save("dms-aos", other_dms.sum(axis=1) / n_channels_total)

            # DSS
            self.cache.save("in-dss", self.dss[0] / n_channels_input)
            dss = npfy(self.dss)
            self.cache.save("dss", dss.sum(axis=1) / n_channels_total)

            # -- ext
            dss05 = npfy(self.dss05)
            dss95 = npfy(self.dss95)
            self.cache.save("dss-ext", (dss05.sum(axis=1) + dss95.sum(axis=1)) / float(n_channels_total))

            self._reset()

        return end_hook


    def watch(self, model):
        h = model.register_forward_hook(self.create_hook("input", is_input=True))
        self.register_handle(h)
        n_channels = [x for x in self.n_channels]

        for label, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                h = module.register_forward_hook(self.create_hook(label, False))
                self.register_handle(h)

                self.lut[label] = len(n_channels)
                n_channels.append(module.num_features)

        h = model.register_forward_hook(self.create_end_hook())
        self.register_handle(h)

        self.n_channels = np.array(n_channels)
        self._reset()
