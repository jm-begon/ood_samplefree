import torch


def save_model(model, path, verbose=True):
    if path is None:
        return
    if verbose:
        print("Saving model in {}...".format(path), end="")
    torch.save(model.state_dict(), path)

def load_model(model, fpath, verbose=False):
    if verbose:
        print("Loading model from {}...".format(fpath))
    model.load_state_dict(torch.load(fpath, map_location="cpu"))

def count_parameters(model):
    return sum(layer.data.nelement() for layer in model.parameters())


def magnitude(number):
    s = str(number)
    for unit in "k", "M", "G", "T", "P":
        if number // 1000 > 1:
            number /= 1000.
            s = "~ {} {}".format(int(number+.5), unit)
    return s

