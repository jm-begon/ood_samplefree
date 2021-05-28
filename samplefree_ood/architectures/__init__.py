from functools import partial

from .arch_32x32 import ResNet50, DenseNet121, WideResNet40_2, ResNet34
from .arch_224x224 import make_resnet50, make_densenet121, \
    make_wideresnet50_2, make_mobilenet2048, make_small_shufflenet, TwoConvNet
from .util import count_parameters, magnitude, save_model



__ARCHITECTURES__ = {
    "resnet50": ResNet50,
    "resnet34": ResNet34,
    "densenet121": DenseNet121,
    "wideresnet": WideResNet40_2,
}


__ARCHITECTURES_224__ = {
    "resnet50": make_resnet50,
    "densenet121": make_densenet121,
    "wideresnet": make_wideresnet50_2,
    "renset50_pretrained": partial(make_resnet50, pretrained=True),
    "densenet121_pretrained": partial(make_densenet121, pretrained=True),
    "wideresnet_pretrained": partial(make_wideresnet50_2, pretrained=True),
    "mobilenet2048": make_mobilenet2048,
    "smallshufflenet": make_small_shufflenet,
    "twoconvnet": TwoConvNet,
}

__all__ = ["__ARCHITECTURES__", "__ARCHITECTURES_224__",
           "count_parameters", "magnitude", "save_model"]