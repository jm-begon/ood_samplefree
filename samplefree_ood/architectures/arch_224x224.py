import torch
from torchvision.models import ShuffleNetV2
from torchvision.models.resnet import resnet50, _resnet, Bottleneck
from torchvision.models.densenet import densenet121
from torchvision.models.mobilenet import MobileNetV2

def make_resnet50(n_outputs=10, pretrained=False):
    model = resnet50(pretrained=pretrained, num_classes=n_outputs)
    model.input_size = (3, 224, 224)
    return model


def make_densenet121(n_outputs=10, pretrained=False):
    model = densenet121(pretrained=pretrained, num_classes=n_outputs)
    model.input_size = (3, 224, 224)
    return model




def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # NOT AVAILABLE IN TORCH 1.1.0
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def make_wideresnet50_2(n_outputs=10, pretrained=False):
    model = wide_resnet50_2(pretrained=pretrained, num_classes=n_outputs)
    model.input_size = (3, 224, 224)
    return model


class MobileNet2048(MobileNetV2):
    def __setattr__(self, name, value):
        if name == "fc":
            self.classifier[1] = value
            return
        if name == "last_channel":
            value = 2048
        super().__setattr__(name, value)

    def __getattr__(self, item):
        if item == "fc":
            return self.classifier[1]
        return super().__getattr__(item)


def make_mobilenet2048(n_outputs=10, pretrained=False):
    if pretrained:
        raise NotImplementedError()
    model = MobileNet2048(num_classes=n_outputs)
    model.input_size = (3, 224, 224)
    return model


def make_small_shufflenet(n_outputs=10, pretrained=False):
    if pretrained:
        raise ValueError("Not available")
    shufflenet = ShuffleNetV2((2, 2, 2), (128, 256, 512, 1024, 2048), n_outputs)
    shufflenet.input_size = (3, 224, 224)
    return shufflenet


class TwoConvNet(torch.nn.Module):
    def __init__(self, n_outputs=10, pretrained=False):
        super().__init__()
        if pretrained:
            raise ValueError("Not available")
        self.input_size = (3, 224, 224)

        # Use of separable convolution (depthwise + pointwise) ~ 1 conv.

        # Separable conv 1
        # |- depthwise
        self.dconv1 = torch.nn.Conv2d(in_channels=3, out_channels=63,
                                      kernel_size=7, stride=2, groups=3)
        # |- Pointwise
        self.pconv1 = torch.nn.Conv2d(in_channels=63, out_channels=512,
                                      kernel_size=1)
        self.sep_conv1 = torch.nn.Sequential(self.dconv1, self.pconv1)
        self.bn1 = torch.nn.BatchNorm2d(512)
        self.relu1 = torch.nn.ReLU(inplace=True)

        self.block1 = torch.nn.Sequential(self.sep_conv1, self.bn1, self.relu1)

        # Separable conv 2
        # |- depthwise
        self.dconv2 = torch.nn.Conv2d(in_channels=512, out_channels=1024,
                                      kernel_size=5, stride=2, groups=512)

        # |- Pointwise
        self.pconv2 = torch.nn.Conv2d(in_channels=1024, out_channels=2048,
                                      kernel_size=1)

        self.sep_conv2 = torch.nn.Sequential(self.dconv2, self.pconv2)

        self.bn2 = torch.nn.BatchNorm2d(2048)
        self.relu2 = torch.nn.ReLU(inplace=True)

        self.block2 = torch.nn.Sequential(self.sep_conv2, self.bn2, self.relu2)


        # Full network
        # self.features = torch.nn.Sequential(self.block1, self.block2)

        self.fc = torch.nn.Linear(2048, n_outputs)

    def forward(self, x):
        # x = self.features(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x
