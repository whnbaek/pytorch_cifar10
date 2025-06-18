from functools import partial as _partial
from .densenet import DenseNet121, DenseNet169, DenseNet201, DenseNet161, densenet_cifar
from .dla_simple import SimpleDLA
from .dla import DLA
from .dpn import DPN26, DPN92
from .efficientnet import EfficientNetB0
from .googlenet import GoogLeNet
from .lenet import LeNet
from .mobilenet import MobileNet
from .mobilenetv2 import MobileNetV2
from .pnasnet import PNASNetA, PNASNetB
from .preact_resnet import (
    PreActResNet18,
    PreActResNet34,
    PreActResNet50,
    PreActResNet101,
    PreActResNet152,
)
from .regnet import RegNetX_200MF, RegNetX_400MF, RegNetY_400MF
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .resnext import ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d
from .senet import SENet18
from .shufflenet import ShuffleNetG2, ShuffleNetG3
from .shufflenetv2 import ShuffleNetV2 as _ShuffleNetV2

for net_size in [0.5, 1, 1.5, 2]:
    globals()[f"ShuffleNetV2_{net_size}x"] = _partial(_ShuffleNetV2, net_size=net_size)
from .vgg import VGG as _VGG

for arch in ("VGG11", "VGG13", "VGG16", "VGG19"):
    globals()[arch] = _partial(_VGG, arch)
