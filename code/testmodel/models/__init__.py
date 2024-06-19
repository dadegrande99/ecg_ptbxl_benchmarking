from .inception1d import Inception1D
from .resnet1d import resnet1d18, resnet1d34, resnet1d50
from .resnet50_dropout_torch import _resnet_dropout

MODEL_LIST = [
    ("ResNet1D-18", resnet1d18),
    ("Inception1D", Inception1D),
    ("ResNet1D-34", resnet1d34),
    ("ResNet1D-50", resnet1d50),
    # ("ResNet50-MCDropout", _resnet_dropout),
]
