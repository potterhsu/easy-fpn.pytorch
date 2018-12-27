from typing import Tuple, Type, NamedTuple

from torch import nn


class Base(object):

    OPTIONS = ['resnet18', 'resnet50', 'resnet101']

    class ConvLayers(NamedTuple):
        conv1: nn.Module
        conv2: nn.Module
        conv3: nn.Module
        conv4: nn.Module
        conv5: nn.Module

    class LateralLayers(NamedTuple):
        lateral_c2: nn.Module
        lateral_c3: nn.Module
        lateral_c4: nn.Module
        lateral_c5: nn.Module

    class DealiasingLayers(NamedTuple):
        dealiasing_p2: nn.Module
        dealiasing_p3: nn.Module
        dealiasing_p4: nn.Module

    @staticmethod
    def from_name(name: str) -> Type['Base']:
        if name == 'resnet18':
            from backbone.resnet18 import ResNet18
            return ResNet18
        elif name == 'resnet50':
            from backbone.resnet50 import ResNet50
            return ResNet50
        elif name == 'resnet101':
            from backbone.resnet101 import ResNet101
            return ResNet101
        else:
            raise ValueError

    def __init__(self, pretrained: bool):
        super().__init__()
        self._pretrained = pretrained

    def features(self) -> Tuple[ConvLayers, LateralLayers, DealiasingLayers, int]:
        raise NotImplementedError
