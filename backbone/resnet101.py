from typing import Tuple

import torchvision
from torch import nn

from backbone.base import Base


class ResNet101(Base):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self) -> Tuple[Base.ConvLayers, Base.LateralLayers, Base.DealiasingLayers, int]:
        resnet101 = torchvision.models.resnet101(pretrained=self._pretrained)

        # list(resnet101.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet101.children())

        conv1 = nn.Sequential(*children[:3])
        conv2 = nn.Sequential(*([children[3]] + list(children[4].children())))
        conv3 = children[5]
        conv4 = children[6]
        conv5 = children[7]

        num_features_out = 256

        lateral_c2 = nn.Conv2d(in_channels=256, out_channels=num_features_out, kernel_size=1)
        lateral_c3 = nn.Conv2d(in_channels=512, out_channels=num_features_out, kernel_size=1)
        lateral_c4 = nn.Conv2d(in_channels=1024, out_channels=num_features_out, kernel_size=1)
        lateral_c5 = nn.Conv2d(in_channels=2048, out_channels=num_features_out, kernel_size=1)

        dealiasing_p2 = nn.Conv2d(in_channels=num_features_out, out_channels=num_features_out, kernel_size=3, padding=1)
        dealiasing_p3 = nn.Conv2d(in_channels=num_features_out, out_channels=num_features_out, kernel_size=3, padding=1)
        dealiasing_p4 = nn.Conv2d(in_channels=num_features_out, out_channels=num_features_out, kernel_size=3, padding=1)

        for parameters in [module.parameters() for module in [conv1, conv2]]:
            for parameter in parameters:
                parameter.requires_grad = False

        conv_layers = Base.ConvLayers(conv1, conv2, conv3, conv4, conv5)
        lateral_layers = Base.LateralLayers(lateral_c2, lateral_c3, lateral_c4, lateral_c5)
        dealiasing_layers = Base.DealiasingLayers(dealiasing_p2, dealiasing_p3, dealiasing_p4)

        return conv_layers, lateral_layers, dealiasing_layers, num_features_out
