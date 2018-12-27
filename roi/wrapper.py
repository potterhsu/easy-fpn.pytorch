import math
from enum import Enum

import torch
from torch import Tensor
from torch.nn import functional as F

from roi.align.crop_and_resize import CropAndResizeFunction


class Wrapper(object):

    class Mode(Enum):
        POOLING = 'pooling'
        ALIGN = 'align'

    OPTIONS = ['pooling', 'align']

    @staticmethod
    def apply(features: Tensor, proposal_bboxes: Tensor, mode: Mode, image_width: int, image_height: int) -> Tensor:
        _, _, feature_map_height, feature_map_width = features.shape
        proposal_bboxes = proposal_bboxes.detach()

        scale_x = image_width / feature_map_width
        scale_y = image_height / feature_map_height

        if mode == Wrapper.Mode.POOLING:
            pool = []
            for proposal_bbox in proposal_bboxes:
                start_x = max(min(round(proposal_bbox[0].item() / scale_x), feature_map_width - 1), 0)      # [0, feature_map_width)
                start_y = max(min(round(proposal_bbox[1].item() / scale_y), feature_map_height - 1), 0)     # (0, feature_map_height]
                end_x = max(min(round(proposal_bbox[2].item() / scale_x) + 1, feature_map_width), 1)        # [0, feature_map_width)
                end_y = max(min(round(proposal_bbox[3].item() / scale_y) + 1, feature_map_height), 1)       # (0, feature_map_height]
                roi_feature_map = features[..., start_y:end_y, start_x:end_x]
                pool.append(F.adaptive_max_pool2d(input=roi_feature_map, output_size=7))
            pool = torch.cat(pool, dim=0)
        elif mode == Wrapper.Mode.ALIGN:
            x1 = proposal_bboxes[:, 0::4] / scale_x
            y1 = proposal_bboxes[:, 1::4] / scale_y
            x2 = proposal_bboxes[:, 2::4] / scale_x
            y2 = proposal_bboxes[:, 3::4] / scale_y

            crops = CropAndResizeFunction(crop_height=7 * 2, crop_width=7 * 2)(
                features,
                torch.cat([y1 / (feature_map_height - 1), x1 / (feature_map_width - 1),
                           y2 / (feature_map_height - 1), x2 / (feature_map_width - 1)],
                          dim=1),
                torch.zeros(proposal_bboxes.shape[0], dtype=torch.int, device=proposal_bboxes.device)
            )
            pool = F.max_pool2d(input=crops, kernel_size=2, stride=2)
        else:
            raise ValueError

        return pool

