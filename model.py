import os
from typing import Union, Tuple, List, NamedTuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from backbone.base import Base as BackboneBase
from bbox import BBox
from nms.nms import NMS
from roi.wrapper import Wrapper as ROIWrapper
from rpn.region_proposal_network import RegionProposalNetwork


class Model(nn.Module):

    class ForwardInput(object):
        class Train(NamedTuple):
            image: Tensor
            gt_classes: Tensor
            gt_bboxes: Tensor

        class Eval(NamedTuple):
            image: Tensor

    class ForwardOutput(object):
        class Train(NamedTuple):
            anchor_objectness_loss: Tensor
            anchor_transformer_loss: Tensor
            proposal_class_loss: Tensor
            proposal_transformer_loss: Tensor

        class Eval(NamedTuple):
            detection_bboxes: Tensor
            detection_classes: Tensor
            detection_probs: Tensor

    def __init__(self, backbone: BackboneBase, num_classes: int, pooling_mode: ROIWrapper.Mode,
                 anchor_ratios: List[Tuple[int, int]], anchor_scales: List[int], rpn_pre_nms_top_n: int, rpn_post_nms_top_n: int):
        super().__init__()

        conv_layers, lateral_layers, dealiasing_layers, num_features_out = backbone.features()
        self.conv1, self.conv2, self.conv3, self.conv4, self.conv5 = conv_layers
        self.lateral_c2, self.lateral_c3, self.lateral_c4, self.lateral_c5 = lateral_layers
        self.dealiasing_p2, self.dealiasing_p3, self.dealiasing_p4 = dealiasing_layers

        self._bn_modules = [it for it in self.conv1.modules() if isinstance(it, nn.BatchNorm2d)] + \
                           [it for it in self.conv2.modules() if isinstance(it, nn.BatchNorm2d)] + \
                           [it for it in self.conv3.modules() if isinstance(it, nn.BatchNorm2d)] + \
                           [it for it in self.conv4.modules() if isinstance(it, nn.BatchNorm2d)] + \
                           [it for it in self.conv5.modules() if isinstance(it, nn.BatchNorm2d)] + \
                           [it for it in self.lateral_c2.modules() if isinstance(it, nn.BatchNorm2d)] + \
                           [it for it in self.lateral_c3.modules() if isinstance(it, nn.BatchNorm2d)] + \
                           [it for it in self.lateral_c4.modules() if isinstance(it, nn.BatchNorm2d)] + \
                           [it for it in self.lateral_c5.modules() if isinstance(it, nn.BatchNorm2d)] + \
                           [it for it in self.dealiasing_p2.modules() if isinstance(it, nn.BatchNorm2d)] + \
                           [it for it in self.dealiasing_p3.modules() if isinstance(it, nn.BatchNorm2d)] + \
                           [it for it in self.dealiasing_p4.modules() if isinstance(it, nn.BatchNorm2d)]

        self.num_classes = num_classes

        self.rpn = RegionProposalNetwork(num_features_out, anchor_ratios, anchor_scales, rpn_pre_nms_top_n, rpn_post_nms_top_n)
        self.detection = Model.Detection(pooling_mode, self.num_classes)

    def forward(self, forward_input: Union[ForwardInput.Train, ForwardInput.Eval]) -> Union[ForwardOutput.Train, ForwardOutput.Eval]:
        # freeze batch normalization modules for each forwarding process just in case model was switched to `train` at any time
        for bn_module in self._bn_modules:
            bn_module.eval()
            for parameter in bn_module.parameters():
                parameter.requires_grad = False

        image = forward_input.image.unsqueeze(dim=0)
        image_height, image_width = image.shape[2], image.shape[3]

        # Bottom-up pathway
        c1 = self.conv1(image)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        # Top-down pathway and lateral connections
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(input=p5, size=(c4.shape[2], c4.shape[3]), mode='nearest')
        p3 = self.lateral_c3(c3) + F.interpolate(input=p4, size=(c3.shape[2], c3.shape[3]), mode='nearest')
        p2 = self.lateral_c2(c2) + F.interpolate(input=p3, size=(c2.shape[2], c2.shape[3]), mode='nearest')

        # Reduce the aliasing effect
        p4 = self.dealiasing_p4(p4)
        p3 = self.dealiasing_p3(p3)
        p2 = self.dealiasing_p2(p2)

        p6 = F.max_pool2d(input=p5, kernel_size=1, stride=2)

        # NOTE: We define the anchors to have areas of {32^2, 64^2, 128^2, 256^2, 512^2} pixels on {P2, P3, P4, P5, P6} respectively

        anchor_objectnesses = []
        anchor_transformers = []
        anchor_bboxes = []
        proposal_bboxes = []

        for p, anchor_size in zip([p2, p3, p4, p5, p6], [32, 64, 128, 256, 512]):
            p_anchor_objectnesses, p_anchor_transformers = self.rpn.forward(features=p, image_width=image_width, image_height=image_height)
            p_anchor_bboxes = self.rpn.generate_anchors(image_width, image_height,
                                                        num_x_anchors=p.shape[3], num_y_anchors=p.shape[2],
                                                        anchor_size=anchor_size).cuda()
            p_proposal_bboxes = self.rpn.generate_proposals(p_anchor_bboxes, p_anchor_objectnesses, p_anchor_transformers,
                                                            image_width, image_height)
            anchor_objectnesses.append(p_anchor_objectnesses)
            anchor_transformers.append(p_anchor_transformers)
            anchor_bboxes.append(p_anchor_bboxes)
            proposal_bboxes.append(p_proposal_bboxes)

        anchor_objectnesses = torch.cat(anchor_objectnesses, dim=0)
        anchor_transformers = torch.cat(anchor_transformers, dim=0)
        anchor_bboxes = torch.cat(anchor_bboxes, dim=0)
        proposal_bboxes = torch.cat(proposal_bboxes, dim=0)

        if self.training:
            forward_input: Model.ForwardInput.Train

            anchor_sample_fg_indices, anchor_sample_selected_indices, gt_anchor_objectnesses, gt_anchor_transformers = self.rpn.sample(anchor_bboxes, forward_input.gt_bboxes, image_width, image_height)
            anchor_objectnesses = anchor_objectnesses[anchor_sample_selected_indices]
            anchor_transformers = anchor_transformers[anchor_sample_fg_indices]
            anchor_objectness_loss, anchor_transformer_loss = self.rpn.loss(anchor_objectnesses, anchor_transformers, gt_anchor_objectnesses, gt_anchor_transformers)

            proposal_sample_fg_indices, proposal_sample_selected_indices, gt_proposal_classes, gt_proposal_transformers = self.detection.sample(proposal_bboxes, forward_input.gt_classes, forward_input.gt_bboxes)
            proposal_bboxes = proposal_bboxes[proposal_sample_selected_indices]
            proposal_classes, proposal_transformers = self.detection.forward(p2, p3, p4, p5, proposal_bboxes, image_width, image_height)
            proposal_class_loss, proposal_transformer_loss = self.detection.loss(proposal_classes, proposal_transformers, gt_proposal_classes, gt_proposal_transformers)

            forward_output = Model.ForwardOutput.Train(anchor_objectness_loss, anchor_transformer_loss, proposal_class_loss, proposal_transformer_loss)
        else:
            proposal_classes, proposal_transformers = self.detection.forward(p2, p3, p4, p5, proposal_bboxes, image_width, image_height)
            detection_bboxes, detection_classes, detection_probs = self.detection.generate_detections(proposal_bboxes, proposal_classes, proposal_transformers, image_width, image_height)
            forward_output = Model.ForwardOutput.Eval(detection_bboxes, detection_classes, detection_probs)

        return forward_output

    def save(self, path_to_checkpoints_dir: str, step: int, optimizer: Optimizer, scheduler: _LRScheduler) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir, f'model-{step}.pth')
        checkpoint = {
            'state_dict': self.state_dict(),
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str, optimizer: Optimizer = None, scheduler: _LRScheduler = None) -> 'Model':
        checkpoint = torch.load(path_to_checkpoint)
        self.load_state_dict(checkpoint['state_dict'])
        step = checkpoint['step']
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return step

    class Detection(nn.Module):

        def __init__(self, pooling_mode: ROIWrapper.Mode, num_classes: int):
            super().__init__()
            self._pooling_mode = pooling_mode
            self._hidden = nn.Sequential(
                nn.Linear(256 * 7 * 7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU()
            )
            self.num_classes = num_classes
            self._class = nn.Linear(1024, num_classes)
            self._transformer = nn.Linear(1024, num_classes * 4)
            self._transformer_normalize_mean = torch.tensor([0., 0., 0., 0.], dtype=torch.float).cuda()
            self._transformer_normalize_std = torch.tensor([.1, .1, .2, .2], dtype=torch.float).cuda()

        def forward(self, p2: Tensor, p3: Tensor, p4: Tensor, p5: Tensor, proposal_bboxes: Tensor, image_width: int, image_height: int) -> Tuple[Tensor, Tensor]:
            w = proposal_bboxes[:, 2] - proposal_bboxes[:, 0]
            h = proposal_bboxes[:, 3] - proposal_bboxes[:, 1]
            k0 = 4
            k = torch.floor(k0 + torch.log2(torch.sqrt(w * h) / 224)).long()
            k = torch.clamp(k, min=2, max=5)

            k_to_p_dict = {2: p2, 3: p3, 4: p4, 5: p5}
            unique_k = torch.unique(k)

            # NOTE: `picked_indices` is for recording the order of selection from `proposal_bboxes`
            #       so that `pools` can be then restored to make it have a consistent correspondence
            #       with `proposal_bboxes`. For example:
            #
            #           proposal_bboxes =>  B0  B1  B2
            #            picked_indices =>   1   2   0
            #                     pools => BP1 BP2 BP0
            #            sorted_indices =>   2   0   1
            #                     pools => BP0 BP1 BP2

            pools = []
            picked_indices = []

            for uk in unique_k:
                uk = uk.item()
                p = k_to_p_dict[uk]
                uk_indices = (k == uk).nonzero().view(-1)
                uk_proposal_bboxes = proposal_bboxes[uk_indices]
                pool = ROIWrapper.apply(p, uk_proposal_bboxes, mode=self._pooling_mode, image_width=image_width, image_height=image_height)
                pools.append(pool)
                picked_indices.append(uk_indices)

            pools = torch.cat(pools, dim=0)
            picked_indices = torch.cat(picked_indices, dim=0)

            _, sorted_indices = torch.sort(picked_indices)
            pools = pools[sorted_indices]

            pools = pools.view(pools.shape[0], -1)
            hidden = self._hidden(pools)
            classes = self._class(hidden)
            transformers = self._transformer(hidden)
            return classes, transformers

        def sample(self, proposal_bboxes: Tensor, gt_classes: Tensor, gt_bboxes: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            sample_fg_indices = torch.arange(end=len(proposal_bboxes), dtype=torch.long)
            sample_selected_indices = torch.arange(end=len(proposal_bboxes), dtype=torch.long)

            # find labels for each `proposal_bboxes`
            labels = torch.ones(len(proposal_bboxes), dtype=torch.long).cuda() * -1
            ious = BBox.iou(proposal_bboxes, gt_bboxes)
            proposal_max_ious, proposal_assignments = ious.max(dim=1)
            labels[proposal_max_ious < 0.5] = 0
            labels[proposal_max_ious >= 0.5] = gt_classes[proposal_assignments[proposal_max_ious >= 0.5]]

            # select 128 samples
            fg_indices = (labels > 0).nonzero().view(-1)
            bg_indices = (labels == 0).nonzero().view(-1)
            fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 32)]]
            bg_indices = bg_indices[torch.randperm(len(bg_indices))[:128 - len(fg_indices)]]
            selected_indices = torch.cat([fg_indices, bg_indices])
            selected_indices = selected_indices[torch.randperm(len(selected_indices))]

            proposal_bboxes = proposal_bboxes[selected_indices]
            gt_proposal_transformers = BBox.calc_transformer(proposal_bboxes, gt_bboxes[proposal_assignments[selected_indices]])
            gt_proposal_classes = labels[selected_indices]

            gt_proposal_transformers = (gt_proposal_transformers - self._transformer_normalize_mean) / self._transformer_normalize_std

            gt_proposal_transformers = gt_proposal_transformers.cuda()
            gt_proposal_classes = gt_proposal_classes.cuda()

            sample_fg_indices = sample_fg_indices[fg_indices]
            sample_selected_indices = sample_selected_indices[selected_indices]

            return sample_fg_indices, sample_selected_indices, gt_proposal_classes, gt_proposal_transformers

        def loss(self, proposal_classes: Tensor, proposal_transformers: Tensor, gt_proposal_classes: Tensor, gt_proposal_transformers: Tensor) -> Tuple[Tensor, Tensor]:
            cross_entropy = F.cross_entropy(input=proposal_classes, target=gt_proposal_classes)

            proposal_transformers = proposal_transformers.view(-1, self.num_classes, 4)
            proposal_transformers = proposal_transformers[torch.arange(end=len(proposal_transformers), dtype=torch.long).cuda(), gt_proposal_classes]

            fg_indices = gt_proposal_classes.nonzero().view(-1)

            # NOTE: The default of `reduction` is `elementwise_mean`, which is divided by N x 4 (number of all elements), here we replaced by N for better performance
            smooth_l1_loss = F.smooth_l1_loss(input=proposal_transformers[fg_indices], target=gt_proposal_transformers[fg_indices], reduction='sum')
            smooth_l1_loss /= len(gt_proposal_transformers)

            return cross_entropy, smooth_l1_loss

        def generate_detections(self, proposal_bboxes: Tensor, proposal_classes: Tensor, proposal_transformers: Tensor, image_width: int, image_height: int) -> Tuple[Tensor, Tensor, Tensor]:
            proposal_transformers = proposal_transformers.view(-1, self.num_classes, 4)
            mean = self._transformer_normalize_mean.repeat(1, self.num_classes, 1)
            std = self._transformer_normalize_std.repeat(1, self.num_classes, 1)

            proposal_transformers = proposal_transformers * std - mean
            proposal_bboxes = proposal_bboxes.view(-1, 1, 4).repeat(1, self.num_classes, 1)
            detection_bboxes = BBox.apply_transformer(proposal_bboxes.view(-1, 4), proposal_transformers.view(-1, 4))

            detection_bboxes = detection_bboxes.view(-1, self.num_classes, 4)

            detection_bboxes[:, :, [0, 2]] = detection_bboxes[:, :, [0, 2]].clamp(min=0, max=image_width)
            detection_bboxes[:, :, [1, 3]] = detection_bboxes[:, :, [1, 3]].clamp(min=0, max=image_height)

            proposal_probs = F.softmax(proposal_classes, dim=1)

            detection_bboxes = detection_bboxes.cpu()
            proposal_probs = proposal_probs.cpu()

            generated_bboxes = []
            generated_classes = []
            generated_probs = []

            for c in range(1, self.num_classes):
                detection_class_bboxes = detection_bboxes[:, c, :]
                proposal_class_probs = proposal_probs[:, c]

                _, sorted_indices = proposal_class_probs.sort(descending=True)
                detection_class_bboxes = detection_class_bboxes[sorted_indices]
                proposal_class_probs = proposal_class_probs[sorted_indices]

                kept_indices = NMS.suppress(detection_class_bboxes.cuda(), threshold=0.3)
                detection_class_bboxes = detection_class_bboxes[kept_indices]
                proposal_class_probs = proposal_class_probs[kept_indices]

                generated_bboxes.append(detection_class_bboxes)
                generated_classes.append(torch.ones(len(kept_indices)) * c)
                generated_probs.append(proposal_class_probs)

            generated_bboxes = torch.cat(generated_bboxes, dim=0)
            generated_classes = torch.cat(generated_classes, dim=0)
            generated_probs = torch.cat(generated_probs, dim=0)
            return generated_bboxes, generated_classes, generated_probs
