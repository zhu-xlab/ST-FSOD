# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsidet.models.builder import HEADS
from rsidet.models.losses import accuracy
from . import CosineSimBBoxHead
from torch import Tensor


@HEADS.register_module()
class CosineSimSTBBoxHeadV2(CosineSimBBoxHead):
    """
    """
    def __init__(self,
                 st_thre=0.968,
                 st_thre_bg=0.95,
                 bg_thre=0.3,
                 novel_class_inds=None,
                 fix_neg_inds=False,
                 **kwargs):
        self.st_thre = st_thre
        self.st_thre_bg=st_thre_bg
        self.bg_thre = bg_thre
        self.novel_class_inds = novel_class_inds
        self.fix_neg_inds = fix_neg_inds
        super(CosineSimSTBBoxHeadV2, self).__init__(**kwargs)

    def loss(self,
             stu_cls_score,
             stu_bbox_pred,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if stu_cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)

            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            neg_inds = (labels < 0) | (labels == bg_class_ind)

            cls_score_probs = F.softmax(cls_score, dim=1).detach()
            probs, pseudo_labels = cls_score_probs.max(dim=1)
            _, pseudo_labels2 = cls_score_probs.topk(dim=1, k=2)
            pseudo_labels2 = pseudo_labels2[:, 1]

            # if self.novel_class_inds is not None:
            assert self.novel_class_inds is not None

            novel_class_inds = torch.tensor(self.novel_class_inds).to(pseudo_labels.device)
            novel_inds = (pseudo_labels.unsqueeze(1) == novel_class_inds.unsqueeze(0)).sum(dim=1)
            novel_inds = novel_inds > 0
            novel_inds2 = (pseudo_labels2.unsqueeze(1) == novel_class_inds.unsqueeze(0)).sum(dim=1)
            novel_inds2 = novel_inds2 > 0

            bg_inds = (pseudo_labels == bg_class_ind)
            bg_novel_inds = (pseudo_labels == bg_class_ind) & pseudo_labels2
            base_inds = ~(novel_inds | bg_inds)

            # For matched foreground boxes, use the ground truth directly
            pseudo_labels[pos_inds] = labels[pos_inds]

            # For background boxes, if the output probability is larger than st_thre, use them as the pseudo labels
            # if self.fix_neg_inds:
            # pseudo_labels[neg_inds] = labels[neg_inds]
            pseudo_labels[neg_inds & (probs < self.st_thre)] = labels[neg_inds & (probs < self.st_thre)]

            pseudo_labels[neg_inds & base_inds] = labels[neg_inds & base_inds]

            # For background boxes, if the output probablity is smaller than st_thre, but larger
            # than bg_thre, ignore them. They could be foreground as well.
            # For base classes, just use the normal label weights
            weight_mask_novel = neg_inds & novel_inds & (probs > self.bg_thre) & (probs < self.st_thre)
            weight_mask_bg = neg_inds & bg_inds & novel_inds2 & (probs < self.st_thre_bg)

            label_weights[weight_mask_novel | weight_mask_bg] = 0


            # labels[neg_inds] = pseudo_labels[neg_inds]
            # label_weights[neg_inds] = (probs[neg_inds] > self.st_thre).sum() / probs.shape[0]
            # label_weights[neg_inds] = (probs[neg_inds] > self.st_thre)

            if stu_cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    stu_cls_score,
                    # labels,
                    pseudo_labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(stu_cls_score, pseudo_labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(stu_cls_score, pseudo_labels)

        if stu_bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    stu_bbox_pred = self.bbox_coder.decode(rois[:, 1:], stu_bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = stu_bbox_pred.view(
                        stu_bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = stu_bbox_pred.view(
                        stu_bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses


