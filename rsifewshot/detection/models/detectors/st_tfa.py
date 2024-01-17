# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import pdb

import torch

# from ..builder import DETECTORS, build_backbone, build_head, build_neck
# from rsidet.models.builder import DETECTORS
# from .base import BaseDetector
from rsidet.models.builder import DETECTORS
from rsidet.models.detectors.two_stage import TwoStageDetector


@DETECTORS.register_module()
class STTFA(TwoStageDetector):
    def __init__(self, merge_proposals=False, apply_st_rpn=True, wandb_cfg=None, **kwargs):
        self.merge_proposals = merge_proposals
        self.apply_st_rpn = apply_st_rpn
        self.wandb_cfg = wandb_cfg
        super().__init__(**kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rsidet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if self.apply_st_rpn:
                rpn_losses, proposal_list_pos, proposal_list_neg = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=gt_labels,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
            else:
                rpn_losses, proposal_list_pos = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
                proposal_list_neg = None

            losses.update(rpn_losses)

        else:
            proposal_list = proposals

        if self.merge_proposals:
            proposal_list = []
            if proposal_list_neg is not None:
                for proposal_pos, proposal_neg in zip(proposal_list_pos, proposal_list_neg):
                    proposal = torch.cat([proposal_pos, proposal_neg], dim=0)
                    _, idx = proposal[:, -1].sort(descending=True)
                    proposal_list.append(proposal[idx])
            else:
                proposal_list = proposal_list_pos

            roi_losses = self.roi_head.forward_train(x, img_metas,
                                                     proposal_list,
                                                     gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
        else:
            roi_losses = self.roi_head.forward_train(x, img_metas,
                                                     proposal_list_pos,
                                                     proposal_list_neg,
                                                     gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
        losses.update(roi_losses)

        if 'states' in losses:
            losses['states'].update({'else|img': img})

        return losses

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
