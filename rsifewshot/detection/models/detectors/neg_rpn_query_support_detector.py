# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import pdb
import copy
from typing import Dict, List, Optional, Union
from mmcv.runner import auto_fp16
from mmcv.utils import ConfigDict


import torch

# from ..builder import DETECTORS, build_backbone, build_head, build_neck
# from rsidet.models.builder import DETECTORS
# from .base import BaseDetector
from rsidet.models.builder import DETECTORS
from .query_support_detector import QuerySupportDetector


@DETECTORS.register_module()
class NegRPNQuerySupportDetector(QuerySupportDetector):

    def forward_train(self,
                      query_data: Dict,
                      support_data: Dict,
                      proposals: Optional[List] = None,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_data (dict): In most cases, dict of query data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            support_data (dict):  In most cases, dict of support data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            proposals (list): Override rpn proposals with custom proposals.
                Use when `with_rpn` is False. Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        query_img = query_data['img']
        support_img = support_data['img']
        query_feats = self.extract_query_feat(query_img)
        support_feats = self.extract_support_feat(support_img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if self.rpn_with_support:
                rpn_losses, proposal_list_pos, proposal_list_neg = self.rpn_head.forward_train(
                    query_feats,
                    support_feats,
                    query_img_metas=query_data['img_metas'],
                    query_gt_bboxes=query_data['gt_bboxes'],
                    query_gt_labels=None,
                    query_gt_bboxes_ignore=query_data.get(
                        'gt_bboxes_ignore', None),
                    support_img_metas=support_data['img_metas'],
                    support_gt_bboxes=support_data['gt_bboxes'],
                    support_gt_labels=support_data['gt_labels'],
                    support_gt_bboxes_ignore=support_data.get(
                        'gt_bboxes_ignore', None),
                    proposal_cfg=proposal_cfg)
            else:
                rpn_losses, proposal_list_pos, proposal_list_neg = self.rpn_head.forward_train(
                    query_feats,
                    copy.deepcopy(query_data['img_metas']),
                    copy.deepcopy(query_data['gt_bboxes']),
                    gt_labels=None,
                    gt_bboxes_ignore=copy.deepcopy(
                        query_data.get('gt_bboxes_ignore', None)),
                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list_pos = proposals['pos']
            proposal_list_neg = proposals['neg']

        roi_losses = self.roi_head.forward_train(
            query_feats,
            support_feats,
            proposals_pos=proposal_list_pos,
            proposals_neg=proposal_list_neg,
            query_img_metas=query_data['img_metas'],
            query_gt_bboxes=query_data['gt_bboxes'],
            query_gt_labels=query_data['gt_labels'],
            query_gt_bboxes_ignore=query_data.get('gt_bboxes_ignore', None),
            support_img_metas=support_data['img_metas'],
            support_gt_bboxes=support_data['gt_bboxes'],
            support_gt_labels=support_data['gt_labels'],
            support_gt_bboxes_ignore=support_data.get('gt_bboxes_ignore',
                                                      None),
            **kwargs)
        losses.update(roi_losses)

        return losses


