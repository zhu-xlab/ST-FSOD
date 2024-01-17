# Copyright (c) OpenMMLab. All rights reserved.
import torch
import pdb
from typing import Dict, List, Optional
from torch import Tensor

from rsidet.core import bbox2result, bbox2roi, build_assigner, build_sampler
# from ..builder import HEADS, build_head, build_roi_extractor
# from .base_roi_head import BaseRoIHead
# from .test_mixins import BBoxTestMixin, MaskTestMixin
from rsidet.models.builder import HEADS
from .meta_rcnn_roi_head import MetaRCNNRoIHead


@HEADS.register_module()
class NegRPNMetaRCNNRoIHead(MetaRCNNRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler_pos = None
        self.bbox_sampler_neg = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler_pos = build_sampler(
                self.train_cfg.sampler_pos, context=self)
            self.bbox_sampler_neg = build_sampler(
                self.train_cfg.sampler_neg, context=self)

    def forward_train(self,
                      query_feats: List[Tensor],
                      support_feats: List[Tensor],
                      proposals_pos: List[Tensor],
                      proposals_neg: List[Tensor],
                      query_img_metas: List[Dict],
                      query_gt_bboxes: List[Tensor],
                      query_gt_labels: List[Tensor],
                      support_gt_labels: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            proposals (list[Tensor]): List of region proposals with positive
                and negative pairs.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `rsidet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each
                query image, each item with shape (num_gts, 4)
                in [tl_x, tl_y, br_x, br_y] format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images, each item with shape (num_gts).
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images, each item with shape (1).
            query_gt_bboxes_ignore (list[Tensor] | None): Specify which
                bounding boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """

        # assign gts and sample proposals
        sampling_results = []
        if self.with_bbox:
            num_imgs = len(query_img_metas)
            if query_gt_bboxes_ignore is None:
                query_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            for i in range(num_imgs):
                # assign_result = self.bbox_assigner.assign(
                #     proposals[i], query_gt_bboxes[i],
                #     query_gt_bboxes_ignore[i], query_gt_labels[i])
                # sampling_result = self.bbox_sampler.sample(
                #     assign_result,
                #     proposals[i],
                #     query_gt_bboxes[i],
                #     query_gt_labels[i],
                #     feats=[lvl_feat[i][None] for lvl_feat in query_feats])
                # sampling_results.append(sampling_result)

                assign_result = self.bbox_assigner.assign(
                    proposals_pos[i], query_gt_bboxes[i], query_gt_bboxes_ignore[i],
                    query_gt_labels[i])
                sampling_result_pos = self.bbox_sampler_pos.sample(
                    assign_result,
                    proposals_pos[i],
                    query_gt_bboxes[i],
                    query_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in query_feats])

                assign_result = self.bbox_assigner.assign(
                    proposals_neg[i], query_gt_bboxes[i], query_gt_bboxes_ignore[i],
                    query_gt_labels[i])
                sampling_result_neg = self.bbox_sampler_neg.sample(
                    assign_result,
                    proposals_neg[i],
                    query_gt_bboxes[i],
                    query_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in query_feats])

                sampling_result = self._merge_sampling_results(sampling_result_pos, sampling_result_neg)
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                query_feats, support_feats, sampling_results, query_img_metas,
                query_gt_bboxes, query_gt_labels, support_gt_labels)
            if bbox_results is not None:
                losses.update(bbox_results['loss_bbox'])

        return losses



    def _merge_sampling_results(self, result1, result2):
        import copy
        result = copy.deepcopy(result1)
        # result.pos_bboxes = torch.cat([result1.pos_bboxes, result2.pos_bboxes])
        result.neg_bboxes = torch.cat([result1.neg_bboxes, result2.neg_bboxes])
        # result.pos_inds = torch.cat([result1.pos_inds, result2.pos_inds])
        result.neg_inds = torch.cat([result1.neg_inds, result2.neg_inds])
        # result.pos_assigned_gt_inds = torch.cat([result1.pos_assigned_gt_inds, result2.pos_assigned_gt_inds])
        # result.pos_is_gt = torch.cat([result1.pos_is_gt, result2.pos_is_gt])
        # result.num_gts = result1.num_gts + result2.num_gts

        return result
