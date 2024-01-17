# Copyright (c) OpenMMLab. All rights reserved.
import torch
import pdb

from rsidet.core import bbox2result, bbox2roi, build_assigner, build_sampler
# from ..builder import HEADS, build_head, build_roi_extractor
# from .base_roi_head import BaseRoIHead
# from .test_mixins import BBoxTestMixin, MaskTestMixin
from rsidet.models.builder import HEADS
from rsidet.models.roi_heads import StandardRoIHead


@HEADS.register_module()
class NegRPNRoIHead(StandardRoIHead):
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
                      x,
                      img_metas,
                      proposal_list_pos,
                      proposal_list_neg,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rsidet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            sampling_results_pos = []
            sampling_results_neg = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list_pos[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result_pos = self.bbox_sampler_pos.sample(
                    assign_result,
                    proposal_list_pos[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                assign_result = self.bbox_assigner.assign(
                    proposal_list_neg[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result_neg = self.bbox_sampler_neg.sample(
                    assign_result,
                    proposal_list_neg[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x]
                )

                sampling_result = self._merge_sampling_results(sampling_result_pos, sampling_result_neg)
                sampling_results.append(sampling_result)
                sampling_results_pos.append(sampling_result_pos)
                sampling_results_neg.append(sampling_result_neg)


        losses = dict()

        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        states = self._prepare_states_rcnn(
            img_metas, sampling_results, bbox_results, self.test_cfg,
        )

        states_proposals_pos = self._prepare_states_proposals(
            img_metas, sampling_results_pos, self.test_cfg, postfix='pos'
        )

        states_proposals_neg = self._prepare_states_proposals(
            img_metas, sampling_results_neg, self.test_cfg, postfix='neg'
        )

        states.update(states_proposals_pos)
        states.update(states_proposals_neg)

        losses['states'] = states

        return losses

    def _prepare_states_rcnn(self, img_metas, sampling_results, bbox_results,
                        vis_rcnn_cfg=None, rescale=False):

        det_bboxes, det_labels = self.decode_train_bbox_results(
            img_metas, sampling_results, bbox_results,
            vis_rcnn_cfg,
            rescale=rescale,
        )

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]


        # if not self.with_mask:
        #     return bbox_results
        # else:
        #     segm_results = self.simple_test_mask(
        #         x, img_metas, det_bboxes, det_labels, rescale=rescale)
        #     return list(zip(bbox_results, segm_results))

        states = {
            'else|bbox_results': bbox_results,
        }
        return states


    def _prepare_states_proposals(self, img_metas, proposal_list, vis_rcnn_cfg=None,
                                  rescale=False, postfix=''):

        det_bboxes, det_labels = self.decode_proposals(
            img_metas, proposal_list, None, rescale=rescale,
        )

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]


        # if not self.with_mask:
        #     return bbox_results
        # else:
        #     segm_results = self.simple_test_mask(
        #         x, img_metas, det_bboxes, det_labels, rescale=rescale)
        #     return list(zip(bbox_results, segm_results))

        states = {
            f'else|proposals_{postfix}': bbox_results,
        }
        return states



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
